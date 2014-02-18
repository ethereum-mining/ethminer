/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Client.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Client.h"

#include <chrono>
#include <thread>
#include "Common.h"
#include "Defaults.h"
using namespace std;
using namespace eth;

Client::Client(std::string const& _clientVersion, Address _us, std::string const& _dbPath):
	m_clientVersion(_clientVersion),
	m_bc(_dbPath),
	m_stateDB(State::openDB(_dbPath)),
	m_preMine(_us, m_stateDB),
	m_postMine(_us, m_stateDB)
{
	Defaults::setDBPath(_dbPath);

	// Synchronise the state according to the head of the block chain.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	m_preMine.sync(m_bc);
	m_postMine = m_preMine;
	m_changed = true;

	static const char* c_threadName = "eth";

	m_work = new thread([&](){
		setThreadName(c_threadName);

		while (m_workState != Deleting) work(); m_workState = Deleted;
	});
}

Client::~Client()
{
	if (m_workState == Active)
		m_workState = Deleting;
	while (m_workState != Deleted)
		this_thread::sleep_for(chrono::milliseconds(10));
}

void Client::startNetwork(unsigned short _listenPort, std::string const& _seedHost, unsigned short _port, NodeMode _mode, unsigned _peers, string const& _publicIP, bool _upnp)
{
	if (m_net)
		return;
	m_net = new PeerServer(m_clientVersion, m_bc, 0, _listenPort, _mode, _publicIP, _upnp);
	m_net->setIdealPeerCount(_peers);
	if (_seedHost.size())
		connect(_seedHost, _port);
}

void Client::connect(std::string const& _seedHost, unsigned short _port)
{
	if (!m_net)
		return;
	m_net->connect(_seedHost, _port);
}

void Client::stopNetwork()
{
	delete m_net;
	m_net = nullptr;
}

void Client::startMining()
{
	m_doMine = true;
	m_restartMining = true;
}

void Client::stopMining()
{
	m_doMine = false;
}

void Client::transact(Secret _secret, Address _dest, u256 _amount, u256s _data)
{
	lock_guard<recursive_mutex> l(m_lock);
	Transaction t;
	t.nonce = m_postMine.transactionsFrom(toAddress(_secret));
	t.receiveAddress = _dest;
	t.value = _amount;
	t.data = _data;
	t.sign(_secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
	m_changed = true;
}

void Client::work()
{
	bool changed = false;

	// Process network events.
	// Synchronise block chain with network.
	// Will broadcast any of our (new) transactions and blocks, and collect & add any of their (new) transactions and blocks.
	if (m_net)
	{
		m_net->process();

		lock_guard<recursive_mutex> l(m_lock);
		if (m_net->sync(m_bc, m_tq, m_stateDB))
			changed = true;
	}

	// Synchronise state to block chain.
	// This should remove any transactions on our queue that are included within our state.
	// It also guarantees that the state reflects the longest (valid!) chain on the block chain.
	//   This might mean reverting to an earlier state and replaying some blocks, or, (worst-case:
	//   if there are no checkpoints before our fork) reverting to the genesis block and replaying
	//   all blocks.
	// Resynchronise state with block chain & trans
	{
		lock_guard<recursive_mutex> l(m_lock);
		if (m_preMine.sync(m_bc) || m_postMine.address() != m_preMine.address())
		{
			if (m_doMine)
				cnote << "New block on chain: Restarting mining operation.";
			changed = true;
			m_restartMining = true;	// need to re-commit to mine.
			m_postMine = m_preMine;
		}
		if (m_postMine.sync(m_tq))
		{
			if (m_doMine)
				cnote << "Additional transaction ready: Restarting mining operation.";
			changed = true;
			m_restartMining = true;
		}
	}

	if (m_doMine)
	{
		if (m_restartMining)
		{
			lock_guard<recursive_mutex> l(m_lock);
			m_postMine.commitToMine(m_bc);
		}

		m_restartMining = false;

		// Mine for a while.
		MineInfo mineInfo = m_postMine.mine(100);
		m_mineProgress.best = max(m_mineProgress.best, mineInfo.best);
		m_mineProgress.current = mineInfo.best;
		m_mineProgress.requirement = mineInfo.requirement;

		if (mineInfo.completed)
		{
			// Import block.
			lock_guard<recursive_mutex> l(m_lock);
			m_bc.attemptImport(m_postMine.blockData(), m_stateDB);
			m_mineProgress.best = 0;
			m_changed = true;
		}
	}
	else
		this_thread::sleep_for(chrono::milliseconds(100));

	m_changed = m_changed || changed;
}

void Client::lock()
{
	m_lock.lock();
}

void Client::unlock()
{
	m_lock.unlock();
}
