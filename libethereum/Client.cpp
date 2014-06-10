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
#include <boost/filesystem.hpp>
#include <libethsupport/Common.h>
#include "Defaults.h"
#include "PeerServer.h"
using namespace std;
using namespace eth;

VersionChecker::VersionChecker(string const& _dbPath, unsigned _protocolVersion):
	m_path(_dbPath.size() ? _dbPath : Defaults::dbPath()),
	m_protocolVersion(_protocolVersion)
{
	m_ok = RLP(contents(m_path + "/protocol")).toInt<unsigned>(RLP::LaisezFaire) == _protocolVersion;
}

void VersionChecker::setOk()
{
	if (!m_ok)
	{
		try
		{
			boost::filesystem::create_directory(m_path);
		}
		catch (...) {}
		writeFile(m_path + "/protocol", rlp(m_protocolVersion));
	}
}

Client::Client(std::string const& _clientVersion, Address _us, std::string const& _dbPath, bool _forceClean):
	m_clientVersion(_clientVersion),
	m_vc(_dbPath, PeerServer::protocolVersion()),
	m_bc(_dbPath, !m_vc.ok() || _forceClean),
	m_stateDB(State::openDB(_dbPath, !m_vc.ok() || _forceClean)),
	m_preMine(_us, m_stateDB),
	m_postMine(_us, m_stateDB),
	m_workState(Active)
{
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
	m_vc.setOk();
	m_changed = true;

	static const char* c_threadName = "eth";

	m_work.reset(new thread([&](){
		setThreadName(c_threadName);
		while (m_workState.load(std::memory_order_acquire) != Deleting)
			work();
		m_workState.store(Deleted, std::memory_order_release);

		// Synchronise the state according to the head of the block chain.
		// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
		m_preMine.sync(m_bc);
		m_postMine = m_preMine;
	}));
}

Client::~Client()
{
	if (m_workState.load(std::memory_order_acquire) == Active)
		m_workState.store(Deleting, std::memory_order_release);
	while (m_workState.load(std::memory_order_acquire) != Deleted)
		this_thread::sleep_for(chrono::milliseconds(10));
	m_work->join();
}

void Client::startNetwork(unsigned short _listenPort, std::string const& _seedHost, unsigned short _port, NodeMode _mode, unsigned _peers, string const& _publicIP, bool _upnp)
{
	if (m_net.get())
		return;
	try
	{
		m_net.reset(new PeerServer(m_clientVersion, m_bc, 0, _listenPort, _mode, _publicIP, _upnp));
	}
	catch (std::exception const&)
	{
		// Probably already have the port open.
		cwarn << "Could not initialize with specified/default port. Trying system-assigned port";
		m_net.reset(new PeerServer(m_clientVersion, m_bc, 0, _mode, _publicIP, _upnp));
	}

	m_net->setIdealPeerCount(_peers);
	if (_seedHost.size())
		connect(_seedHost, _port);
}

std::vector<PeerInfo> Client::peers()
{
	return m_net ? m_net->peers() : std::vector<PeerInfo>();
}

size_t Client::peerCount() const
{
	return m_net ? m_net->peerCount() : 0;
}

void Client::connect(std::string const& _seedHost, unsigned short _port)
{
	if (!m_net.get())
		return;
	m_net->connect(_seedHost, _port);
}

void Client::stopNetwork()
{
	m_net.reset(nullptr);
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

void Client::transact(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	lock_guard<recursive_mutex> l(m_lock);
	Transaction t;
	t.nonce = m_postMine.transactionsFrom(toAddress(_secret));
	t.value = _value;
	t.gasPrice = _gasPrice;
	t.gas = _gas;
	t.receiveAddress = _dest;
	t.data = _data;
	t.sign(_secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
}

Address Client::transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	lock_guard<recursive_mutex> l(m_lock);
	Transaction t;
	t.nonce = m_postMine.transactionsFrom(toAddress(_secret));
	t.value = _endowment;
	t.gasPrice = _gasPrice;
	t.gas = _gas;
	t.receiveAddress = Address();
	t.data = _init;
	t.sign(_secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
	return right160(sha3(rlpList(t.sender(), t.nonce)));
}

void Client::inject(bytesConstRef _rlp)
{
	lock_guard<recursive_mutex> l(m_lock);
	m_tq.attemptImport(_rlp);
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
		if (m_postMine.sync(m_tq, &changed))
		{
			if (m_doMine)
				cnote << "Additional transaction ready: Restarting mining operation.";
			m_restartMining = true;
		}
	}

	if (m_doMine)
	{
		if (m_restartMining)
		{
			m_mineProgress.best = (double)-1;
			m_mineProgress.hashes = 0;
			m_mineProgress.ms = 0;
			lock_guard<recursive_mutex> l(m_lock);
			if (m_paranoia)
			{
				if (m_postMine.amIJustParanoid(m_bc))
				{
					cnote << "I'm just paranoid. Block is fine.";
					m_postMine.commitToMine(m_bc);
				}
				else
				{
					cwarn << "I'm not just paranoid. Cannot mine. Please file a bug report.";
					m_doMine = false;
				}
			}
			else
				m_postMine.commitToMine(m_bc);
		}
	}

	if (m_doMine)
	{
		m_restartMining = false;

		// Mine for a while.
		MineInfo mineInfo = m_postMine.mine(100);

		m_mineProgress.best = min(m_mineProgress.best, mineInfo.best);
		m_mineProgress.current = mineInfo.best;
		m_mineProgress.requirement = mineInfo.requirement;
		m_mineProgress.ms += 100;
		m_mineProgress.hashes += mineInfo.hashes;
		{
			lock_guard<recursive_mutex> l(m_lock);
			m_mineHistory.push_back(mineInfo);
		}

		if (mineInfo.completed)
		{
			// Import block.
			lock_guard<recursive_mutex> l(m_lock);
			m_bc.attemptImport(m_postMine.blockData(), m_stateDB);
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
