/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Client.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "Client.h"
using namespace std;
using namespace eth;

Client::Client(std::string const& _dbPath):
	m_bc(_dbPath),
	m_stateDB(State::openDB(_dbPath)),
	m_s(m_stateDB)
{
	Defaults::setDBPath(_dbPath);

	// Synchronise the state according to the block chain - i.e. replay all transactions in block chain, in order.
	// In practise this won't need to be done since the State DB will contain the keys for the tries for most recent (and many old) blocks.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	s.sync(bc);
	s.sync(tq);

	m_work = new thread([&](){ while (m_workState != Deleting) work(); m_workState = Deleted; });
}

Client::~Client()
{
	if (m_workState == Active)
		m_workState = Deleting;
	while (m_workState != Deleted)
		usleep(10000);
}

void Client::transact(Address _dest, u256 _amount, u256 _fee, u256s _data = u256s(), Secret _secret)
{
}

BlockChain const& Client::blockChain() const
{
}

TransactionQueue const& Client::transactionQueue() const
{
}

unsigned Client::peerCount() const
{
}

void Client::startNetwork(short _listenPort = 30303, std::string const& _seedHost, short _port = 30303)
{
	if (m_net)
		return;
	m_net = new PeerServer(m_bc, 0, _listenPort);
	if (_seedHost.size())
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
}

void Client::stopMining()
{
	m_doMine = false;
}

std::pair<unsigned, unsigned> Client::miningProgress() const
{
}

void Client::work(string const& _seedHost, short _port)
{
	// Process network events.
	// Synchronise block chain with network.
	// Will broadcast any of our (new) transactions and blocks, and collect & add any of their (new) transactions and blocks.
	m_net->process(m_bc, m_tq);

	// Synchronise state to block chain.
	// This should remove any transactions on our queue that are included within our state.
	// It also guarantees that the state reflects the longest (valid!) chain on the block chain.
	//   This might mean reverting to an earlier state and replaying some blocks, or, (worst-case:
	//   if there are no checkpoints before our fork) reverting to the genesis block and replaying
	//   all blocks.
	m_s.sync(m_bc);		// Resynchronise state with block chain & trans
	m_s.sync(m_tq);

	if (m_doMine)
	{
		// Mine for a while.
		bytes b = s.mine(100);

		if (b.size())
			// Import block.
			bc.attemptImport(b, stateDB);
	}
}
