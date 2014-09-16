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
/** @file EthereumHost.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Eric Lombrozo <elombrozo@gmail.com>
 * @date 2014
 */

#include "EthereumHost.h"

#include <set>
#include <chrono>
#include <thread>
#include <libdevcore/Common.h>
#include <libp2p/Host.h>
#include <libp2p/Session.h>
#include <libethcore/Exceptions.h>
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "BlockQueue.h"
#include "EthereumPeer.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

EthereumHost::EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId):
	HostCapability<EthereumPeer>(),
	Worker("ethsync"),
	m_chain		(_ch),
	m_tq		(_tq),
	m_bq		(_bq),
	m_networkId	(_networkId)
{
	m_latestBlockSent = _ch.currentHash();
}

EthereumHost::~EthereumHost()
{
	for (auto const& i: peers())
		i->cap<EthereumPeer>()->giveUpOnFetch();
}

h256Set EthereumHost::neededBlocks(h256Set const& _exclude)
{
	Guard l(x_blocksNeeded);
	h256Set ret;
	if (m_blocksNeeded.size())
	{
		int s = m_blocksNeeded.size() - 1;
		for (; ret.size() < c_maxBlocksAsk && s < (int)m_blocksNeeded.size() && s >= 0; --s)
			if (!_exclude.count(m_blocksNeeded[s]))
			{
				auto it = m_blocksNeeded.begin() + s;
				ret.insert(*it);
				m_blocksOnWay.insert(*it);
				m_blocksNeeded.erase(it);
			}
	}
	if (!ret.size())
		for (auto i = m_blocksOnWay.begin(); ret.size() < c_maxBlocksAsk && i != m_blocksOnWay.end() && !_exclude.count(*i); ++i)
			ret.insert(*i);

	clog(NetMessageSummary) << "Asking for" << ret.size() << "blocks that we don't yet have." << m_blocksNeeded.size() << "blocks still needed," << m_blocksOnWay.size() << "blocks on way.";
	return ret;
}

bool EthereumHost::ensureInitialised(TransactionQueue& _tq)
{
	if (!m_latestBlockSent)
	{
		// First time - just initialise.
		m_latestBlockSent = m_chain.currentHash();
		clog(NetNote) << "Initialising: latest=" << m_latestBlockSent.abridged();

		for (auto const& i: _tq.transactions())
			m_transactionsSent.insert(i.first);
		return true;
	}
	return false;
}

void EthereumHost::noteDoneBlocks()
{
	clog(NetNote) << "Peer given up on blocks fetch.";
	if (m_blocksOnWay.empty())
	{
		// Done our chain-get.
		if (m_blocksNeeded.size())
			clog(NetNote) << "No more blocks coming. Missing" << m_blocksNeeded.size() << "blocks.";
		else
			clog(NetNote) << "No more blocks to get.";
		m_latestBlockSent = m_chain.currentHash();
	}
}

bool EthereumHost::noteBlock(h256 _hash, bytesConstRef _data)
{
	Guard l(x_blocksNeeded);
	m_blocksOnWay.erase(_hash);
	if (!m_chain.details(_hash))
	{
		lock_guard<recursive_mutex> l(m_incomingLock);
		m_incomingBlocks.push_back(_data.toBytes());
		return true;
	}
	return false;
}

void EthereumHost::doWork()
{
	bool netChange = ensureInitialised(m_tq);
	auto h = m_chain.currentHash();
	maintainTransactions(m_tq, h);
	maintainBlocks(m_bq, h);
//	return netChange;
	// TODO: Figure out what to do with netChange.
}

void EthereumHost::maintainTransactions(TransactionQueue& _tq, h256 _currentHash)
{
	bool resendAll = (_currentHash != m_latestBlockSent);

	for (auto it = m_incomingTransactions.begin(); it != m_incomingTransactions.end(); ++it)
		if (_tq.import(&*it))
		{}//ret = true;		// just putting a transaction in the queue isn't enough to change the state - it might have an invalid nonce...
		else
			m_transactionsSent.insert(sha3(*it));	// if we already had the transaction, then don't bother sending it on.
	m_incomingTransactions.clear();

	// Send any new transactions.
	for (auto const& p: peers())
	{
		auto ep = p->cap<EthereumPeer>();
		bytes b;
		unsigned n = 0;
		for (auto const& i: _tq.transactions())
			if ((!m_transactionsSent.count(i.first) && !ep->m_knownTransactions.count(i.first)) || ep->m_requireTransactions || resendAll)
			{
				b += i.second;
				++n;
				m_transactionsSent.insert(i.first);
			}
		if (n)
		{
			RLPStream ts;
			EthereumPeer::prep(ts);
			ts.appendList(n + 1) << TransactionsPacket;
			ts.appendRaw(b, n).swapOut(b);
			seal(b);
			ep->send(&b);
		}
		ep->m_knownTransactions.clear();
		ep->m_requireTransactions = false;
	}
}

void EthereumHost::maintainBlocks(BlockQueue& _bq, h256 _currentHash)
{
	// Import new blocks
	{
		lock_guard<recursive_mutex> l(m_incomingLock);
		for (auto it = m_incomingBlocks.rbegin(); it != m_incomingBlocks.rend(); ++it)
			if (_bq.import(&*it, m_chain))
			{}
			else{} // TODO: don't forward it.
		m_incomingBlocks.clear();
	}

	// If we've finished our initial sync...
	{
		Guard l(x_blocksNeeded);
		if (m_blocksOnWay.size())
			return;
	}
	// ...send any new blocks.
	if (m_latestBlockSent != _currentHash)
	{
		RLPStream ts;
		EthereumPeer::prep(ts);
		bytes bs;
		unsigned c = 0;
		for (auto h: m_chain.treeRoute(m_latestBlockSent, _currentHash, nullptr, false, true))
		{
			bs += m_chain.block(h);
			++c;
		}
		clog(NetMessageSummary) << "Sending" << c << "new blocks (current is" << _currentHash << ", was" << m_latestBlockSent << ")";
		ts.appendList(1 + c).append(BlocksPacket).appendRaw(bs, c);
		bytes b;
		ts.swapOut(b);
		seal(b);

		for (auto j: peers())
		{
			auto p = j->cap<EthereumPeer>();
			Guard l(p->x_knownBlocks);
			if (!p->m_knownBlocks.count(_currentHash))
				p->send(&b);
			p->m_knownBlocks.clear();
		}
		m_latestBlockSent = _currentHash;
	}
}

void EthereumHost::noteHaveChain(EthereumPeer* _from)
{
	auto td = _from->m_totalDifficulty;

	if (_from->m_neededBlocks.empty())
		return;

	clog(NetNote) << "Hash-chain COMPLETE:" << _from->m_totalDifficulty << "vs" << m_chain.details().totalDifficulty << "," << m_totalDifficultyOfNeeded << ";" << _from->m_neededBlocks.size() << " blocks, ends" << _from->m_neededBlocks.back().abridged();

	if ((m_totalDifficultyOfNeeded && td < m_totalDifficultyOfNeeded) || td < m_chain.details().totalDifficulty)
	{
		clog(NetNote) << "Difficulty of hashchain LOWER. Ignoring.";
		return;
	}

	clog(NetNote) << "Difficulty of hashchain HIGHER. Replacing fetch queue.";

	// Looks like it's the best yet for total difficulty. Set to download.
	{
		Guard l(x_blocksNeeded);
		m_blocksNeeded = _from->m_neededBlocks;
		m_blocksOnWay.clear();
		m_totalDifficultyOfNeeded = td;
	}

	for (auto j: peers())
		j->cap<EthereumPeer>()->restartGettingChain();
}
