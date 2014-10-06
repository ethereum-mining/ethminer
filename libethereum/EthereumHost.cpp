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
 * @author Gav Wood <i@gavwood.com>
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
#include "DownloadMan.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

EthereumHost::EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId):
	HostCapability<EthereumPeer>(),
	Worker		("ethsync"),
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

void EthereumHost::notePeerStateChanged(EthereumPeer* _who)
{
	clog(NetAllDetail) << "Peer state changed.";

	// TODO: FIX: BUG: Better state management!

	// if already downloading hash-chain, ignore.
	if (m_grabbing != Asking::Nothing)
	{
		for (auto const& i: peers())
			if (i->cap<EthereumPeer>()->m_grabbing == m_grabbing || m_grabbing == Asking::Presync)
			{
				clog(NetAllDetail) << "Already downloading chain. Just set to help out.";
				_who->ensureGettingChain();
				return;
			}
		m_grabbing = Asking::Nothing;
	}

	// otherwise check to see if we should be downloading...
	_who->tryGrabbingHashChain();
}

void EthereumHost::updateGrabbing(Asking _g)
{
	m_grabbing = _g;
	if (_g == Asking::Nothing)
		readyForSync();
	else if (_g == Asking::Chain)
		for (auto j: peers())
			j->cap<EthereumPeer>()->ensureGettingChain();
}

void EthereumHost::noteHaveChain(EthereumPeer* _from)
{
	auto td = _from->m_totalDifficulty;

	if (_from->m_neededBlocks.empty())
	{
		_from->setGrabbing(Asking::Nothing);
		updateGrabbing(Asking::Nothing);
		return;
	}

	clog(NetNote) << "Hash-chain COMPLETE:" << _from->m_totalDifficulty << "vs" << m_chain.details().totalDifficulty << ";" << _from->m_neededBlocks.size() << " blocks, ends" << _from->m_neededBlocks.back().abridged();

	if (td < m_chain.details().totalDifficulty || (td == m_chain.details().totalDifficulty && m_chain.currentHash() == _from->m_latestHash))
	{
		clog(NetNote) << "Difficulty of hashchain not HIGHER. Ignoring.";
		_from->setGrabbing(Asking::Nothing);
		updateGrabbing(Asking::Nothing);
		return;
	}

	clog(NetNote) << "Difficulty of hashchain HIGHER. Replacing fetch queue [latest now" << _from->m_latestHash.abridged() << ", was" << m_latestBlockSent.abridged() << "]";

	// Looks like it's the best yet for total difficulty. Set to download.
	m_man.resetToChain(_from->m_neededBlocks);
	m_latestBlockSent = _from->m_latestHash;

	_from->setGrabbing(Asking::Chain);
	updateGrabbing(Asking::Chain);
}

void EthereumHost::readyForSync()
{
	// start grabbing next hash chain if there is one.
	for (auto j: peers())
	{
		j->cap<EthereumPeer>()->tryGrabbingHashChain();
		if (j->cap<EthereumPeer>()->m_grabbing == Asking::Hashes)
		{
			m_grabbing = Asking::Hashes;
			return;
		}
	}
	clog(NetNote) << "No more peers to sync with.";
}

void EthereumHost::noteDoneBlocks(EthereumPeer* _who)
{
	if (m_man.isComplete())
	{
		// Done our chain-get.
		clog(NetNote) << "Chain download complete.";
		updateGrabbing(Asking::Nothing);
		m_man.reset();
	}
	if (_who->m_grabbing == Asking::Chain)
	{
		// Done our chain-get.
		clog(NetNote) << "Chain download failed. Peer with blocks didn't have them all. This peer is bad and should be punished.";
		// TODO: note that peer is BADBADBAD!
		updateGrabbing(Asking::Nothing);
		m_man.reset();
	}
}

bool EthereumHost::noteBlock(h256 _hash, bytesConstRef _data)
{
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
	(void)netChange;
}

void EthereumHost::maintainTransactions(TransactionQueue& _tq, h256 _currentHash)
{
	bool resendAll = (m_grabbing == Asking::Nothing && m_chain.isKnown(m_latestBlockSent) && _currentHash != m_latestBlockSent);
	{
		lock_guard<recursive_mutex> l(m_incomingLock);
		for (auto it = m_incomingTransactions.begin(); it != m_incomingTransactions.end(); ++it)
			if (_tq.import(&*it))
			{}//ret = true;		// just putting a transaction in the queue isn't enough to change the state - it might have an invalid nonce...
			else
				m_transactionsSent.insert(sha3(*it));	// if we already had the transaction, then don't bother sending it on.
		m_incomingTransactions.clear();
	}

	// Send any new transactions.
	for (auto const& p: peers())
		if (auto ep = p->cap<EthereumPeer>())
		{
			bytes b;
			unsigned n = 0;
			for (auto const& i: _tq.transactions())
				if ((!m_transactionsSent.count(i.first) && !ep->m_knownTransactions.count(i.first)) || ep->m_requireTransactions || resendAll)
				{
					b += i.second;
					++n;
					m_transactionsSent.insert(i.first);
				}
			ep->clearKnownTransactions();
			
			if (n || ep->m_requireTransactions)
			{
				RLPStream ts;
				EthereumPeer::prep(ts);
				ts.appendList(n + 1) << TransactionsPacket;
				ts.appendRaw(b, n).swapOut(b);
				seal(b);
				ep->send(&b);
			}
			ep->m_requireTransactions = false;
		}
}

void EthereumHost::reset()
{
	m_grabbing = Asking::Nothing;

	m_man.resetToChain(h256s());

	m_incomingTransactions.clear();
	m_incomingBlocks.clear();

	m_latestBlockSent = h256();
	m_transactionsSent.clear();
}

void EthereumHost::maintainBlocks(BlockQueue& _bq, h256 _currentHash)
{
	// If we've finished our initial sync send any new blocks.
	if (m_grabbing == Asking::Nothing && m_chain.isKnown(m_latestBlockSent) && m_chain.details(m_latestBlockSent).totalDifficulty < m_chain.details(_currentHash).totalDifficulty)
	{
		// TODO: clean up
		h256s hs;
		hs.push_back(_currentHash);
		RLPStream ts;
		EthereumPeer::prep(ts);
		bytes bs;
		for (auto h: hs)
			bs += m_chain.block(h);
		clog(NetMessageSummary) << "Sending" << hs.size() << "new blocks (current is" << _currentHash << ", was" << m_latestBlockSent << ")";

		ts.appendList(1 + hs.size()).append(BlocksPacket).appendRaw(bs, hs.size());
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
