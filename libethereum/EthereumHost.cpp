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
		i->cap<EthereumPeer>()->abortSync();
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

void EthereumHost::noteNeedsSyncing(EthereumPeer* _who)
{
	// if already downloading hash-chain, ignore.
	if (isSyncing())
	{
		clog(NetAllDetail) << "Sync in progress: Just set to help out.";
		if (m_syncer->m_asking == Asking::Blocks)
			_who->transition(Asking::Blocks);
	}
	else
		// otherwise check to see if we should be downloading...
		_who->attemptSync();
}

void EthereumHost::updateSyncer(EthereumPeer* _syncer)
{
	if (_syncer)
	{
		for (auto j: peers())
			if (j->cap<EthereumPeer>().get() != _syncer && j->cap<EthereumPeer>()->m_asking == Asking::Nothing)
				j->cap<EthereumPeer>()->transition(Asking::Blocks);
	}
	else
	{
		// start grabbing next hash chain if there is one.
		for (auto j: peers())
		{
			j->cap<EthereumPeer>()->attemptSync();
			if (isSyncing())
				return;
		}
		clog(NetNote) << "No more peers to sync with.";
	}
}

void EthereumHost::noteDoneBlocks(EthereumPeer* _who, bool _clemency)
{
	if (m_man.isComplete())
	{
		// Done our chain-get.
		clog(NetNote) << "Chain download complete.";
		// 1/100th for each useful block hash.
		_who->addRating(m_man.chain().size() / 100);
		m_man.reset();
	}
	if (_who->isSyncing())
	{
		if (_clemency)
			clog(NetNote) << "Chain download failed. Aborted while incomplete.";
		else
		{
			// Done our chain-get.
			clog(NetNote) << "Chain download failed. Peer with blocks didn't have them all. This peer is bad and should be punished.";

			m_banned.insert(_who->session()->id());			// We know who you are!
			_who->disable("Peer sent hashes but was unable to provide the blocks.");
		}
		m_man.reset();
	}
}

void EthereumHost::reset()
{
	if (m_syncer)
		m_syncer->abortSync();

	m_man.resetToChain(h256s());

	m_latestBlockSent = h256();
	m_transactionsSent.clear();
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
	bool resendAll = (!isSyncing() && m_chain.isKnown(m_latestBlockSent) && _currentHash != m_latestBlockSent);

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

void EthereumHost::maintainBlocks(BlockQueue& _bq, h256 _currentHash)
{
	// If we've finished our initial sync send any new blocks.
	if (!isSyncing() && m_chain.isKnown(m_latestBlockSent) && m_chain.details(m_latestBlockSent).totalDifficulty < m_chain.details(_currentHash).totalDifficulty)
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
