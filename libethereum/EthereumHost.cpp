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
	for (auto i: peerSessions())
		i.first->cap<EthereumPeer>().get()->abortSync();
}

bool EthereumHost::ensureInitialised()
{
	if (!m_latestBlockSent)
	{
		// First time - just initialise.
		m_latestBlockSent = m_chain.currentHash();
		clog(NetNote) << "Initialising: latest=" << m_latestBlockSent.abridged();

		for (auto const& i: m_tq.transactions())
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

void EthereumHost::changeSyncer(EthereumPeer* _syncer)
{
	if (_syncer)
		clog(NetAllDetail) << "Changing syncer to" << _syncer->session()->socketId();
	else
		clog(NetAllDetail) << "Clearing syncer.";

	m_syncer = _syncer;
	if (isSyncing())
	{
		if (_syncer->m_asking == Asking::Blocks)
			for (auto j: peerSessions())
			{
				auto e = j.first->cap<EthereumPeer>().get();
				if (e != _syncer && e->m_asking == Asking::Nothing)
					e->transition(Asking::Blocks);
			}
	}
	else
	{
		// start grabbing next hash chain if there is one.
		for (auto j: peerSessions())
		{
			j.first->cap<EthereumPeer>()->attemptSync();
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
	else if (_who->isSyncing())
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
	bool netChange = ensureInitialised();
	auto h = m_chain.currentHash();
	// If we've finished our initial sync (including getting all the blocks into the chain so as to reduce invalid transactions), start trading transactions & blocks
	if (!isSyncing() && m_chain.isKnown(m_latestBlockSent))
	{
		maintainTransactions();
		maintainBlocks(h);
	}
//	return netChange;
	// TODO: Figure out what to do with netChange.
	(void)netChange;
}

void EthereumHost::maintainTransactions()
{
	// Send any new transactions.
	for (auto p: peerSessions())
		if (auto ep = p.first->cap<EthereumPeer>().get())
		{
			bytes b;
			unsigned n = 0;
			for (auto const& i: m_tq.transactions())
				if (ep->m_requireTransactions || (!m_transactionsSent.count(i.first) && !ep->m_knownTransactions.count(i.first)))
				{
					b += i.second;
					++n;
					m_transactionsSent.insert(i.first);
				}
			ep->clearKnownTransactions();

			if (n || ep->m_requireTransactions)
			{
				RLPStream ts;
				ep->prep(ts, TransactionsPacket, n).appendRaw(b, n);
				ep->sealAndSend(ts);
			}
			ep->m_requireTransactions = false;
		}
}

void EthereumHost::maintainBlocks(h256 _currentHash)
{
	// Send any new blocks.
	if (m_chain.details(m_latestBlockSent).totalDifficulty < m_chain.details(_currentHash).totalDifficulty)
	{
		clog(NetMessageSummary) << "Sending a new block (current is" << _currentHash << ", was" << m_latestBlockSent << ")";

		for (auto j: peerSessions())
		{
			auto p = j.first->cap<EthereumPeer>().get();

			RLPStream ts;
			p->prep(ts, NewBlockPacket, 2).appendRaw(m_chain.block(), 1).append(m_chain.details().totalDifficulty);

			Guard l(p->x_knownBlocks);
			if (!p->m_knownBlocks.count(_currentHash))
				p->sealAndSend(ts);
			p->m_knownBlocks.clear();
		}
		m_latestBlockSent = _currentHash;
	}
}
