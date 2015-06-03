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
		clog(NetNote) << "Initialising: latest=" << m_latestBlockSent;

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

void EthereumHost::changeSyncer(EthereumPeer* _syncer, bool _needHelp)
{
	if (_syncer)
		clog(NetAllDetail) << "Changing syncer to" << _syncer->session()->socketId();
	else
		clog(NetAllDetail) << "Clearing syncer.";

	m_syncer = _syncer;
	if (isSyncing())
	{
		if (_needHelp && _syncer->m_asking == Asking::Blocks)
			for (auto j: peerSessions())
			{
				clog(NetNote) << "Getting help with downloading blocks";
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
		_who->addRating(m_man.chainSize() / 100);
		m_man.reset();
	}
	else if (_who->isSyncing())
	{
		if (_clemency)
			clog(NetNote) << "Chain download failed. Aborted while incomplete.";
		else
		{
			// Done our chain-get.
			clog(NetWarn) << "Chain download failed. Peer with blocks didn't have them all. This peer is bad and should be punished.";
			clog(NetWarn) << m_man.remaining();
			clog(NetWarn) << "WOULD BAN.";
//			m_banned.insert(_who->session()->id());			// We know who you are!
//			_who->disable("Peer sent hashes but was unable to provide the blocks.");
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
		if (m_newTransactions)
		{
			m_newTransactions = false;
			maintainTransactions();
		}
		if (m_newBlocks)
		{
			m_newBlocks = false;
			maintainBlocks(h);
		}
	}

	for (auto p: peerSessions())
		if (shared_ptr<EthereumPeer> const& ep = p.first->cap<EthereumPeer>())
			ep->tick();

//	return netChange;
	// TODO: Figure out what to do with netChange.
	(void)netChange;
}

void EthereumHost::maintainTransactions()
{
	// Send any new transactions.
	unordered_map<std::shared_ptr<EthereumPeer>, h256s> peerTransactions;
	auto ts = m_tq.transactions();
	for (auto const& i: ts)
	{
		bool unsent = !m_transactionsSent.count(i.first);
		for (auto const& p: get<1>(randomSelection(0, [&](EthereumPeer* p) { return p->m_requireTransactions || (unsent && !p->m_knownTransactions.count(i.first)); })))
			peerTransactions[p].push_back(i.first);
	}
	for (auto const& t: ts)
		m_transactionsSent.insert(t.first);
	for (auto p: peerSessions())
		if (auto ep = p.first->cap<EthereumPeer>())
		{
			bytes b;
			unsigned n = 0;
			for (auto const& h: peerTransactions[ep])
			{
				ep->m_knownTransactions.insert(h);
				b += ts[h].rlp();
				++n;
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

tuple<vector<shared_ptr<EthereumPeer>>, vector<shared_ptr<EthereumPeer>>, vector<shared_ptr<Session>>> EthereumHost::randomSelection(unsigned _percent, std::function<bool(EthereumPeer*)> const& _allow)
{
	vector<shared_ptr<EthereumPeer>> chosen;
	vector<shared_ptr<EthereumPeer>> allowed;
	vector<shared_ptr<Session>> sessions;
	
	auto const& ps = peerSessions();
	allowed.reserve(ps.size());
	for (auto const& j: ps)
	{
		auto pp = j.first->cap<EthereumPeer>();
		if (_allow(pp.get()))
		{
			allowed.push_back(move(pp));
			sessions.push_back(move(j.first));
		}
	}

	chosen.reserve((ps.size() * _percent + 99) / 100);
	for (unsigned i = (ps.size() * _percent + 99) / 100; i-- && allowed.size();)
	{
		unsigned n = rand() % allowed.size();
		chosen.push_back(std::move(allowed[n]));
		allowed.erase(allowed.begin() + n);
	}
	return make_tuple(move(chosen), move(allowed), move(sessions));
}

void EthereumHost::maintainBlocks(h256 const& _currentHash)
{
	// Send any new blocks.
	auto detailsFrom = m_chain.details(m_latestBlockSent);
	auto detailsTo = m_chain.details(_currentHash);
	if (detailsFrom.totalDifficulty < detailsTo.totalDifficulty)
	{
		if (diff(detailsFrom.number, detailsTo.number) < 20)
		{
			// don't be sending more than 20 "new" blocks. if there are any more we were probably waaaay behind.
			clog(NetMessageSummary) << "Sending a new block (current is" << _currentHash << ", was" << m_latestBlockSent << ")";

			h256s blocks = get<0>(m_chain.treeRoute(m_latestBlockSent, _currentHash, false, false, true));

			auto s = randomSelection(25, [&](EthereumPeer* p){ DEV_GUARDED(p->x_knownBlocks) return !p->m_knownBlocks.count(_currentHash); return false; });
			for (shared_ptr<EthereumPeer> const& p: get<0>(s))
				for (auto const& b: blocks)
				{
					RLPStream ts;
					p->prep(ts, NewBlockPacket, 2).appendRaw(m_chain.block(b), 1).append(m_chain.details(b).totalDifficulty);

					Guard l(p->x_knownBlocks);
					p->sealAndSend(ts);
					p->m_knownBlocks.clear();
				}
			for (shared_ptr<EthereumPeer> const& p: get<1>(s))
			{
				RLPStream ts;
				p->prep(ts, NewBlockHashesPacket, blocks.size());
				for (auto const& b: blocks)
					ts.append(b);

				Guard l(p->x_knownBlocks);
				p->sealAndSend(ts);
				p->m_knownBlocks.clear();
			}
		}
		m_latestBlockSent = _currentHash;
	}
}
