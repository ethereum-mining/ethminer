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
#include <libethcore/Params.h>
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "BlockQueue.h"
#include "EthereumPeer.h"
#include "DownloadMan.h"
#include "BlockChainSync.h"

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

unsigned const EthereumHost::c_oldProtocolVersion = 60; //TODO: remove this once v61+ is common
unsigned const c_chainReorgSize = 30000;

char const* const EthereumHost::s_stateNames[static_cast<int>(SyncState::Size)] = {"Idle", "Waiting", "Hashes", "Blocks", "NewBlocks" };

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

void EthereumHost::reset()
{
	Guard l(x_sync);
	if (m_sync)
		m_sync->abortSync();
	m_sync.reset();

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

	foreachPeer([](EthereumPeer* _p) { _p->tick(); return true; });

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
		auto peers = get<1>(randomSelection(0, [&](EthereumPeer* p) { return p->m_requireTransactions || (unsent && !p->m_knownTransactions.count(i.first)); }));
		for (auto const& p: peers)
			peerTransactions[p].push_back(i.first);
	}
	for (auto const& t: ts)
		m_transactionsSent.insert(t.first);
	foreachPeerPtr([&](shared_ptr<EthereumPeer> _p)
	{
		bytes b;
		unsigned n = 0;
		for (auto const& h: peerTransactions[_p])
		{
			_p->m_knownTransactions.insert(h);
			b += ts[h].rlp();
			++n;
		}

		_p->clearKnownTransactions();

		if (n || _p->m_requireTransactions)
		{
			RLPStream ts;
			_p->prep(ts, TransactionsPacket, n).appendRaw(b, n);
			_p->sealAndSend(ts);
			cnote << "Sent" << n << "transactions to " << _p->session()->info().clientVersion;
		}
		_p->m_requireTransactions = false;
		return true;
	});
}

void EthereumHost::foreachPeer(std::function<bool(EthereumPeer*)> const& _f) const
{
	foreachPeerPtr([&](std::shared_ptr<EthereumPeer> _p)
	{
		if (_p)
			return _f(_p.get());
		return true;
	});
}

void EthereumHost::foreachPeerPtr(std::function<bool(std::shared_ptr<EthereumPeer>)> const& _f) const
{
	for (auto s: peerSessions())
		if (!_f(s.first->cap<EthereumPeer>()))
			return;
	for (auto s: peerSessions(c_oldProtocolVersion)) //TODO: remove once v61+ is common
		if (!_f(s.first->cap<EthereumPeer>(c_oldProtocolVersion)))
			return;
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

BlockChainSync& EthereumHost::sync()
{
	if (m_sync)
		return *m_sync; // We only chose sync strategy once

	bool pv61 = false;
	foreachPeer([&](EthereumPeer* _p)
	{
		if (_p->m_protocolVersion == protocolVersion())
			pv61 = true;
		return !pv61;
	});
	m_sync.reset(pv61 ? new PV60Sync(*this) : new PV60Sync(*this));
	return *m_sync;
}

void EthereumHost::onPeerStatus(EthereumPeer* _peer)
{
	Guard l(x_sync);
	sync().onPeerStatus(_peer);
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	Guard l(x_sync);
	sync().onPeerHashes(_peer, _hashes);
}

void EthereumHost::onPeerBlocks(EthereumPeer* _peer, RLP const& _r)
{
	Guard l(x_sync);
	sync().onPeerBlocks(_peer, _r);
}

void EthereumHost::onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	Guard l(x_sync);
	sync().onPeerNewHashes(_peer, _hashes);
}

void EthereumHost::onPeerNewBlock(EthereumPeer* _peer, RLP const& _r)
{
	Guard l(x_sync);
	sync().onPeerNewBlock(_peer, _r);
}

void EthereumHost::onPeerTransactions(EthereumPeer* _peer, RLP const& _r)
{
	if (_peer->isCriticalSyncing())
	{
		clog(NetAllDetail) << "Ignoring transaction from peer we are syncing with";
		return;
	}
	unsigned itemCount = _r.itemCount();
	clog(NetAllDetail) << "Transactions (" << dec << itemCount << "entries)";
	Guard l(_peer->x_knownTransactions);
	for (unsigned i = 0; i < min<unsigned>(itemCount, 256); ++i)	// process 256 transactions at most. TODO: much better solution.
	{
		auto h = sha3(_r[i].data());
		_peer->m_knownTransactions.insert(h);
		ImportResult ir = m_tq.import(_r[i].data());
		switch (ir)
		{
		case ImportResult::Malformed:
			_peer->addRating(-100);
			break;
		case ImportResult::AlreadyKnown:
			// if we already had the transaction, then don't bother sending it on.
			m_transactionsSent.insert(h);
			_peer->addRating(0);
			break;
		case ImportResult::Success:
			_peer->addRating(100);
			break;
		default:;
		}
	}
}

void EthereumHost::onPeerAborting(EthereumPeer* _peer)
{
	Guard l(x_sync);
	if (m_sync)
		m_sync->onPeerAborting(_peer);
}

bool EthereumHost::isSyncing() const
{
	Guard l(x_sync);
	if (!m_sync)
		return false;
	return m_sync->isSyncing();
}

SyncStatus EthereumHost::status() const
{
	Guard l(x_sync);
	if (!m_sync)
		return SyncStatus();
	return m_sync->status();
}
