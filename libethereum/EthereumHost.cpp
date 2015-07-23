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
static unsigned const c_maxSendTransactions = 256;

char const* const EthereumHost::s_stateNames[static_cast<int>(SyncState::Size)] = {"Idle", "Waiting", "Hashes", "Blocks", "NewBlocks" };

#ifdef _WIN32
const char* EthereumHostTrace::name() { return EthPurple "^" EthGray "  "; }
#else
const char* EthereumHostTrace::name() { return EthPurple "⧫" EthGray " "; }
#endif

EthereumHost::EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId):
	HostCapability<EthereumPeer>(),
	Worker		("ethsync"),
	m_chain		(_ch),
	m_tq		(_tq),
	m_bq		(_bq),
	m_networkId	(_networkId)
{
	m_latestBlockSent = _ch.currentHash();
	m_tq.onImport([this](ImportResult _ir, h256 const& _h, h512 const& _nodeId) { onTransactionImported(_ir, _h, _nodeId); });
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
		clog(EthereumHostTrace) << "Initialising: latest=" << m_latestBlockSent;

		Guard l(x_transactions);
		m_transactionsSent = m_tq.knownTransactions();
		return true;
	}
	return false;
}

void EthereumHost::reset()
{
	RecursiveGuard l(x_sync);
	if (m_sync)
		m_sync->abortSync();
	m_sync.reset();
	m_syncStart = 0;

	m_latestBlockSent = h256();
	Guard tl(x_transactions);
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

	foreachPeer([](std::shared_ptr<EthereumPeer> _p) { _p->tick(); return true; });

	if (m_syncStart)
	{
		DEV_RECURSIVE_GUARDED(x_sync)
			if (!m_sync)
			{
				time_t now = std::chrono::system_clock::to_time_t(chrono::system_clock::now());
				if (now - m_syncStart > 10)
				{
					m_sync.reset(new PV60Sync(*this));
					m_syncStart = 0;
					m_sync->restartSync();
				}
			}
	}

//	return netChange;
	// TODO: Figure out what to do with netChange.
	(void)netChange;
}

void EthereumHost::maintainTransactions()
{
	// Send any new transactions.
	unordered_map<std::shared_ptr<EthereumPeer>, std::vector<size_t>> peerTransactions;
	auto ts = m_tq.topTransactions(c_maxSendTransactions);
	{
		Guard l(x_transactions);
		for (size_t i = 0; i < ts.size(); ++i)
		{
			auto const& t = ts[i];
			bool unsent = !m_transactionsSent.count(t.sha3());
			auto peers = get<1>(randomSelection(0, [&](EthereumPeer* p) { return p->m_requireTransactions || (unsent && !p->m_knownTransactions.count(t.sha3())); }));
			for (auto const& p: peers)
				peerTransactions[p].push_back(i);
		}
		for (auto const& t: ts)
			m_transactionsSent.insert(t.sha3());
	}
	foreachPeer([&](shared_ptr<EthereumPeer> _p)
	{
		bytes b;
		unsigned n = 0;
		for (auto const& i: peerTransactions[_p])
		{
			_p->m_knownTransactions.insert(ts[i].sha3());
			b += ts[i].rlp();
			++n;
		}

		_p->clearKnownTransactions();

		if (n || _p->m_requireTransactions)
		{
			RLPStream ts;
			_p->prep(ts, TransactionsPacket, n).appendRaw(b, n);
			_p->sealAndSend(ts);
			clog(EthereumHostTrace) << "Sent" << n << "transactions to " << _p->session()->info().clientVersion;
		}
		_p->m_requireTransactions = false;
		return true;
	});
}

void EthereumHost::foreachPeer(std::function<bool(std::shared_ptr<EthereumPeer>)> const& _f) const
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
	
	size_t peerCount = 0;
	foreachPeer([&](std::shared_ptr<EthereumPeer> _p)
	{
		if (_allow(_p.get()))
		{
			allowed.push_back(_p);
			sessions.push_back(_p->session());
		}
		++peerCount;
		return true;
	});

	size_t chosenSize = (peerCount * _percent + 99) / 100;
	chosen.reserve(chosenSize);
	for (unsigned i = chosenSize; i && allowed.size(); i--)
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
			clog(EthereumHostTrace) << "Sending a new block (current is" << _currentHash << ", was" << m_latestBlockSent << ")";

			h256s blocks = get<0>(m_chain.treeRoute(m_latestBlockSent, _currentHash, false, false, true));

			auto s = randomSelection(25, [&](EthereumPeer* p){
				DEV_GUARDED(p->x_knownBlocks)
					return !p->m_knownBlocks.count(_currentHash);
				return false;
			});
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

BlockChainSync* EthereumHost::sync()
{
	if (m_sync)
		return m_sync.get(); // We only chose sync strategy once

	bool pv61 = false;
	foreachPeer([&](std::shared_ptr<EthereumPeer> _p)
	{
		if (_p->m_protocolVersion == protocolVersion())
			pv61 = true;
		return !pv61;
	});
	if (pv61)
	{
		m_syncStart = 0;
		m_sync.reset(new PV61Sync(*this));
	}
	else if (!m_syncStart)
		m_syncStart = std::chrono::system_clock::to_time_t(chrono::system_clock::now());

	return m_sync.get();
}

void EthereumHost::onPeerStatus(std::shared_ptr<EthereumPeer> _peer)
{
	RecursiveGuard l(x_sync);
	if (sync())
		sync()->onPeerStatus(_peer);
}

void EthereumHost::onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	if (sync())
		sync()->onPeerHashes(_peer, _hashes);
}

void EthereumHost::onPeerBlocks(std::shared_ptr<EthereumPeer> _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	if (sync())
		sync()->onPeerBlocks(_peer, _r);
}

void EthereumHost::onPeerNewHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	if (sync())
		sync()->onPeerNewHashes(_peer, _hashes);
}

void EthereumHost::onPeerNewBlock(std::shared_ptr<EthereumPeer> _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	if (sync())
		sync()->onPeerNewBlock(_peer, _r);
}

void EthereumHost::onPeerTransactions(std::shared_ptr<EthereumPeer> _peer, RLP const& _r)
{
	unsigned itemCount = _r.itemCount();
	clog(EthereumHostTrace) << "Transactions (" << dec << itemCount << "entries)";
	m_tq.enqueue(_r, _peer->session()->id());
}

void EthereumHost::onPeerAborting()
{
	RecursiveGuard l(x_sync);
	try
	{
		if (m_sync)
			m_sync->onPeerAborting();
	}
	catch (Exception&)
	{
		cwarn << "Exception on peer destruciton: " << boost::current_exception_diagnostic_information();
	}
}

bool EthereumHost::isSyncing() const
{
	RecursiveGuard l(x_sync);
	if (!m_sync)
		return false;
	return m_sync->isSyncing();
}

SyncStatus EthereumHost::status() const
{
	RecursiveGuard l(x_sync);
	if (!m_sync)
		return SyncStatus();
	return m_sync->status();
}

void EthereumHost::onTransactionImported(ImportResult _ir, h256 const& _h, h512 const& _nodeId)
{
	auto session = host()->peerSession(_nodeId);
	if (!session)
		return;

	std::shared_ptr<EthereumPeer> peer = session->cap<EthereumPeer>();
	if (!peer)
		peer = session->cap<EthereumPeer>(c_oldProtocolVersion);
	if (!peer)
		return;

	Guard l(peer->x_knownTransactions);
	peer->m_knownTransactions.insert(_h);
	switch (_ir)
	{
	case ImportResult::Malformed:
		peer->addRating(-100);
		break;
	case ImportResult::AlreadyKnown:
		// if we already had the transaction, then don't bother sending it on.
		DEV_GUARDED(x_transactions)
			m_transactionsSent.insert(_h);
		peer->addRating(0);
		break;
	case ImportResult::Success:
		peer->addRating(100);
		break;
	default:;
	}
}
