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
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

unsigned const EthereumHost::c_oldProtocolVersion = 60; //TODO: remove this once v61+ is common

EthereumHost::EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId):
	HostCapability<EthereumPeer>(),
	Worker		("ethsync"),
	m_chain		(_ch),
	m_tq		(_tq),
	m_bq		(_bq),
	m_networkId	(_networkId)
{
	m_latestBlockSent = _ch.currentHash();
	m_hashMan.reset(m_chain.number() + 1);
}

EthereumHost::~EthereumHost()
{
	forEachPeer([](EthereumPeer* _p) { _p->abortSync(); });
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
	forEachPeer([](EthereumPeer* _p) { _p->abortSync(); });
	m_man.resetToChain(h256s());
	m_hashMan.reset(m_chain.number() + 1);
	m_needSyncBlocks = true;
	m_needSyncHashes = true;
	m_syncingLatestHash = h256();
	m_syncingTotalDifficulty = 0;
	m_latestBlockSent = h256();
	m_transactionsSent.clear();
	m_hashes.clear();
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

	forEachPeer([](EthereumPeer* _p) { _p->tick(); });

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
		for (auto const& p: randomSelection(0, [&](EthereumPeer* p) { return p->m_requireTransactions || (unsent && !p->m_knownTransactions.count(i.first)); }).second)
			peerTransactions[p].push_back(i.first);
	}
	for (auto const& t: ts)
		m_transactionsSent.insert(t.first);
	forEachPeerPtr([&](shared_ptr<EthereumPeer> _p)
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
		}
		_p->m_requireTransactions = false;
	});
}

void EthereumHost::forEachPeer(std::function<void(EthereumPeer*)> const& _f) const
{
	forEachPeerPtr([&](std::shared_ptr<EthereumPeer> _p)
	{
		if (_p)
			_f(_p.get());
	});
}

void EthereumHost::forEachPeerPtr(std::function<void(std::shared_ptr<EthereumPeer>)> const& _f) const
{
	for (auto s: peerSessions())
		_f(s.first->cap<EthereumPeer>());
	for (auto s: peerSessions(c_oldProtocolVersion)) //TODO: remove once v61+ is common
		_f(s.first->cap<EthereumPeer>(c_oldProtocolVersion));
}

pair<vector<shared_ptr<EthereumPeer>>, vector<shared_ptr<EthereumPeer>>> EthereumHost::randomSelection(unsigned _percent, std::function<bool(EthereumPeer*)> const& _allow)
{
	pair<vector<shared_ptr<EthereumPeer>>, vector<shared_ptr<EthereumPeer>>> ret;
	forEachPeerPtr([&](shared_ptr<EthereumPeer> _p)
	{
		if (_p && _allow(_p.get()))
			ret.second.push_back(_p);
	});

	size_t size = (ret.second.size() * _percent + 99) / 100;
	ret.second.reserve(size);
	for (unsigned i = size; i-- && ret.second.size();)
	{
		unsigned n = rand() % ret.second.size();
		ret.first.push_back(std::move(ret.second[n]));
		ret.second.erase(ret.second.begin() + n);
	}
	return ret;
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
			for (shared_ptr<EthereumPeer> const& p: s.first)
				for (auto const& b: blocks)
				{
					RLPStream ts;
					p->prep(ts, NewBlockPacket, 2).appendRaw(m_chain.block(b), 1).append(m_chain.details(b).totalDifficulty);

					Guard l(p->x_knownBlocks);
					p->sealAndSend(ts);
					p->m_knownBlocks.clear();
				}
			for (shared_ptr<EthereumPeer> const& p: s.second)
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

void EthereumHost::onPeerStatus(EthereumPeer* _peer)
{
	Guard l(x_sync);
	if (_peer->m_genesisHash != m_chain.genesisHash())
		_peer->disable("Invalid genesis hash");
	else if (_peer->m_protocolVersion != protocolVersion() && _peer->m_protocolVersion != c_oldProtocolVersion)
		_peer->disable("Invalid protocol version.");
	else if (_peer->m_networkId != networkId())
		_peer->disable("Invalid network identifier.");
	else if (_peer->session()->info().clientVersion.find("/v0.7.0/") != string::npos)
		_peer->disable("Blacklisted client version.");
	else if (isBanned(_peer->session()->id()))
		_peer->disable("Peer banned for previous bad behaviour.");
	else
	{
		if (_peer->m_protocolVersion != protocolVersion())
			estimatePeerHashes(_peer);
		else if (_peer->m_latestBlockNumber > m_chain.number())
			_peer->m_expectedHashes = (unsigned)_peer->m_latestBlockNumber - m_chain.number();
		if (m_hashMan.chainSize() < _peer->m_expectedHashes)
			m_hashMan.resetToRange(m_chain.number() + 1, _peer->m_expectedHashes);
		continueSync(_peer);
	}
}

void EthereumHost::estimatePeerHashes(EthereumPeer* _peer)
{
	BlockInfo block = m_chain.info();
	time_t lastBlockTime = (block.hash() == m_chain.genesisHash()) ? 1428192000 : (time_t)block.timestamp;
	time_t now = time(0);
	unsigned blockCount = 1000;
	if (lastBlockTime > now)
		clog(NetWarn) << "Clock skew? Latest block is in the future";
	else
		blockCount += (now - lastBlockTime) / (unsigned)c_durationLimit;
	clog(NetAllDetail) << "Estimated hashes: " << blockCount;
	_peer->m_expectedHashes = blockCount;
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	Guard l(x_sync);
	assert(_peer->m_asking == Asking::Nothing);
	onPeerHashes(_peer, _hashes, false);
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes, bool _complete)
{
	if (_hashes.empty())
	{
		onPeerDoneHashes(_peer, true);
		return;
	}
	unsigned knowns = 0;
	unsigned unknowns = 0;
	h256s neededBlocks;
	for (unsigned i = 0; i < _hashes.size(); ++i)
	{
		_peer->addRating(1);
		auto h = _hashes[i];
		auto status = m_bq.blockStatus(h);
		if (status == QueueStatus::Importing || status == QueueStatus::Ready || m_chain.isKnown(h))
		{
			clog(NetMessageSummary) << "block hash ready:" << h << ". Start blocks download...";
			m_hashes += neededBlocks;
			onPeerDoneHashes(_peer, true);
			return;
		}
		else if (status == QueueStatus::Bad)
		{
			cwarn << "block hash bad!" << h << ". Bailing...";
			_peer->setIdle();
			return;
		}
		else if (status == QueueStatus::Unknown)
		{
			unknowns++;
			neededBlocks.push_back(h);
		}
		else
			knowns++;
		m_syncingLatestHash = h;
	}
	m_hashes += neededBlocks;
	clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns; now at" << m_syncingLatestHash;
	if (_complete)
	{
		m_needSyncBlocks = true;
		continueSync(_peer);
	}
	else if (m_hashes.size() > _peer->m_expectedHashes)
	{
		_peer->disable("Too many hashes");
		m_hashes.clear();
		m_syncingLatestHash = h256();
		continueSync(); ///Try with some other peer, keep the chain
	}
	else
		continueSync(_peer); /// Grab next hashes
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, unsigned /*_index*/, h256s const& _hashes)
{
	Guard l(x_sync);
	assert(_peer->m_asking == Asking::Nothing);
	if (_hashes.empty())
	{
		onPeerDoneHashes(_peer, true);
		return;
	}
	unsigned knowns = 0;
	unsigned unknowns = 0;
	h256s neededBlocks;
	for (unsigned i = 0; i < _hashes.size(); ++i)
	{
		_peer->addRating(1);
		auto h = _hashes[i];
		auto status = m_bq.blockStatus(h);
		if (status == QueueStatus::Importing || status == QueueStatus::Ready || m_chain.isKnown(h))
		{
			clog(NetWarn) << "block hash already known:" << h;
		}
		else if (status == QueueStatus::Bad)
		{
			clog(NetWarn) << "block hash bad!" << h << ". Bailing...";
			_peer->setIdle();
			return;
		}
		else if (status == QueueStatus::Unknown)
		{
			unknowns++;
			neededBlocks.push_back(h);
		}
		else
			knowns++;
	}
	m_man.appendToChain(neededBlocks);
	clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns";

	if (m_hashMan.isComplete())
	{
		// Done our chain-get.
		m_needSyncHashes = false;
		clog(NetNote) << "Hashes download complete.";
		// 1/100th for each useful block hash.
		_peer->addRating(m_man.chainSize() / 100); //TODO: what about other peers?
		m_hashMan.reset(m_chain.number() + 1);
		continueSync();
	}
	else
		continueSync(_peer);
}

void EthereumHost::onPeerDoneHashes(EthereumPeer* _peer, bool _localChain)
{
	assert(_peer->m_asking == Asking::Nothing);
	m_needSyncHashes = false;
	if (_peer->m_protocolVersion != protocolVersion() || _localChain)
	{
		m_man.resetToChain(m_hashes);
		m_hashes.clear();
	}
	continueSync();
}

void EthereumHost::onPeerBlocks(EthereumPeer* _peer, RLP const& _r)
{
	Guard l(x_sync);
	assert(_peer->m_asking == Asking::Nothing);
	unsigned itemCount = _r.itemCount();
	clog(NetMessageSummary) << "Blocks (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreBlocks");

	if (itemCount == 0)
	{
		// Got to this peer's latest block - just give up.
		clog(NetNote) << "Finishing blocks fetch...";
		// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
		_peer->setIdle();
		return;
	}

	unsigned success = 0;
	unsigned future = 0;
	unsigned unknown = 0;
	unsigned got = 0;
	unsigned repeated = 0;

	for (unsigned i = 0; i < itemCount; ++i)
	{
		auto h = BlockInfo::headerHash(_r[i].data());
		if (_peer->m_sub.noteBlock(h))
		{
			_peer->addRating(10);
			switch (m_bq.import(_r[i].data(), m_chain))
			{
			case ImportResult::Success:
				success++;
				break;

			case ImportResult::Malformed:
			case ImportResult::BadChain:
				_peer->disable("Malformed block received.");
				return;

			case ImportResult::FutureTime:
				future++;
				break;

			case ImportResult::AlreadyInChain:
			case ImportResult::AlreadyKnown:
				got++;
				break;

			case ImportResult::UnknownParent:
				unknown++;
				break;

			default:;
			}
		}
		else
		{
			_peer->addRating(0);	// -1?
			repeated++;
		}
	}

	clog(NetMessageSummary) << dec << success << "imported OK," << unknown << "with unknown parents," << future << "with future timestamps," << got << " already known," << repeated << " repeats received.";

	if (m_man.isComplete() && !m_needSyncHashes)
	{
		// Done our chain-get.
		m_needSyncBlocks = false;
		clog(NetNote) << "Chain download complete.";
		// 1/100th for each useful block hash.
		_peer->addRating(m_man.chainSize() / 100); //TODO: what about other peers?
		m_man.reset();
	}
	continueSync(_peer);
}

void EthereumHost::onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	Guard l(x_sync);
	if (_peer->m_asking != Asking::Nothing)
	{
		clog(NetMessageSummary) << "Ignoring new hashes since we're already downloading.";
		return;
	}
	clog(NetNote) << "New block hash discovered: syncing without help.";
	onPeerHashes(_peer, _hashes, true);
}

void EthereumHost::onPeerNewBlock(EthereumPeer* _peer, RLP const& _r)
{
	Guard l(x_sync);
	if (_peer->m_asking != Asking::Nothing)
	{
		clog(NetMessageSummary) << "Ignoring new blocks since we're already downloading.";
		return;
	}
	auto h = BlockInfo::headerHash(_r[0].data());
	clog(NetMessageSummary) << "NewBlock: " << h;

	if (_r.itemCount() != 2)
		_peer->disable("NewBlock without 2 data fields.");
	else
	{
		bool sync = false;
		switch (m_bq.import(_r[0].data(), m_chain))
		{
		case ImportResult::Success:
			_peer->addRating(100);
			break;
		case ImportResult::FutureTime:
			//TODO: Rating dependent on how far in future it is.
			break;

		case ImportResult::Malformed:
		case ImportResult::BadChain:
			_peer->disable("Malformed block received.");
			return;

		case ImportResult::AlreadyInChain:
		case ImportResult::AlreadyKnown:
			break;

		case ImportResult::UnknownParent:
			if (h)
			{
				u256 difficulty = _r[1].toInt<u256>();
				if (m_syncingTotalDifficulty < difficulty)
				{
					clog(NetMessageSummary) << "Received block with no known parent. Resyncing...";
					_peer->m_latestHash = h;
					_peer->m_totalDifficulty = difficulty;
					m_needSyncHashes = true;
					m_needSyncBlocks = true;
					m_syncingLatestHash = _peer->m_latestHash;
					sync = true;
				}
			}
			break;
		default:;
		}

		DEV_GUARDED(_peer->x_knownBlocks)
			_peer->m_knownBlocks.insert(h);

		if (sync)
			continueSync(_peer);
	}
}

void EthereumHost::onPeerTransactions(EthereumPeer* _peer, RLP const& _r)
{
	unsigned itemCount = _r.itemCount();
	clog(NetAllDetail) << "Transactions (" << dec << itemCount << "entries)";
	Guard l(_peer->x_knownTransactions);
	for (unsigned i = 0; i < itemCount; ++i)
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

void EthereumHost::continueSync()
{
	clog(NetAllDetail) << "Getting help with downloading hashes and blocks";
	forEachPeer([&](EthereumPeer* _p)
	{
		if (_p->m_asking == Asking::Nothing)
			continueSync(_p);
	});
}

void EthereumHost::continueSync(EthereumPeer* _peer)
{
	assert(_peer->m_asking == Asking::Nothing);
	bool otherPeerSync = false;
	if (m_needSyncHashes && peerShouldGrabChain(_peer))
	{
		forEachPeer([&](EthereumPeer* _p)
		{
			if (_p != _peer && _p->m_asking == Asking::Hashes && _p->m_protocolVersion != protocolVersion())
				otherPeerSync = true; // Already have a peer downloading hash chain with old protocol, do nothing
		});
		if (otherPeerSync)
		{
			/// Downloading from other peer with v60 protocol, nothing ese we can do
			_peer->setIdle();
			return;
		}
		if (_peer->m_protocolVersion == protocolVersion() && !m_syncingLatestHash)
			_peer->requestHashes(); /// v61+ and not catching up to a particular hash
		else
		{
			// Restart/continue sync in single peer mode
			if (!m_syncingLatestHash)
			{
				m_syncingLatestHash =_peer->m_latestHash;
				m_syncingTotalDifficulty = _peer->m_totalDifficulty;
			}
			_peer->requestHashes(m_syncingLatestHash);
		}
	}
	else if (m_needSyncBlocks && peerShouldGrabBlocks(_peer)) // Check if this peer can help with downloading blocks
		_peer->requestBlocks();
	else
		_peer->setIdle();
}

bool EthereumHost::peerShouldGrabBlocks(EthereumPeer* _peer) const
{
	auto td = _peer->m_totalDifficulty;
	auto lh = m_syncingLatestHash;
	auto ctd = m_chain.details().totalDifficulty;

	clog(NetAllDetail) << "Should grab blocks? " << td << "vs" << ctd;
	if (td < ctd || (td == ctd && m_chain.currentHash() == lh))
		return false;
	return true;
}

bool EthereumHost::peerShouldGrabChain(EthereumPeer* _peer) const
{
	h256 c = m_chain.currentHash();
	unsigned n = m_chain.number();
	u256 td = m_chain.details().totalDifficulty;

	clog(NetAllDetail) << "Attempt chain-grab? Latest:" << c << ", number:" << n << ", TD:" << td << " versus " << _peer->m_totalDifficulty;
	if (td >= _peer->m_totalDifficulty)
	{
		clog(NetAllDetail) << "No. Our chain is better.";
		return false;
	}
	else
	{
		clog(NetAllDetail) << "Yes. Their chain is better.";
		return true;
	}
}

bool EthereumHost::isSyncing() const
{
	Guard l(x_sync);
	bool syncing = false;
	forEachPeer([&](EthereumPeer* _p)
	{
		if (_p->m_asking != Asking::Nothing)
			syncing = true;
	});
	return syncing;
}
