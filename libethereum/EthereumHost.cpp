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
unsigned const c_chainReorgSize = 30000;

char const* const EthereumHost::s_stateNames[static_cast<int>(SyncState::Size)] = {"Idle", "WaitingQueue", "HashesNegotiate", "HashesSingle", "HashesParallel", "Blocks", "NewBlocks" };

EthereumHost::EthereumHost(BlockChain const& _ch, TransactionQueue& _tq, BlockQueue& _bq, u256 _networkId):
	HostCapability<EthereumPeer>(),
	Worker		("ethsync"),
	m_chain		(_ch),
	m_tq		(_tq),
	m_bq		(_bq),
	m_networkId	(_networkId)
{
	setState(SyncState::HashesNegotiate);
	m_latestBlockSent = _ch.currentHash();
	m_hashMan.reset(m_chain.number() + 1);
	m_bqRoomAvailable = m_bq.onRoomAvailable([this](){ m_continueSync = true; });
}

EthereumHost::~EthereumHost()
{
	foreachPeer([](EthereumPeer* _p) { _p->abortSync(); });
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
	foreachPeer([](EthereumPeer* _p) { _p->abortSync(); });
	m_man.resetToChain(h256s());
	m_hashMan.reset(m_chain.number() + 1);
	setState(SyncState::HashesNegotiate);
	m_syncingLatestHash = h256();
	m_syncingTotalDifficulty = 0;
	m_latestBlockSent = h256();
	m_transactionsSent.clear();
	m_hashes.clear();
}

void EthereumHost::resetSyncTo(h256 const& _h)
{
	setState(SyncState::HashesNegotiate);
	m_syncingLatestHash = _h;
}


void EthereumHost::setState(SyncState _s)
{
	if (m_state != _s)
	{
		clog(NetAllDetail) << "SyncState changed from " << stateName(m_state) << " to " << stateName(_s);
		m_state = _s;
	}
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

	if (m_continueSync)
	{
		m_continueSync = false;
		RecursiveGuard l(x_sync);
		continueSync();
	}

	foreachPeer([](EthereumPeer* _p) { _p->tick(); });

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
	});
}

void EthereumHost::foreachPeer(std::function<void(EthereumPeer*)> const& _f) const
{
	foreachPeerPtr([&](std::shared_ptr<EthereumPeer> _p)
	{
		if (_p)
			_f(_p.get());
	});
}

void EthereumHost::foreachPeerPtr(std::function<void(std::shared_ptr<EthereumPeer>)> const& _f) const
{
	for (auto s: peerSessions())
		_f(s.first->cap<EthereumPeer>());
	for (auto s: peerSessions(c_oldProtocolVersion)) //TODO: remove once v61+ is common
		_f(s.first->cap<EthereumPeer>(c_oldProtocolVersion));
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

void EthereumHost::onPeerStatus(EthereumPeer* _peer)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
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
		unsigned estimatedHashes = estimateHashes();
		if (_peer->m_protocolVersion == protocolVersion())
		{
			if (_peer->m_latestBlockNumber > m_chain.number())
				_peer->m_expectedHashes = (unsigned)_peer->m_latestBlockNumber - m_chain.number();
			if (_peer->m_expectedHashes > estimatedHashes)
				_peer->disable("Too many hashes");
			else if (needHashes() && m_hashMan.chainSize() < _peer->m_expectedHashes)
				m_hashMan.resetToRange(m_chain.number() + 1, _peer->m_expectedHashes);
		}
		else
			_peer->m_expectedHashes = estimatedHashes;
		continueSync(_peer);
	}
	DEV_INVARIANT_CHECK;
}

unsigned EthereumHost::estimateHashes()
{
	BlockInfo block = m_chain.info();
	time_t lastBlockTime = (block.hash() == m_chain.genesisHash()) ? 1428192000 : (time_t)block.timestamp;
	time_t now = time(0);
	unsigned blockCount = c_chainReorgSize;
	if (lastBlockTime > now)
		clog(NetWarn) << "Clock skew? Latest block is in the future";
	else
		blockCount += (now - lastBlockTime) / (unsigned)c_durationLimit;
	clog(NetAllDetail) << "Estimated hashes: " << blockCount;
	return blockCount;
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	if (_peer->m_syncHashNumber > 0)
		_peer->m_syncHashNumber += _hashes.size();

	_peer->setAsking(Asking::Nothing);
	onPeerHashes(_peer, _hashes, false);
}

void EthereumHost::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes, bool _complete)
{
	DEV_INVARIANT_CHECK;
	if (_hashes.empty())
	{
		_peer->m_hashSub.doneFetch();
		continueSync();
		return;
	}

	bool syncByNumber = _peer->m_syncHashNumber;
	if (!syncByNumber && !_complete && _peer->m_syncHash != m_syncingLatestHash)
	{
		// Obsolete hashes, discard
		continueSync(_peer);
		return;
	}

	unsigned knowns = 0;
	unsigned unknowns = 0;
	h256s neededBlocks;
	unsigned firstNumber = _peer->m_syncHashNumber - _hashes.size();
	for (unsigned i = 0; i < _hashes.size(); ++i)
	{
		_peer->addRating(1);
		auto h = _hashes[i];
		auto status = m_bq.blockStatus(h);
		if (status == QueueStatus::Importing || status == QueueStatus::Ready || m_chain.isKnown(h))
		{
			clog(NetMessageSummary) << "Block hash already known:" << h;
			if (!syncByNumber)
			{
				m_hashes += neededBlocks;
				clog(NetMessageSummary) << "Start blocks download...";
				onPeerDoneHashes(_peer, true);
				return;
			}
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

		if (!syncByNumber)
			m_syncingLatestHash = h;
		else
			_peer->m_hashSub.noteHash(firstNumber + i, 1);
	}
	if (syncByNumber)
	{
		m_man.appendToChain(neededBlocks);	// Append to download manager immediatelly
		clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns";
	}
	else
	{
		m_hashes += neededBlocks;			// Append to local list
		clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns; now at" << m_syncingLatestHash;
	}
	if (_complete)
	{
		clog(NetMessageSummary) << "Start new blocks download...";
		m_syncingLatestHash = h256();
		setState(SyncState::NewBlocks);
		m_man.resetToChain(m_hashes);
		m_hashes.clear();
		m_hashMan.reset(m_chain.number() + 1);
		continueSync(_peer);
	}
	else if (syncByNumber && m_hashMan.isComplete())
	{
		// Done our chain-get.
		clog(NetNote) << "Hashes download complete.";
		onPeerDoneHashes(_peer, false);
	}
	else if (m_hashes.size() > _peer->m_expectedHashes)
	{
		_peer->disable("Too many hashes");
		m_hashes.clear();
		m_syncingLatestHash = h256();
		setState(SyncState::HashesNegotiate);
		continueSync(); ///Try with some other peer, keep the chain
	}
	else
		continueSync(_peer); /// Grab next hashes
	DEV_INVARIANT_CHECK;
}

void EthereumHost::onPeerDoneHashes(EthereumPeer* _peer, bool _localChain)
{
	assert(_peer->m_asking == Asking::Nothing);
	m_syncingLatestHash = h256();
	setState(SyncState::Blocks);
	if (_peer->m_protocolVersion != protocolVersion() || _localChain)
	{
		m_man.resetToChain(m_hashes);
		_peer->addRating(m_man.chainSize() / 100); //TODO: what about other peers?
	}
	m_hashMan.reset(m_chain.number() + 1);
	m_hashes.clear();
	continueSync();
}

void EthereumHost::onPeerBlocks(EthereumPeer* _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	_peer->setAsking(Asking::Nothing);
	unsigned itemCount = _r.itemCount();
	clog(NetMessageSummary) << "Blocks (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreBlocks");

	if (itemCount == 0)
	{
		// Got to this peer's latest block - just give up.
		clog(NetNote) << "Finishing blocks fetch...";
		// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
		_peer->m_sub.doneFetch();
		_peer->setIdle();
		return;
	}

	unsigned success = 0;
	unsigned future = 0;
	unsigned unknown = 0;
	unsigned got = 0;
	unsigned repeated = 0;
	h256 lastUnknown;

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
				lastUnknown = h;
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

	if (m_state == SyncState::NewBlocks && unknown > 0)
	{
		_peer->m_latestHash = lastUnknown;
		resetSyncTo(lastUnknown);
	}

	continueSync(_peer);
	DEV_INVARIANT_CHECK;
}

void EthereumHost::onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if (isSyncing() || _peer->isConversing())
	{
		clog(NetMessageSummary) << "Ignoring new hashes since we're already downloading.";
		return;
	}
	clog(NetNote) << "New block hash discovered: syncing without help.";
	_peer->m_syncHashNumber = 0;
	onPeerHashes(_peer, _hashes, true);
	DEV_INVARIANT_CHECK;
}

void EthereumHost::onPeerNewBlock(EthereumPeer* _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if ((isSyncing() || _peer->isConversing()) && m_state != SyncState::NewBlocks)
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
					resetSyncTo(h);;
					sync = true;
				}
			}
			break;
		default:;
		}

		DEV_GUARDED(_peer->x_knownBlocks)
			_peer->m_knownBlocks.insert(h);

		if (sync)
			continueSync();
	}
	DEV_INVARIANT_CHECK;
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

void EthereumHost::onPeerAborting(EthereumPeer* _peer)
{
	RecursiveGuard l(x_sync);
	if (_peer->isConversing())
	{
		_peer->setIdle();
//		if (_peer->isCriticalSyncing())
			_peer->setRude();
		continueSync();
	}
}

void EthereumHost::continueSync()
{
	if (m_state == SyncState::WaitingQueue)
		setState(m_lastActiveState);
	clog(NetAllDetail) << "Continuing sync for all peers";
	foreachPeer([&](EthereumPeer* _p)
	{
		if (_p->m_asking == Asking::Nothing)
			continueSync(_p);
	});
}

void EthereumHost::continueSync(EthereumPeer* _peer)
{
	DEV_INVARIANT_CHECK;
	assert(_peer->m_asking == Asking::Nothing);
	bool otherPeerV60Sync = false;
	bool otherPeerV61Sync = false;
	if (needHashes())
	{
		if (!peerShouldGrabChain(_peer))
		{
			_peer->setIdle();
			return;
		}

		foreachPeer([&](EthereumPeer* _p)
		{
			if (_p != _peer && _p->m_asking == Asking::Hashes)
			{
				if (_p->m_protocolVersion != protocolVersion())
					otherPeerV60Sync = true; // Already have a peer downloading hash chain with old protocol, do nothing
				else
					otherPeerV61Sync = true; // Already have a peer downloading hash chain with V61+ protocol, join if supported
			}
		});
		if (otherPeerV60Sync && !m_hashes.empty())
		{
			/// Downloading from other peer with v60 protocol, nothing else we can do
			_peer->setIdle();
			return;
		}
		if (otherPeerV61Sync && _peer->m_protocolVersion != protocolVersion())
		{
			/// Downloading from other peer with v61+ protocol which this peer does not support,
			_peer->setIdle();
			return;
		}
		if (_peer->m_protocolVersion == protocolVersion() && !m_hashMan.isComplete())
		{
			setState(SyncState::HashesParallel);
			_peer->requestHashes(); /// v61+ and not catching up to a particular hash
		}
		else
		{
			// Restart/continue sync in single peer mode
			if (!m_syncingLatestHash)
			{
				m_syncingLatestHash =_peer->m_latestHash;
				m_syncingTotalDifficulty = _peer->m_totalDifficulty;
			}
			if (_peer->m_totalDifficulty >= m_syncingTotalDifficulty)
			{
				_peer->requestHashes(m_syncingLatestHash);
				setState(SyncState::HashesSingle);
				m_estimatedHashes = _peer->m_expectedHashes - (_peer->m_protocolVersion == protocolVersion() ? 0 : c_chainReorgSize);
			}
			else
				_peer->setIdle();
		}
	}
	else if (needBlocks())
	{
		if (m_man.isComplete())
		{
			// Done our chain-get.
			setState(SyncState::Idle);
			clog(NetNote) << "Chain download complete.";
			// 1/100th for each useful block hash.
			_peer->addRating(m_man.chainSize() / 100); //TODO: what about other peers?
			m_man.reset();
			_peer->setIdle();
			return;
		}
		else if (peerCanHelp(_peer))
		{
			// Check block queue status
			if (m_bq.unknownFull())
			{
				clog(NetWarn) << "Too many unknown blocks, restarting sync";
				m_bq.clear();
				reset();
				continueSync();
			}
			else if (m_bq.knownFull())
			{
				clog(NetAllDetail) << "Waiting for block queue before downloading blocks";
				m_lastActiveState = m_state;
				setState(SyncState::WaitingQueue);
				_peer->setIdle();
			}
			else
				_peer->requestBlocks();
		}
	}
	else
		_peer->setIdle();
	DEV_INVARIANT_CHECK;
}

bool EthereumHost::peerCanHelp(EthereumPeer* _peer) const
{
	(void)_peer;
	return true;
}

bool EthereumHost::peerShouldGrabBlocks(EthereumPeer* _peer) const
{
	// this is only good for deciding whether to go ahead and grab a particular peer's hash chain,
	// yet it's being used in determining whether to allow a peer help with downloading an existing
	// chain of blocks.
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
	// Early exit if this peer has proved unreliable.
	if (_peer->isRude())
		return false;

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
	return m_state != SyncState::Idle;
}

SyncStatus EthereumHost::status() const
{
	RecursiveGuard l(x_sync);
	SyncStatus res;
	res.state = m_state;
	if (m_state == SyncState::HashesParallel)
	{
		res.hashesReceived = m_hashMan.hashesGot().size();
		res.hashesTotal = m_hashMan.chainSize();
	}
	else if (m_state == SyncState::HashesSingle)
	{
		res.hashesTotal = m_estimatedHashes;
		res.hashesReceived = static_cast<unsigned>(m_hashes.size());
		res.hashesEstimated = true;
	}
	else if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks || m_state == SyncState::WaitingQueue)
	{
		res.blocksTotal = m_man.chainSize();
		res.blocksReceived = m_man.blocksGot().size();
	}
	return res;
}


bool EthereumHost::invariants() const
{
	if (m_state == SyncState::HashesNegotiate && !m_hashes.empty())
		return false;
	if (needBlocks() && (m_syncingLatestHash || !m_hashes.empty()))
		return false;

	return true;
}
