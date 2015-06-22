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
/** @file BlockChainSync.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "BlockChainSync.h"

#include <chrono>
#include <libdevcore/Common.h>
#include <libp2p/Host.h>
#include <libp2p/Session.h>
#include <libethcore/Exceptions.h>
#include <libethcore/Params.h>
#include "BlockChain.h"
#include "BlockQueue.h"
#include "EthereumPeer.h"
#include "EthereumHost.h"
#include "DownloadMan.h"

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

unsigned const c_chainReorgSize = 30000;

BlockChainSync::BlockChainSync(EthereumHost& _host):
	m_host(_host)
{
	m_bqRoomAvailable = host().bq().onRoomAvailable([this]()
	{
		RecursiveGuard l(x_sync);
		continueSync();
	});
}

BlockChainSync::~BlockChainSync()
{
	RecursiveGuard l(x_sync);
	abortSync();
}

DownloadMan const& BlockChainSync::downloadMan() const
{
	return host().downloadMan();
}

DownloadMan& BlockChainSync::downloadMan()
{
	return host().downloadMan();
}

void BlockChainSync::abortSync()
{
	DEV_INVARIANT_CHECK;
	host().foreachPeer([this](EthereumPeer* _p) { onPeerAborting(_p); return true; });
	downloadMan().resetToChain(h256s());
	DEV_INVARIANT_CHECK;
}

void BlockChainSync::onPeerStatus(EthereumPeer* _peer)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if (_peer->m_genesisHash != host().chain().genesisHash())
		_peer->disable("Invalid genesis hash");
	else if (_peer->m_protocolVersion != host().protocolVersion() && _peer->m_protocolVersion != EthereumHost::c_oldProtocolVersion)
		_peer->disable("Invalid protocol version.");
	else if (_peer->m_networkId != host().networkId())
		_peer->disable("Invalid network identifier.");
	else if (_peer->session()->info().clientVersion.find("/v0.7.0/") != string::npos)
		_peer->disable("Blacklisted client version.");
	else if (host().isBanned(_peer->session()->id()))
		_peer->disable("Peer banned for previous bad behaviour.");
	else
	{
		unsigned hashes = estimatedHashes();
		_peer->m_expectedHashes = hashes;
		onNewPeer(_peer);
	}
	DEV_INVARIANT_CHECK;
}

unsigned BlockChainSync::estimatedHashes() const
{
	BlockInfo block = host().chain().info();
	time_t lastBlockTime = (block.hash() == host().chain().genesisHash()) ? 1428192000 : (time_t)block.timestamp;
	time_t now = time(0);
	unsigned blockCount = c_chainReorgSize;
	if (lastBlockTime > now)
		clog(NetWarn) << "Clock skew? Latest block is in the future";
	else
		blockCount += (now - lastBlockTime) / (unsigned)c_durationLimit;
	clog(NetAllDetail) << "Estimated hashes: " << blockCount;
	return blockCount;
}

void BlockChainSync::requestBlocks(EthereumPeer* _peer)
{
	if (host().bq().knownFull())
	{
		clog(NetAllDetail) << "Waiting for block queue before downloading blocks";
		pauseSync();
		_peer->setIdle();
		return;
	}
	_peer->requestBlocks();
	if (_peer->m_asking != Asking::Blocks) //nothing to download
	{
		peerDoneBlocks(_peer);
		if (downloadMan().isComplete())
			completeSync();
		return;
	}
}

void BlockChainSync::onPeerBlocks(EthereumPeer* _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	unsigned itemCount = _r.itemCount();
	clog(NetMessageSummary) << "Blocks (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreBlocks");

	_peer->setIdle();
	if (m_state != SyncState::Blocks && m_state != SyncState::NewBlocks)
		clog(NetWarn) << "Unexpected Blocks received!";
	if (m_state == SyncState::Waiting)
	{
		clog(NetAllDetail) << "Ignored blocks while waiting";
		return;
	}

	if (itemCount == 0)
	{
		// Got to this peer's latest block - just give up.
		peerDoneBlocks(_peer);
		if (downloadMan().isComplete())
			completeSync();
		return;
	}

	unsigned success = 0;
	unsigned future = 0;
	unsigned unknown = 0;
	unsigned got = 0;
	unsigned repeated = 0;
	u256 maxUnknownNumber = 0;
	h256 maxUnknown;

	for (unsigned i = 0; i < itemCount; ++i)
	{
		auto h = BlockInfo::headerHash(_r[i].data());
		if (_peer->m_sub.noteBlock(h))
		{
			_peer->addRating(10);
			switch (host().bq().import(_r[i].data(), host().chain()))
			{
			case ImportResult::Success:
				success++;
				break;

			case ImportResult::Malformed:
			case ImportResult::BadChain:
				_peer->disable("Malformed block received.");
				return;

			case ImportResult::FutureTimeKnown:
				future++;
				break;
			case ImportResult::AlreadyInChain:
			case ImportResult::AlreadyKnown:
				got++;
				break;

			case ImportResult::FutureTimeUnknown:
				future++; //Fall through

			case ImportResult::UnknownParent:
			{
				unknown++;
				if (m_state == SyncState::NewBlocks)
				{
					BlockInfo bi;
					bi.populateFromHeader(_r[i][0]);
					if (bi.number > maxUnknownNumber)
					{
						maxUnknownNumber = bi.number;
						maxUnknown = h;
					}
				}
				break;
			}

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

	if (host().bq().unknownFull())
	{
		clog(NetWarn) << "Too many unknown blocks, restarting sync";
		restartSync();
		return;
	}

	if (m_state == SyncState::NewBlocks && unknown > 0)
	{
		completeSync();
		resetSyncFor(_peer, maxUnknown, std::numeric_limits<u256>::max()); //TODO: proper total difficuty
	}
	if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks)
	{
		if (downloadMan().isComplete())
			completeSync();
		else
			requestBlocks(_peer); // Some of the blocks might have been downloaded by helping peers, proceed anyway
	}
	DEV_INVARIANT_CHECK;
}

void BlockChainSync::onPeerNewBlock(EthereumPeer* _peer, RLP const& _r)
{
	DEV_INVARIANT_CHECK;
	RecursiveGuard l(x_sync);
	auto h = BlockInfo::headerHash(_r[0].data());
	clog(NetMessageSummary) << "NewBlock: " << h;

	if (_r.itemCount() != 2)
		_peer->disable("NewBlock without 2 data fields.");
	else
	{
		switch (host().bq().import(_r[0].data(), host().chain()))
		{
		case ImportResult::Success:
			_peer->addRating(100);
			break;
		case ImportResult::FutureTimeKnown:
			//TODO: Rating dependent on how far in future it is.
			break;

		case ImportResult::Malformed:
		case ImportResult::BadChain:
			_peer->disable("Malformed block received.");
			return;

		case ImportResult::AlreadyInChain:
		case ImportResult::AlreadyKnown:
			break;

		case ImportResult::FutureTimeUnknown:
		case ImportResult::UnknownParent:
			clog(NetMessageSummary) << "Received block with no known parent. Resyncing...";
			resetSyncFor(_peer, h, _r[1].toInt<u256>());
			break;
		default:;
		}

		DEV_GUARDED(_peer->x_knownBlocks)
			_peer->m_knownBlocks.insert(h);
	}
	DEV_INVARIANT_CHECK;
}

PV60Sync::PV60Sync(EthereumHost& _host):
	BlockChainSync(_host)
{
	resetSync();
}

SyncStatus PV60Sync::status() const
{
	RecursiveGuard l(x_sync);
	SyncStatus res;
	res.state = m_state;
	if (m_state == SyncState::Hashes)
	{
		res.hashesTotal = m_estimatedHashes;
		res.hashesReceived = static_cast<unsigned>(m_syncingNeededBlocks.size());
		res.hashesEstimated = true;
	}
	else if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks || m_state == SyncState::Waiting)
	{
		res.blocksTotal = downloadMan().chainSize();
		res.blocksReceived = downloadMan().blocksGot().size();
	}
	return res;
}

void PV60Sync::setState(EthereumPeer* _peer, SyncState _s, bool _isSyncing, bool _needHelp)
{
	bool changedState = (m_state != _s);
	m_state = _s;

	if (_isSyncing != (m_syncer == _peer) || (_isSyncing && changedState))
		changeSyncer(_isSyncing ? _peer : nullptr, _needHelp);
	else if (_s == SyncState::Idle)
		changeSyncer(nullptr, _needHelp);

	assert(!!m_syncer || _s == SyncState::Idle);
}

void PV60Sync::resetSync()
{
	m_syncingLatestHash = h256();
	m_syncingLastReceivedHash = h256();
	m_syncingTotalDifficulty = 0;
	m_syncingNeededBlocks.clear();
}

void PV60Sync::restartSync()
{
	resetSync();
	host().bq().clear();
	if (isSyncing())
		transition(m_syncer, SyncState::Idle);
}

void PV60Sync::completeSync()
{
	if (isSyncing())
		transition(m_syncer, SyncState::Idle);
}

void PV60Sync::pauseSync()
{
	if (isSyncing())
		setState(m_syncer, SyncState::Waiting, true);
}

void PV60Sync::continueSync()
{
	transition(m_syncer, SyncState::Blocks);
}

void PV60Sync::onNewPeer(EthereumPeer* _peer)
{
	setNeedsSyncing(_peer, _peer->m_latestHash, _peer->m_totalDifficulty);
}

void PV60Sync::transition(EthereumPeer* _peer, SyncState _s, bool _force, bool _needHelp)
{
	clog(NetMessageSummary) << "Transition!" << EthereumHost::stateName(_s) << "from" << EthereumHost::stateName(m_state) << ", " << (isSyncing(_peer) ? "syncing" : "holding") << (needsSyncing(_peer) ? "& needed" : "");

	if (m_state == SyncState::Idle && _s != SyncState::Idle)
		_peer->m_requireTransactions = true;

	RLPStream s;
	if (_s == SyncState::Hashes)
	{
		if (m_state == SyncState::Idle)
		{
			if (isSyncing(_peer))
				clog(NetWarn) << "Bad state: not asking for Hashes, yet syncing!";

			m_syncingLatestHash = _peer->m_latestHash;
			m_syncingTotalDifficulty = _peer->m_totalDifficulty;
			setState(_peer, _s, true);
			_peer->requestHashes(m_syncingLastReceivedHash ? m_syncingLastReceivedHash : m_syncingLatestHash);
			DEV_INVARIANT_CHECK;
			return;
		}
		else if (m_state == SyncState::Hashes)
		{
			if (!isSyncing(_peer))
				clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";

			setState(_peer, _s, true);
			_peer->requestHashes(m_syncingLastReceivedHash);
			DEV_INVARIANT_CHECK;
			return;
		}
	}
	else if (_s == SyncState::Blocks)
	{
		if (m_state == SyncState::Hashes)
		{
			if (!isSyncing(_peer))
			{
				clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";
				return;
			}
			if (shouldGrabBlocks(_peer))
			{
				clog(NetNote) << "Difficulty of hashchain HIGHER. Grabbing" << m_syncingNeededBlocks.size() << "blocks [latest now" << m_syncingLatestHash << ", was" << host().latestBlockSent() << "]";
				downloadMan().resetToChain(m_syncingNeededBlocks);
				resetSync();
			}
			else
			{
				clog(NetNote) << "Difficulty of hashchain not HIGHER. Ignoring.";
				resetSync();
				setState(_peer, SyncState::Idle, false);
				return;
			}
			assert (isSyncing(_peer));
		}
		// run through into...
		if (m_state == SyncState::Idle || m_state == SyncState::Hashes || m_state == SyncState::Blocks || m_state == SyncState::Waiting)
		{
			// Looks like it's the best yet for total difficulty. Set to download.
			setState(_peer, SyncState::Blocks, isSyncing(_peer), _needHelp);		// will kick off other peers to help if available.
			requestBlocks(_peer);
			DEV_INVARIANT_CHECK;
			return;
		}
	}
	else if (_s == SyncState::NewBlocks)
	{
		if (m_state != SyncState::Idle && m_state != SyncState::NewBlocks && m_state != SyncState::Waiting)
			clog(NetWarn) << "Bad state: Asking new blocks while syncing!";
		else
		{
			setState(_peer, SyncState::NewBlocks, true, _needHelp);
			requestBlocks(_peer);
			DEV_INVARIANT_CHECK;
			return;
		}
	}
	else if (_s == SyncState::Waiting)
	{
		if (m_state != SyncState::Blocks && m_state != SyncState::NewBlocks && m_state != SyncState::Hashes && m_state != SyncState::Waiting)
			clog(NetWarn) << "Bad state: Entering waiting state while not downloading blocks!";
		else
		{
			setState(_peer, SyncState::Waiting, isSyncing(_peer), _needHelp);
			return;
		}
	}
	else if (_s == SyncState::Idle)
	{
		host().foreachPeer([this](EthereumPeer* _p) { _p->setIdle(); return true; });
		if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks)
		{
			clog(NetNote) << "Finishing blocks fetch...";

			// a bit overkill given that the other nodes may yet have the needed blocks, but better to be safe than sorry.
			if (isSyncing(_peer))
				noteDoneBlocks(_peer, _force);

			// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
			_peer->m_sub.doneFetch();
			_peer->setIdle();
			setState(_peer, SyncState::Idle, false);
		}
		else if (m_state == SyncState::Hashes)
		{
			clog(NetNote) << "Finishing hashes fetch...";
			setState(_peer, SyncState::Idle, false);
		}
		// Otherwise it's fine. We don't care if it's Nothing->Nothing.
		DEV_INVARIANT_CHECK;
		return;
	}

	clog(NetWarn) << "Invalid state transition:" << EthereumHost::stateName(_s) << "from" << EthereumHost::stateName(m_state) << ", " << (isSyncing(_peer) ? "syncing" : "holding") << (needsSyncing(_peer) ? "& needed" : "");
}

void PV60Sync::resetSyncFor(EthereumPeer* _peer, h256 const& _latestHash, u256 const& _td)
{
	setNeedsSyncing(_peer, _latestHash, _td);
}

void PV60Sync::setNeedsSyncing(EthereumPeer* _peer, h256 const& _latestHash, u256 const& _td)
{
	_peer->m_latestHash = _latestHash;
	_peer->m_totalDifficulty = _td;

	if (_peer->m_latestHash)
		noteNeedsSyncing(_peer);

	_peer->session()->addNote("sync", string(isSyncing(_peer) ? "ongoing" : "holding") + (needsSyncing(_peer) ? " & needed" : ""));
}

bool PV60Sync::needsSyncing(EthereumPeer* _peer) const
{
	return !!_peer->m_latestHash;
}

bool PV60Sync::isSyncing(EthereumPeer* _peer) const
{
	return m_syncer == _peer;
}

bool PV60Sync::shouldGrabBlocks(EthereumPeer* _peer) const
{
	auto td = _peer->m_totalDifficulty;
	auto lh = _peer->m_latestHash;
	auto ctd = host().chain().details().totalDifficulty;

	if (m_syncingNeededBlocks.empty())
		return false;

	clog(NetNote) << "Should grab blocks? " << td << "vs" << ctd << ";" << m_syncingNeededBlocks.size() << " blocks, ends" << m_syncingNeededBlocks.back();

	if (td < ctd || (td == ctd && host().chain().currentHash() == lh))
		return false;

	return true;
}

void PV60Sync::attemptSync(EthereumPeer* _peer)
{
	if (m_state != SyncState::Idle)
	{
		clog(NetAllDetail) << "Can't sync with this peer - outstanding asks.";
		return;
	}

	// if already done this, then ignore.
	if (!needsSyncing(_peer))
	{
		clog(NetAllDetail) << "Already synced with this peer.";
		return;
	}

	unsigned n = host().chain().number();
	u256 td = host().chain().details().totalDifficulty;
	if (host().bq().isActive())
		td += host().bq().difficulty();

	clog(NetAllDetail) << "Attempt chain-grab? Latest:" << (m_syncingLastReceivedHash ? m_syncingLastReceivedHash : m_syncingLatestHash) << ", number:" << n << ", TD:" << td << " versus " << _peer->m_totalDifficulty;
	if (td >= _peer->m_totalDifficulty)
	{
		clog(NetAllDetail) << "No. Our chain is better.";
		resetNeedsSyncing(_peer);
		transition(_peer, SyncState::Idle);
	}
	else
	{
		clog(NetAllDetail) << "Yes. Their chain is better.";
		m_estimatedHashes = _peer->m_expectedHashes - c_chainReorgSize;
		transition(_peer, SyncState::Hashes);
	}
}

void PV60Sync::noteNeedsSyncing(EthereumPeer* _peer)
{
	// if already downloading hash-chain, ignore.
	if (isSyncing())
	{
		clog(NetAllDetail) << "Sync in progress: Just set to help out.";
		if (m_state == SyncState::Blocks)
			requestBlocks(_peer);
	}
	else
		// otherwise check to see if we should be downloading...
		attemptSync(_peer);
}

void PV60Sync::changeSyncer(EthereumPeer* _syncer, bool _needHelp)
{
	if (_syncer)
		clog(NetAllDetail) << "Changing syncer to" << _syncer->session()->socketId();
	else
		clog(NetAllDetail) << "Clearing syncer.";

	m_syncer = _syncer;
	if (isSyncing())
	{
		if (_needHelp && (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks))
			host().foreachPeer([&](EthereumPeer* _p)
			{
				clog(NetNote) << "Getting help with downloading blocks";
				if (_p != _syncer && _p->m_asking == Asking::Nothing)
					transition(_p, m_state);
				return true;
			});
	}
	else
	{
		// start grabbing next hash chain if there is one.
		host().foreachPeer([this](EthereumPeer* _p)
		{
			attemptSync(_p);
			return !isSyncing();
		});
		if (!isSyncing())
		{
			if (m_state != SyncState::Idle)
				setState(_syncer, SyncState::Idle);
			clog(NetNote) << "No more peers to sync with.";
		}
	}
	assert(!!m_syncer || m_state == SyncState::Idle);
}

void PV60Sync::peerDoneBlocks(EthereumPeer* _peer)
{
	noteDoneBlocks(_peer, false);
}

void PV60Sync::noteDoneBlocks(EthereumPeer* _peer, bool _clemency)
{
	resetNeedsSyncing(_peer);
	if (downloadMan().isComplete())
	{
		// Done our chain-get.
		clog(NetNote) << "Chain download complete.";
		// 1/100th for each useful block hash.
		_peer->addRating(downloadMan().chainSize() / 100);
		downloadMan().reset();
	}
	else if (isSyncing(_peer))
	{
		if (_clemency)
			clog(NetNote) << "Chain download failed. Aborted while incomplete.";
		else
		{
			// Done our chain-get.
			clog(NetWarn) << "Chain download failed. Peer with blocks didn't have them all. This peer is bad and should be punished.";
			clog(NetWarn) << downloadMan().remaining();
			clog(NetWarn) << "WOULD BAN.";
//			m_banned.insert(_peer->session()->id());			// We know who you are!
//			_peer->disable("Peer sent hashes but was unable to provide the blocks.");
		}
		resetSync();
		downloadMan().reset();
		transition(_peer, SyncState::Idle);
	}
	_peer->m_sub.doneFetch();
}

void PV60Sync::onPeerHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	_peer->setIdle();
	if (!isSyncing(_peer))
	{
		clog(NetMessageSummary) << "Ignoring hashes since not syncing";
		return;
	}
	if (_hashes.size() == 0)
	{
		transition(_peer, SyncState::Blocks);
		return;
	}
	unsigned knowns = 0;
	unsigned unknowns = 0;
	for (unsigned i = 0; i < _hashes.size(); ++i)
	{
		auto h = _hashes[i];
		auto status = host().bq().blockStatus(h);
		if (status == QueueStatus::Importing || status == QueueStatus::Ready || host().chain().isKnown(h))
		{
			clog(NetMessageSummary) << "block hash ready:" << h << ". Start blocks download...";
			assert (isSyncing(_peer));
			transition(_peer, SyncState::Blocks);
			return;
		}
		else if (status == QueueStatus::Bad)
		{
			cwarn << "block hash bad!" << h << ". Bailing...";
			transition(_peer, SyncState::Idle);
			return;
		}
		else if (status == QueueStatus::Unknown)
		{
			unknowns++;
			m_syncingNeededBlocks.push_back(h);
		}
		else
			knowns++;
		m_syncingLastReceivedHash = h;
	}
	clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns; now at" << m_syncingLastReceivedHash;
	if (m_syncingNeededBlocks.size() > _peer->m_expectedHashes)
	{
		_peer->disable("Too many hashes");
		restartSync();
		return;
	}
	// run through - ask for more.
	transition(_peer, SyncState::Hashes);
	DEV_INVARIANT_CHECK;
}

void PV60Sync::onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if (isSyncing())
	{
		clog(NetMessageSummary) << "Ignoring since we're already downloading.";
		return;
	}
	clog(NetMessageDetail) << "Not syncing and new block hash discovered: syncing without help.";
	unsigned knowns = 0;
	unsigned unknowns = 0;
	for (auto const& h: _hashes)
	{
		_peer->addRating(1);
		DEV_GUARDED(_peer->x_knownBlocks)
			_peer->m_knownBlocks.insert(h);
		auto status = host().bq().blockStatus(h);
		if (status == QueueStatus::Importing || status == QueueStatus::Ready || host().chain().isKnown(h))
			knowns++;
		else if (status == QueueStatus::Bad)
		{
			cwarn << "block hash bad!" << h << ". Bailing...";
			return;
		}
		else if (status == QueueStatus::Unknown)
		{
			unknowns++;
			m_syncingNeededBlocks.push_back(h);
		}
		else
			knowns++;
	}
	clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns";
	if (unknowns > 0)
	{
		clog(NetNote) << "Not syncing and new block hash discovered: syncing without help.";
		downloadMan().resetToChain(m_syncingNeededBlocks);
		resetSync();
		transition(_peer, SyncState::NewBlocks, false, false);
	}
	DEV_INVARIANT_CHECK;
}

void PV60Sync::abortSync(EthereumPeer* _peer)
{
	// Can't check invariants here since the peers is already removed from the list and the state is not updated yet.
	if (isSyncing(_peer))
	{
		host().foreachPeer([this](EthereumPeer* _p) { _p->setIdle(); return true; });
		transition(_peer, SyncState::Idle, true);
	}
	DEV_INVARIANT_CHECK;
}

void PV60Sync::onPeerAborting(EthereumPeer* _peer)
{
	RecursiveGuard l(x_sync);
	// Can't check invariants here since the peers is already removed from the list and the state is not updated yet.
	abortSync(_peer);
	DEV_INVARIANT_CHECK;
}

bool PV60Sync::invariants() const
{
	if (m_state == SyncState::Idle && !!m_syncer)
		return false;
	if (m_state != SyncState::Idle && !m_syncer)
		return false;
	if (m_state == SyncState::Hashes)
	{
		bool hashes = false;
		host().foreachPeer([&](EthereumPeer* _p) { if (_p->m_asking == Asking::Hashes) hashes = true; return !hashes; });
		if (!hashes)
			return false;
		if (!m_syncingLatestHash)
			return false;
		if (m_syncingNeededBlocks.empty() != (!m_syncingLastReceivedHash))
			return false;
	}
	if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks)
	{
		bool blocks = false;
		host().foreachPeer([&](EthereumPeer* _p) { if (_p->m_asking == Asking::Blocks) blocks = true; return !blocks; });
		if (!blocks)
			return false;
		if (downloadMan().isComplete())
			return false;
	}
	if (m_state == SyncState::Idle)
	{
		bool busy = false;
		host().foreachPeer([&](EthereumPeer* _p) { if (_p->m_asking != Asking::Nothing && _p->m_asking != Asking::State) busy = true; return !busy; });
		if (busy)
			return false;
	}
	if (m_state == SyncState::Waiting && !host().bq().isActive())
		return false;
	return true;
}
