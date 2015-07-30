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

unsigned const c_chainReorgSize = 30000; /// Added to estimated hashes to account for potential chain reorganiation
unsigned const c_hashSubchainSize = 8192; /// PV61 subchain size

std::ostream& dev::eth::operator<<(std::ostream& _out, SyncStatus const& _sync)
{
	_out << "protocol: " << _sync.protocolVersion << endl;
	_out << "state: " << EthereumHost::stateName(_sync.state) << " ";
	if (_sync.state == SyncState::Hashes)
		_out << _sync.hashesReceived << "/" << (_sync.hashesEstimated ? "~" : "") << _sync.hashesTotal;
	if (_sync.state == SyncState::Blocks || _sync.state == SyncState::NewBlocks)
		_out << _sync.blocksReceived << "/" << _sync.blocksTotal;
	return _out;
}

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
	downloadMan().reset();
}

void BlockChainSync::onPeerStatus(std::shared_ptr<EthereumPeer> _peer)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	std::shared_ptr<Session> session = _peer->session();
	if (!session)
		return; // Expired
	if (_peer->m_genesisHash != host().chain().genesisHash())
		_peer->disable("Invalid genesis hash");
	else if (_peer->m_protocolVersion != host().protocolVersion() && _peer->m_protocolVersion != EthereumHost::c_oldProtocolVersion)
		_peer->disable("Invalid protocol version.");
	else if (_peer->m_networkId != host().networkId())
		_peer->disable("Invalid network identifier.");
	else if (session->info().clientVersion.find("/v0.7.0/") != string::npos)
		_peer->disable("Blacklisted client version.");
	else if (host().isBanned(session->id()))
		_peer->disable("Peer banned for previous bad behaviour.");
	else
	{
		unsigned hashes = estimatedHashes();
		_peer->m_expectedHashes = hashes;
		onNewPeer(_peer);
	}
}

unsigned BlockChainSync::estimatedHashes() const
{
	BlockInfo block = host().chain().info();
	time_t lastBlockTime = (block.hash() == host().chain().genesisHash()) ? 1428192000 : (time_t)block.timestamp();
	time_t now = time(0);
	unsigned blockCount = c_chainReorgSize;
	if (lastBlockTime > now)
		clog(NetWarn) << "Clock skew? Latest block is in the future";
	else
		blockCount += (now - lastBlockTime) / (unsigned)c_durationLimit;
	clog(NetAllDetail) << "Estimated hashes: " << blockCount;
	return blockCount;
}

void BlockChainSync::requestBlocks(std::shared_ptr<EthereumPeer> _peer)
{
	if (host().bq().knownFull())
	{
		clog(NetAllDetail) << "Waiting for block queue before downloading blocks";
		pauseSync();
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

void BlockChainSync::logNewBlock(h256 const& _h)
{
	if (m_state == SyncState::NewBlocks)
		clog(NetNote) << "NewBlock: " << _h;
	m_knownNewHashes.erase(_h);
}

void BlockChainSync::onPeerBlocks(std::shared_ptr<EthereumPeer> _peer, RLP const& _r)
{
	RecursiveGuard l(x_sync);
	unsigned itemCount = _r.itemCount();
	clog(NetMessageSummary) << "Blocks (" << dec << itemCount << "entries)" << (itemCount ? "" : ": NoMoreBlocks");

	if (m_state != SyncState::Blocks && m_state != SyncState::NewBlocks && m_state != SyncState::Waiting)
	{
		clog(NetMessageSummary) << "Ignoring unexpected blocks";
		return;
	}
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
		auto h = BlockInfo::headerHashFromBlock(_r[i].data());
		if (_peer->m_sub.noteBlock(h))
		{
			_peer->addRating(10);
			switch (host().bq().import(_r[i].data()))
			{
			case ImportResult::Success:
				success++;
				logNewBlock(h);
				break;

			case ImportResult::Malformed:
			case ImportResult::BadChain:
				logNewBlock(h);
				_peer->disable("Malformed block received.");
				restartSync();
				return;

			case ImportResult::FutureTimeKnown:
				logNewBlock(h);
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
				logNewBlock(h);
				if (m_state == SyncState::NewBlocks)
				{
					BlockInfo bi(_r[i].data());
					if (bi.number() > maxUnknownNumber)
					{
						maxUnknownNumber = bi.number();
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

void BlockChainSync::onPeerNewBlock(std::shared_ptr<EthereumPeer> _peer, RLP const& _r)
{
	DEV_INVARIANT_CHECK;
	RecursiveGuard l(x_sync);
	auto h = BlockInfo::headerHashFromBlock(_r[0].data());

	if (_r.itemCount() != 2)
		_peer->disable("NewBlock without 2 data fields.");
	else
	{
		switch (host().bq().import(_r[0].data()))
		{
		case ImportResult::Success:
			_peer->addRating(100);
			logNewBlock(h);
			break;
		case ImportResult::FutureTimeKnown:
			//TODO: Rating dependent on how far in future it is.
			break;

		case ImportResult::Malformed:
		case ImportResult::BadChain:
			logNewBlock(h);
			_peer->disable("Malformed block received.");
			return;

		case ImportResult::AlreadyInChain:
		case ImportResult::AlreadyKnown:
			break;

		case ImportResult::FutureTimeUnknown:
		case ImportResult::UnknownParent:
		{
			logNewBlock(h);
			u256 totalDifficulty = _r[1].toInt<u256>();
			if (totalDifficulty > _peer->m_totalDifficulty)
			{
				clog(NetMessageDetail) << "Received block with no known parent. Peer needs syncing...";
				resetSyncFor(_peer, h, totalDifficulty);
			}
			break;
		}
		default:;
		}

		DEV_GUARDED(_peer->x_knownBlocks)
			_peer->m_knownBlocks.insert(h);
	}
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
	res.protocolVersion = EthereumHost::c_oldProtocolVersion;
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

void PV60Sync::setState(std::shared_ptr<EthereumPeer> _peer, SyncState _s, bool _isSyncing, bool _needHelp)
{
	bool changedState = (m_state != _s);
	m_state = _s;

	if (_isSyncing != (m_syncer.lock() == _peer) || (_isSyncing && changedState))
		changeSyncer(_isSyncing ? _peer : nullptr, _needHelp);
	else if (_s == SyncState::Idle)
		changeSyncer(nullptr, _needHelp);

	assert(isSyncing() || _s == SyncState::Idle);
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
	std::shared_ptr<EthereumPeer> syncer = m_syncer.lock();
	if (syncer)
		transition(syncer, SyncState::Idle);
}

void PV60Sync::completeSync()
{
	std::shared_ptr<EthereumPeer> syncer = m_syncer.lock();
	if (syncer)
		transition(syncer, SyncState::Idle);
}

void PV60Sync::pauseSync()
{
	std::shared_ptr<EthereumPeer> syncer = m_syncer.lock();
	if (syncer)
		transition(syncer, SyncState::Waiting, true);
}

void PV60Sync::continueSync()
{
	std::shared_ptr<EthereumPeer> syncer = m_syncer.lock();
	if (syncer)
		transition(syncer, SyncState::Blocks);
}

void PV60Sync::onNewPeer(std::shared_ptr<EthereumPeer> _peer)
{
	setNeedsSyncing(_peer, _peer->m_latestHash, _peer->m_totalDifficulty);
}

void PV60Sync::transition(std::shared_ptr<EthereumPeer> _peer, SyncState _s, bool _force, bool _needHelp)
{
	clog(NetMessageSummary) << "Transition!" << EthereumHost::stateName(_s) << "from" << EthereumHost::stateName(m_state) << ", " << (isSyncing(_peer) ? "syncing" : "holding") << (needsSyncing(_peer) ? "& needed" : "");

	if (m_state == SyncState::Idle && _s != SyncState::Idle)
		_peer->m_requireTransactions = true;

	if (_s == SyncState::Hashes)
	{
		if (m_state == SyncState::Idle || m_state == SyncState::Hashes)
		{
			m_estimatedHashes = _peer->m_expectedHashes - c_chainReorgSize;
			syncHashes(_peer);
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
				clog(NetMessageDetail) << "Difficulty of hashchain HIGHER. Grabbing" << m_syncingNeededBlocks.size() << "blocks [latest now" << m_syncingLatestHash << ", was" << host().latestBlockSent() << "]";
				downloadMan().resetToChain(m_syncingNeededBlocks);
				resetSync();
			}
			else
			{
				clog(NetMessageDetail) << "Difficulty of hashchain not HIGHER. Ignoring.";
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
		if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks)
		{
			clog(NetMessageDetail) << "Finishing blocks fetch...";

			// a bit overkill given that the other nodes may yet have the needed blocks, but better to be safe than sorry.
			if (isSyncing(_peer))
				noteDoneBlocks(_peer, _force);

			// NOTE: need to notify of giving up on chain-hashes, too, altering state as necessary.
			_peer->m_sub.doneFetch();
			setState(_peer, SyncState::Idle, false);
		}
		else if (m_state == SyncState::Hashes)
		{
			clog(NetMessageDetail) << "Finishing hashes fetch...";
			setState(_peer, SyncState::Idle, false);
		}
		// Otherwise it's fine. We don't care if it's Nothing->Nothing.
		DEV_INVARIANT_CHECK;
		return;
	}

	clog(NetWarn) << "Invalid state transition:" << EthereumHost::stateName(_s) << "from" << EthereumHost::stateName(m_state) << ", " << (isSyncing(_peer) ? "syncing" : "holding") << (needsSyncing(_peer) ? "& needed" : "");
}

void PV60Sync::resetSyncFor(std::shared_ptr<EthereumPeer> _peer, h256 const& _latestHash, u256 const& _td)
{
	setNeedsSyncing(_peer, _latestHash, _td);
}

void PV60Sync::setNeedsSyncing(std::shared_ptr<EthereumPeer> _peer, h256 const& _latestHash, u256 const& _td)
{
	_peer->m_latestHash = _latestHash;
	_peer->m_totalDifficulty = _td;

	if (_peer->m_latestHash)
		noteNeedsSyncing(_peer);

	shared_ptr<Session> session = _peer->session();
	if (session)
		session->addNote("sync", string(isSyncing(_peer) ? "ongoing" : "holding") + (needsSyncing(_peer) ? " & needed" : ""));
}

bool PV60Sync::needsSyncing(std::shared_ptr<EthereumPeer> _peer) const
{
	return !!_peer->m_latestHash;
}

bool PV60Sync::isSyncing(std::shared_ptr<EthereumPeer> _peer) const
{
	return m_syncer.lock() == _peer;
}

bool PV60Sync::shouldGrabBlocks(std::shared_ptr<EthereumPeer> _peer) const
{
	auto td = _peer->m_totalDifficulty;
	auto lh = _peer->m_latestHash;
	auto ctd = host().chain().details().totalDifficulty;

	if (m_syncingNeededBlocks.empty())
		return false;

	clog(NetMessageDetail) << "Should grab blocks? " << td << "vs" << ctd << ";" << m_syncingNeededBlocks.size() << " blocks, ends" << m_syncingNeededBlocks.back();

	if (td < ctd || (td == ctd && host().chain().currentHash() == lh))
		return false;

	return true;
}

void PV60Sync::attemptSync(std::shared_ptr<EthereumPeer> _peer)
{
	if (m_state != SyncState::Idle || _peer->m_asking != Asking::Nothing)
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
		transition(_peer, SyncState::Hashes);
	}
}

void PV60Sync::noteNeedsSyncing(std::shared_ptr<EthereumPeer> _peer)
{
	// if already downloading hash-chain, ignore.
	if (isSyncing())
	{
		clog(NetAllDetail) << "Sync in progress: Just set to help out.";
		if (m_state == SyncState::Hashes && _peer->m_asking == Asking::Nothing)
			requestSubchain(_peer);
		else if (m_state == SyncState::Blocks)
			requestBlocks(_peer);
	}
	else
		// otherwise check to see if we should be downloading...
		attemptSync(_peer);
}

void PV60Sync::changeSyncer(std::shared_ptr<EthereumPeer> _syncer, bool _needHelp)
{
	if (_syncer)
		clog(NetAllDetail) << "Changing syncer to" << _syncer->session()->socketId();
	else
		clog(NetAllDetail) << "Clearing syncer.";

	m_syncer = _syncer;
	if (isSyncing())
	{
		if (_needHelp && (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks))
			host().foreachPeer([&](std::shared_ptr<EthereumPeer> _p)
			{
				clog(NetMessageDetail) << "Getting help with downloading blocks";
				if (_p != _syncer && _p->m_asking == Asking::Nothing)
					transition(_p, m_state);
				return true;
			});
	}
	else
	{
		// start grabbing next hash chain if there is one.
		host().foreachPeer([this](std::shared_ptr<EthereumPeer> _p)
		{
			attemptSync(_p);
			return !isSyncing();
		});
		if (!isSyncing())
		{
			if (m_state != SyncState::Idle)
				setState(_syncer, SyncState::Idle);
			clog(NetMessageDetail) << "No more peers to sync with.";
		}
	}
	assert(isSyncing() || m_state == SyncState::Idle);
}

void PV60Sync::peerDoneBlocks(std::shared_ptr<EthereumPeer> _peer)
{
	noteDoneBlocks(_peer, false);
}

void PV60Sync::noteDoneBlocks(std::shared_ptr<EthereumPeer> _peer, bool _clemency)
{
	resetNeedsSyncing(_peer);
	if (downloadMan().isComplete())
	{
		// Done our chain-get.
		clog(NetMessageDetail) << "Chain download complete.";
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
			// This can happen when the leading peer aborts and the one that is selected instead does not have all the blocks.
			// Just stop syncing to this peer. Sync will restart if there are no more peers to sync with.
			clog(NetNote) << "Peer does not have required blocks";
			resetNeedsSyncing(_peer);
		}
		resetSync();
		downloadMan().reset();
	}
	_peer->m_sub.doneFetch();
}

void PV60Sync::syncHashes(std::shared_ptr<EthereumPeer> _peer)
{
	if (m_state == SyncState::Idle)
	{
		if (isSyncing(_peer))
			clog(NetWarn) << "Bad state: not asking for Hashes, yet syncing!";

		m_syncingLatestHash = _peer->m_latestHash;
		m_syncingTotalDifficulty = _peer->m_totalDifficulty;
		setState(_peer, SyncState::Hashes, true);
		_peer->requestHashes(m_syncingLastReceivedHash ? m_syncingLastReceivedHash : m_syncingLatestHash);
	}
	else if (m_state == SyncState::Hashes)
	{
		if (!isSyncing(_peer))
			clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";

		setState(_peer, SyncState::Hashes, true);
		_peer->requestHashes(m_syncingLastReceivedHash);
	}
}

void PV60Sync::onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if (!isSyncing(_peer))
	{
		clog(NetMessageSummary) << "Ignoring hashes since not syncing";
		return;
	}
	if (_peer->m_syncHash != (m_syncingLastReceivedHash ? m_syncingLastReceivedHash : m_syncingLatestHash))
	{
		clog(NetMessageSummary) << "Ignoring unexpected hashes";
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
			_peer->disable("Bad blocks");
			restartSync();
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
}

void PV60Sync::onPeerNewHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	if (isSyncing() && (m_state != SyncState::NewBlocks || isSyncing(_peer)))
	{
		clog(NetMessageDetail) << "Ignoring new hashes since we're already downloading.";
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
		if (m_state == SyncState::NewBlocks)
		{
			clog(NetMessageDetail) << "Downloading new blocks and seeing new hashes. Trying grabbing blocks";
			_peer->requestBlocks(m_syncingNeededBlocks);
		}
		else
		{
			clog(NetMessageDetail) << "Not syncing and new block hash discovered: syncing without help.";
			downloadMan().resetToChain(m_syncingNeededBlocks);
			transition(_peer, SyncState::NewBlocks, false, false);
		}
		for (auto const& h: m_syncingNeededBlocks)
			if (!m_knownNewHashes.count(h))
			{
				m_knownNewHashes.insert(h);
				clog(NetNote) << "NewHash: " << h;
			}
		resetSync();
	}
}

void PV60Sync::abortSync()
{
	DEV_INVARIANT_CHECK;
	// Can't check invariants here since the peers is already removed from the list and the state is not updated yet.
	bool continueSync = false;
	if (m_state == SyncState::Blocks)
	{
		// Main syncer aborted, try to find a replacement
		host().foreachPeer([&](std::shared_ptr<EthereumPeer> _p)
		{
			if (_p->m_asking == Asking::Blocks)
			{
				setState(_p, SyncState::Blocks, true, true);		// will kick off other peers to help if available.
				continueSync = true;
				return false;
			}
			if (_p->m_asking == Asking::Nothing && shouldGrabBlocks(_p))
			{
				transition(_p, SyncState::Blocks);
				clog(NetMessageDetail) << "New sync peer selected";
				continueSync = true;
				return false;
			}
			return true;
		});
	}
	if (!continueSync)
	{
		// Just set to idle. Hashchain is keept, Sync will be continued if there are more peers to sync with
		setState(std::shared_ptr<EthereumPeer>(), SyncState::Idle, false, true);
	}
}

void PV60Sync::onPeerAborting()
{
	RecursiveGuard l(x_sync);
	DEV_INVARIANT_CHECK;
	// Can't check invariants here since the peers is already removed from the list and the state is not updated yet.
	if (m_syncer.expired() && m_state != SyncState::Idle)
	{
		clog(NetNote) << "Syncing peer disconnected";
		m_syncer.reset();
		abortSync();
	}
}

bool PV60Sync::invariants() const
{
	if (m_state == SyncState::Idle && isSyncing())
		BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Idle while peer syncing"));
	if (m_state != SyncState::Idle && !isSyncing())
		BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Active while peer not syncing"));
	if (m_state == SyncState::Hashes)
	{
		if (!m_syncingLatestHash)
			BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("m_syncingLatestHash is not set while downloading hashes"));
		if (m_syncingNeededBlocks.empty() != (!m_syncingLastReceivedHash))
			BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Received hashes but the hashes list is empty (or the other way around)"));
	}
	if (m_state == SyncState::Blocks || m_state == SyncState::NewBlocks)
	{
		if (downloadMan().isComplete())
			BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Block download complete but the state is still Blocks"));
	}
	if (m_state == SyncState::Waiting && !host().bq().isActive())
		BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Waiting while block queue is idle"));
	return true;
}

PV61Sync::PV61Sync(EthereumHost& _host):
	PV60Sync(_host)
{
}

void PV61Sync::syncHashes(std::shared_ptr<EthereumPeer> _peer)
{
	if (_peer->m_protocolVersion != host().protocolVersion())
	{
		m_readyChainMap.clear();
		m_completeChainMap.clear();
		m_downloadingChainMap.clear();
		m_syncingBlockNumber = 0;
		m_chainSyncPeers.clear();
		m_knownHashes.clear();
		m_hashScanComplete = false;
		PV60Sync::syncHashes(_peer);
		return;
	}
	if (m_state == SyncState::Idle)
	{
		if (isSyncing(_peer))
			clog(NetWarn) << "Bad state: not asking for Hashes, yet syncing!";

		if (m_syncingBlockNumber == 0)
			m_syncingBlockNumber = host().chain().number() + c_hashSubchainSize;
		m_syncingTotalDifficulty = _peer->m_totalDifficulty;
		setState(_peer, SyncState::Hashes, true);
		_peer->requestHashes(m_syncingBlockNumber, 1);
	}
	else if (m_state == SyncState::Hashes)
	{
		if (!isSyncing(_peer))
			clog(NetWarn) << "Bad state: asking for Hashes yet not syncing!";

		m_syncingBlockNumber += c_hashSubchainSize;
		setState(_peer, SyncState::Hashes, true);
		_peer->requestHashes(m_syncingBlockNumber, 1);
	}
}

void PV61Sync::requestSubchain(std::shared_ptr<EthereumPeer> _peer)
{
	auto syncPeer = m_chainSyncPeers.find(_peer);
	if (syncPeer != m_chainSyncPeers.end())
	{
		// Already downoading, request next batch
		SubChain const& s = m_downloadingChainMap.at(syncPeer->second);
		_peer->requestHashes(s.lastHash);
	}
	else if (needsSyncing(_peer))
	{
		if (!m_readyChainMap.empty())
		{
			clog(NetAllDetail) << "Helping with hashchin download";
			SubChain& s = m_readyChainMap.begin()->second;
			_peer->requestHashes(s.lastHash);
			m_downloadingChainMap[m_readyChainMap.begin()->first] = move(s);
			m_chainSyncPeers[_peer] = m_readyChainMap.begin()->first;
			m_readyChainMap.erase(m_readyChainMap.begin());
		}
		else if (!m_downloadingChainMap.empty() && m_hashScanComplete)
			// Lead syncer is done, just grab whatever we can
			_peer->requestHashes(m_downloadingChainMap.begin()->second.lastHash);
	}
}

void PV61Sync::requestSubchains()
{
	host().foreachPeer([this](std::shared_ptr<EthereumPeer> _p)
	{
		if (_p->m_asking == Asking::Nothing)
			requestSubchain(_p);
		return true;
	});
}

void PV61Sync::completeSubchain(std::shared_ptr<EthereumPeer> _peer, unsigned _n)
{
	m_completeChainMap[_n] = move(m_downloadingChainMap.at(_n));
	m_downloadingChainMap.erase(_n);
	for (auto s = m_chainSyncPeers.begin(); s != m_chainSyncPeers.end(); ++s)
		if (s->second == _n) //TODO: optimize this
		{
			m_chainSyncPeers.erase(s);
			break;
		}

	_peer->m_syncHashNumber = 0;

	auto syncer = m_syncer.lock();
	if (!syncer)
	{
		restartSync();
		return;
	}

	if (m_readyChainMap.empty() && m_downloadingChainMap.empty() && m_hashScanComplete)
	{
		//Done chain-get
		m_syncingNeededBlocks.clear();
		// Add hashes to download skipping onces that are already downloaded
		for (auto h = m_completeChainMap.rbegin(); h != m_completeChainMap.rend(); ++h)
			if (!host().chain().isKnown(h->second.hashes.front()) && !host().chain().isKnown(h->second.hashes.back()))
			{
				if (host().bq().blockStatus(h->second.hashes.front()) == QueueStatus::Unknown || host().bq().blockStatus(h->second.hashes.back()) == QueueStatus::Unknown)
					m_syncingNeededBlocks.insert(m_syncingNeededBlocks.end(), h->second.hashes.begin(), h->second.hashes.end());
				else
					for (h256 const& hash: h->second.hashes)
						if (!host().chain().isKnown(hash) && host().bq().blockStatus(hash) == QueueStatus::Unknown)
							m_syncingNeededBlocks.insert(m_syncingNeededBlocks.end(), hash);
			}
		transition(syncer, SyncState::Blocks);
	}
	else
		requestSubchain(_peer);
}

void PV61Sync::restartSync()
{
	m_completeChainMap.clear();
	m_readyChainMap.clear();
	m_downloadingChainMap.clear();
	m_chainSyncPeers.clear();
	m_syncingBlockNumber = 0;
	m_knownHashes.clear();
	m_hashScanComplete = false;
	PV60Sync::restartSync();
}

void PV61Sync::onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes)
{
	RecursiveGuard l(x_sync);
	if (m_syncingBlockNumber == 0 || (_peer == m_syncer.lock() && _peer->m_protocolVersion != host().protocolVersion()))
	{
		// Syncing in pv60 mode
		PV60Sync::onPeerHashes(_peer, _hashes);
		return;
	}
	if (_hashes.size() == 0)
	{
		if (isSyncing(_peer) && _peer->m_syncHashNumber == m_syncingBlockNumber)
		{
			// End of hash chain, add last chunk to download
			m_readyChainMap.insert(make_pair(m_syncingBlockNumber, SubChain{ h256s{ _peer->m_latestHash }, _peer->m_latestHash }));
			m_knownHashes.insert(_peer->m_latestHash);
			m_hashScanComplete = true;
			_peer->m_syncHashNumber = 0;
			requestSubchain(_peer);
		}
		else
		{
			auto syncPeer = m_chainSyncPeers.find(_peer);
			if (syncPeer == m_chainSyncPeers.end())
				clog(NetMessageDetail) << "Hashes response from unexpected peer";
			else
			{
				// Peer does not have request hashes, move back from downloading to ready
				unsigned number = syncPeer->second;
				m_chainSyncPeers.erase(_peer);
				m_readyChainMap[number] = move(m_downloadingChainMap.at(number));
				m_downloadingChainMap.erase(number);
				resetNeedsSyncing(_peer);
				requestSubchains();
			}
		}
		return;
	}
	if (isSyncing(_peer) && _peer->m_syncHashNumber == m_syncingBlockNumber)
	{
		// Got new subchain marker
		if (_hashes.size() != 1)
		{
			clog(NetWarn) << "Peer sent too many hashes";
			_peer->disable("Too many hashes");
			restartSync();
			return;
		}
		m_knownHashes.insert(_hashes[0]);
		m_readyChainMap.insert(make_pair(m_syncingBlockNumber, SubChain{ h256s{ _hashes[0] }, _hashes[0] }));
		if ((m_readyChainMap.size() + m_downloadingChainMap.size() + m_completeChainMap.size()) * c_hashSubchainSize > _peer->m_expectedHashes)
		{
			_peer->disable("Too many hashes from lead peer");
			restartSync();
			return;
		}
		transition(_peer, SyncState::Hashes);
		requestSubchains();
	}
	else
	{
		auto syncPeer = m_chainSyncPeers.find(_peer);
		unsigned number = 0;
		if (syncPeer == m_chainSyncPeers.end())
		{
			//check downlading peers
			for (auto const& downloader: m_downloadingChainMap)
				if (downloader.second.lastHash == _peer->m_syncHash)
				{
					number = downloader.first;
					break;
				}
		}
		else
			number = syncPeer->second;
		if (number == 0)
		{
			clog(NetAllDetail) << "Hashes response from unexpected/expired peer";
			return;
		}

		auto downloadingPeer = m_downloadingChainMap.find(number);
		if (downloadingPeer == m_downloadingChainMap.end() || downloadingPeer->second.lastHash != _peer->m_syncHash)
		{
			// Too late, other peer has already downloaded our hashes
			m_chainSyncPeers.erase(_peer);
			requestSubchain(_peer);
			return;
		}

		SubChain& subChain = downloadingPeer->second;
		unsigned knowns = 0;
		unsigned unknowns = 0;
		for (unsigned i = 0; i < _hashes.size(); ++i)
		{
			auto h = _hashes[i];
			auto status = host().bq().blockStatus(h);
			if (status == QueueStatus::Importing || status == QueueStatus::Ready || host().chain().isKnown(h) || !!m_knownHashes.count(h))
			{
				clog(NetMessageSummary) << "Subchain download complete";
				m_chainSyncPeers.erase(_peer);
				completeSubchain(_peer, number);
				return;
			}
			else if (status == QueueStatus::Bad)
			{
				cwarn << "block hash bad!" << h << ". Bailing...";
				_peer->disable("Bad hashes");
				if (isSyncing(_peer))
					restartSync();
				else
				{
					//try with other peer
					m_readyChainMap[number] = move(m_downloadingChainMap.at(number));
					m_downloadingChainMap.erase(number);
					m_chainSyncPeers.erase(_peer);
				}
				return;
			}
			else if (status == QueueStatus::Unknown)
			{
				unknowns++;
				subChain.hashes.push_back(h);
			}
			else
				knowns++;
			subChain.lastHash = h;
		}
		clog(NetMessageSummary) << knowns << "knowns," << unknowns << "unknowns; now at" << subChain.lastHash;
		if (subChain.hashes.size() > c_hashSubchainSize)
		{
			_peer->disable("Too many subchain hashes");
			restartSync();
			return;
		}
		requestSubchain(_peer);
	}
	DEV_INVARIANT_CHECK;
}

void PV61Sync::onPeerAborting()
{
	RecursiveGuard l(x_sync);
	// Can't check invariants here since the peers is already removed from the list and the state is not updated yet.
	for (auto s = m_chainSyncPeers.begin(); s != m_chainSyncPeers.end();)
	{
		if (s->first.expired())
		{
			unsigned number = s->second;
			auto downloading = m_downloadingChainMap.find(number);
			if (downloading != m_downloadingChainMap.end())
			{
				m_readyChainMap[number] = move(downloading->second);
				m_downloadingChainMap.erase(number);
			}
			m_chainSyncPeers.erase(s++);
		}
		else
			++s;
	}
	if (m_syncer.expired())
	{
		if (m_state == SyncState::Hashes)
		{
			// Main syncer aborted, other peers are probably still downloading hashes, just set one of them as syncer
			host().foreachPeer([&](std::shared_ptr<EthereumPeer> _p)
			{
				if (_p->m_asking != Asking::Hashes)
					return true;
				setState(_p, SyncState::Hashes, true, true);
				return false;
			});
		}

		if (m_syncer.expired())
			PV60Sync::onPeerAborting();
	}
	else if (isPV61Syncing() && m_state == SyncState::Hashes)
		requestSubchains();
	DEV_INVARIANT_CHECK;
}

SyncStatus PV61Sync::status() const
{
	RecursiveGuard l(x_sync);
	SyncStatus res = PV60Sync::status();
	res.protocolVersion = 61;
	if (m_state == SyncState::Hashes && isPV61Syncing())
	{
		res.hashesReceived = 0;
		for (auto const& d : m_readyChainMap)
			res.hashesReceived += d.second.hashes.size();
		for (auto const& d : m_downloadingChainMap)
			res.hashesReceived += d.second.hashes.size();
		for (auto const& d : m_completeChainMap)
			res.hashesReceived += d.second.hashes.size();
	}
	return res;
}

bool PV61Sync::isPV61Syncing() const
{
	return m_syncingBlockNumber != 0;
}

void PV61Sync::completeSync()
{
	m_completeChainMap.clear();
	m_knownHashes.clear();
	m_syncingBlockNumber = 0;
	PV60Sync::completeSync();
}

bool PV61Sync::invariants() const
{
	if (m_state == SyncState::Hashes)
	{
		if (isPV61Syncing() && !m_syncingBlockNumber)
			BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment("Syncing in PV61 with no block number set"));
	}
	else if (!PV60Sync::invariants())
			return false;
	return true;
}
