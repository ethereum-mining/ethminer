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
/** @file BlockChainSync.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>

#include <libdevcore/Guards.h>
#include <libdevcore/RangeMask.h>
#include <libethcore/Common.h>
#include <libp2p/Common.h>
#include "CommonNet.h"
#include "DownloadMan.h"

namespace dev
{

class RLPStream;

namespace eth
{

class EthereumHost;
class BlockQueue;
class EthereumPeer;

/**
 * @brief Base BlockChain synchronization strategy class.
 * Syncs to peers and keeps up to date. Base class handles blocks downloading but does not contain any details on state transfer logic.
 */
class BlockChainSync: public HasInvariants
{
public:
	BlockChainSync(EthereumHost& _host);
	virtual ~BlockChainSync();
	void abortSync(); ///< Abort all sync activity

	DownloadMan const& downloadMan() const;
	DownloadMan& downloadMan();

	/// @returns true is Sync is in progress
	virtual bool isSyncing() const = 0;

	/// Called by peer to report status
	virtual void onPeerStatus(EthereumPeer* _peer);

	/// Called by peer once it has new blocks during syn
	virtual void onPeerBlocks(EthereumPeer* _peer, RLP const& _r);

	/// Called by peer once it has new blocks
	virtual void onPeerNewBlock(EthereumPeer* _peer, RLP const& _r);

	/// Called by peer once it has new hashes
	virtual void onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes) = 0;

	/// Called by peer once it has another sequential block of hashes during sync
	virtual void onPeerHashes(EthereumPeer* _peer, h256s const& _hashes) = 0;

	/// Called by peer when it is disconnecting
	virtual void onPeerAborting(EthereumPeer* _peer) = 0;

	/// @returns Synchonization status
	virtual SyncStatus status() const = 0;

	static char const* stateName(SyncState _s) { return s_stateNames[static_cast<int>(_s)]; }

protected:
	//To be implemented in derived classes:
	/// New valid peer appears
	virtual void onNewPeer(EthereumPeer* _peer) = 0;

	/// Peer done downloading blocks
	virtual void peerDoneBlocks(EthereumPeer* _peer) = 0;

	/// Resume downloading after witing state
	virtual void continueSync() = 0;

	/// Restart sync
	virtual void restartSync() = 0;

	/// Called after all blocks have been donloaded
	virtual void completeSync() = 0;

	/// Enter waiting state
	virtual void pauseSync() = 0;

	/// Restart sync for given peer
	virtual void resetSyncFor(EthereumPeer* _peer, h256 const& _latestHash, u256 const& _td) = 0;

	EthereumHost& host() { return m_host; }
	EthereumHost const& host() const { return m_host; }

	/// Estimates max number of hashes peers can give us.
	unsigned estimateHashes() const;

	/// Request blocks from peer if needed
	void requestBlocks(EthereumPeer* _peer);

protected:
	Handler m_bqRoomAvailable;
	mutable RecursiveMutex x_sync;
	SyncState m_state = SyncState::Idle;			///< Current sync state
	SyncState m_lastActiveState = SyncState::Idle; 	///< Saved state before entering waiting queue mode
	unsigned m_estimatedHashes = 0;					///< Number of estimated hashes for the last peer over PV60. Used for status reporting only.

private:
	static char const* const s_stateNames[static_cast<int>(SyncState::Size)];
	bool invariants() const override = 0;
	EthereumHost& m_host;
	HashDownloadMan m_hashMan;
};


/**
 * @brief Syncrhonization over PV60. Selects a single peer and tries to downloading hashes from it. After hash downaload is complete
 * Syncs to peers and keeps up to date
 */
class PV60Sync: public BlockChainSync
{
public:
	PV60Sync(EthereumHost& _host);

	/// @returns true is Sync is in progress
	bool isSyncing() const override { return !!m_syncer; }

	/// Called by peer once it has new hashes
	void onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes) override;

	/// Called by peer once it has another sequential block of hashes during sync
	void onPeerHashes(EthereumPeer* _peer, h256s const& _hashes) override;

	/// Called by peer when it is disconnecting
	void onPeerAborting(EthereumPeer* _peer) override;

	/// @returns Sync status
	SyncStatus status() const override;

	void onNewPeer(EthereumPeer* _peer) override;
	void continueSync() override;
	void peerDoneBlocks(EthereumPeer* _peer) override;
	void restartSync() override;
	void completeSync() override;
	void pauseSync() override;
	void resetSyncFor(EthereumPeer* _peer, h256 const& _latestHash, u256 const& _td) override;

private:
	/// Transition sync state in a particular direction. @param _peer Peer that is responsible for state tranfer
	void transition(EthereumPeer* _peer, SyncState _s, bool _force = false, bool _needHelp = true);

	/// Reset peer syncing requirements state.
	void resetNeedsSyncing(EthereumPeer* _peer) { setNeedsSyncing(_peer, h256(), 0); }

	/// Update peer syncing requirements state.
	void setNeedsSyncing(EthereumPeer* _peer, h256 const& _latestHash, u256 const& _td);

	/// Do we presently need syncing with this peer?
	bool needsSyncing(EthereumPeer* _peer) const;

	/// Check whether the session should bother grabbing blocks from a peer.
	bool shouldGrabBlocks(EthereumPeer* _peer) const;

	/// Attempt to begin syncing with the peer; first check the peer has a more difficlult chain to download, then start asking for hashes, then move to blocks
	void attemptSync(EthereumPeer* _peer);

	/// Update our syncing state
	void setState(EthereumPeer* _peer, SyncState _s, bool _isSyncing = false, bool _needHelp = false);

	/// Check if peer is main syncer
	bool isSyncing(EthereumPeer* _peer) const;

	/// Check if we need (re-)syncing with the peer.
	void noteNeedsSyncing(EthereumPeer* _who);

	/// Set main syncing peer
	void changeSyncer(EthereumPeer* _syncer, bool _needHelp);

	/// Called when peer done downloading blocks
	void noteDoneBlocks(EthereumPeer* _who, bool _clemency);

	/// Abort syncing for peer
	void abortSync(EthereumPeer* _peer);

	/// Reset hash chain syncing
	void resetSync();

	bool invariants() const override;

	h256s m_syncingNeededBlocks;				///< The blocks that we should download from this peer.
	h256 m_syncingLastReceivedHash;				///< Hash most recently received from peer.
	h256 m_syncingLatestHash;					///< Peer's latest block's hash, as of the current sync.
	u256 m_syncingTotalDifficulty;				///< Peer's latest block's total difficulty, as of the current sync.
	EthereumPeer* m_syncer = nullptr;	// TODO: switch to weak_ptr
};
}
}
