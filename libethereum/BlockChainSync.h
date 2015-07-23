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

	/// Restart sync
	virtual void restartSync() = 0;

	/// Called by peer to report status
	virtual void onPeerStatus(std::shared_ptr<EthereumPeer> _peer);

	/// Called by peer once it has new blocks during syn
	virtual void onPeerBlocks(std::shared_ptr<EthereumPeer> _peer, RLP const& _r);

	/// Called by peer once it has new blocks
	virtual void onPeerNewBlock(std::shared_ptr<EthereumPeer> _peer, RLP const& _r);

	/// Called by peer once it has new hashes
	virtual void onPeerNewHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes) = 0;

	/// Called by peer once it has another sequential block of hashes during sync
	virtual void onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes) = 0;

	/// Called by peer when it is disconnecting
	virtual void onPeerAborting() = 0;

	/// @returns Synchonization status
	virtual SyncStatus status() const = 0;

	static char const* stateName(SyncState _s) { return s_stateNames[static_cast<int>(_s)]; }

protected:
	//To be implemented in derived classes:
	/// New valid peer appears
	virtual void onNewPeer(std::shared_ptr<EthereumPeer> _peer) = 0;

	/// Peer done downloading blocks
	virtual void peerDoneBlocks(std::shared_ptr<EthereumPeer> _peer) = 0;

	/// Resume downloading after witing state
	virtual void continueSync() = 0;

	/// Called after all blocks have been donloaded
	virtual void completeSync() = 0;

	/// Enter waiting state
	virtual void pauseSync() = 0;

	/// Restart sync for given peer
	virtual void resetSyncFor(std::shared_ptr<EthereumPeer> _peer, h256 const& _latestHash, u256 const& _td) = 0;

	EthereumHost& host() { return m_host; }
	EthereumHost const& host() const { return m_host; }

	/// Estimates max number of hashes peers can give us.
	unsigned estimatedHashes() const;

	/// Request blocks from peer if needed
	void requestBlocks(std::shared_ptr<EthereumPeer> _peer);

protected:
	Handler<> m_bqRoomAvailable;			///< Triggered once block queue
	mutable RecursiveMutex x_sync;
	SyncState m_state = SyncState::Idle;	///< Current sync state
	unsigned m_estimatedHashes = 0;			///< Number of estimated hashes for the last peer over PV60. Used for status reporting only.
	h256Hash m_knownNewHashes; 					///< New hashes we know about use for logging only

private:
	static char const* const s_stateNames[static_cast<int>(SyncState::Size)];
	bool invariants() const override = 0;
	void logNewBlock(h256 const& _h);

	EthereumHost& m_host;
};


/**
 * @brief Syncrhonization over PV60. Selects a single peer and tries to downloading hashes from it. After hash download is complete
 * syncs to peers and keeps up to date
 */

/**
 * Transitions:
 *
 * Idle->Hashes
 * 		Triggered when:
 * 			* A new peer appears that we can sync to
 * 			* Transtition to Idle, there are peers we can sync to
 * 		Effects:
 * 			* Set chain sync  (m_syncingTotalDifficulty, m_syncingLatestHash, m_syncer)
 * 			* Requests hashes from m_syncer
 *
 *  Hashes->Idle
 * 		Triggered when:
 * 			* Received too many hashes
 * 			* Received 0 total hashes from m_syncer
 * 			* m_syncer aborts
 * 		Effects:
 * 			In case of too many hashes sync is reset
 *
 *  Hashes->Blocks
 * 		Triggered when:
 * 			* Received known hash from m_syncer
 * 			* Received 0 hashes from m_syncer and m_syncingTotalBlocks not empty
 * 		Effects:
 * 			* Set up download manager, clear m_syncingTotalBlocks. Set all peers to help with downloading if they can
 *
 *  Blocks->Idle
 * 		Triggered when:
 * 			* m_syncer aborts
 * 			* m_syncer does not have required block
 * 			* All blocks downloaded
 * 			* Block qeueue is full with unknown blocks
 * 		Effects:
 * 			* Download manager is reset
 *
 *  Blocks->Waiting
 * 		Triggered when:
 * 			* Block queue is full with known blocks
 * 		Effects:
 * 			* Stop requesting blocks from peers
 *
 *  Waiting->Blocks
 * 		Triggered when:
 * 			* Block queue has space for new blocks
 * 		Effects:
 * 			* Continue requesting blocks from peers
 *
 *  Idle->NewBlocks
 * 		Triggered when:
 * 			* New block hashes arrive
 * 		Effects:
 * 			* Set up download manager, clear m_syncingTotalBlocks. Download blocks from a single peer. If downloaded blocks have unknown parents, set the peer to sync
 *
 *  NewBlocks->Idle
 * 		Triggered when:
 * 			* m_syncer aborts
 * 			* m_syncer does not have required block
 * 			* All new blocks downloaded
 * 			* Block qeueue is full with unknown blocks
 * 		Effects:
 * 			* Download manager is reset
 *
 */
class PV60Sync: public BlockChainSync
{
public:
	PV60Sync(EthereumHost& _host);

	/// @returns true is Sync is in progress
	bool isSyncing() const override { return !!m_syncer.lock(); }

	/// Called by peer once it has new hashes
	void onPeerNewHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes) override;

	/// Called by peer once it has another sequential block of hashes during sync
	void onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes) override;

	/// Called by peer when it is disconnecting
	void onPeerAborting() override;

	/// @returns Sync status
	SyncStatus status() const override;

protected:
	void onNewPeer(std::shared_ptr<EthereumPeer> _peer) override;
	void continueSync() override;
	void peerDoneBlocks(std::shared_ptr<EthereumPeer> _peer) override;
	void restartSync() override;
	void completeSync() override;
	void pauseSync() override;
	void resetSyncFor(std::shared_ptr<EthereumPeer> _peer, h256 const& _latestHash, u256 const& _td) override;

protected:
	/// Transition sync state in a particular direction. @param _peer Peer that is responsible for state tranfer
	void transition(std::shared_ptr<EthereumPeer> _peer, SyncState _s, bool _force = false, bool _needHelp = true);

	/// Reset peer syncing requirements state.
	void resetNeedsSyncing(std::shared_ptr<EthereumPeer> _peer) { setNeedsSyncing(_peer, h256(), 0); }

	/// Update peer syncing requirements state.
	void setNeedsSyncing(std::shared_ptr<EthereumPeer> _peer, h256 const& _latestHash, u256 const& _td);

	/// Do we presently need syncing with this peer?
	bool needsSyncing(std::shared_ptr<EthereumPeer> _peer) const;

	/// Check whether the session should bother grabbing blocks from a peer.
	bool shouldGrabBlocks(std::shared_ptr<EthereumPeer> _peer) const;

	/// Attempt to begin syncing with the peer; first check the peer has a more difficlult chain to download, then start asking for hashes, then move to blocks
	void attemptSync(std::shared_ptr<EthereumPeer> _peer);

	/// Update our syncing state
	void setState(std::shared_ptr<EthereumPeer> _peer, SyncState _s, bool _isSyncing = false, bool _needHelp = false);

	/// Check if peer is main syncer
	bool isSyncing(std::shared_ptr<EthereumPeer> _peer) const;

	/// Check if we need (re-)syncing with the peer.
	void noteNeedsSyncing(std::shared_ptr<EthereumPeer> _who);

	/// Set main syncing peer
	void changeSyncer(std::shared_ptr<EthereumPeer> _syncer, bool _needHelp);

	/// Called when peer done downloading blocks
	void noteDoneBlocks(std::shared_ptr<EthereumPeer> _who, bool _clemency);

	/// Start chainhash sync
	virtual void syncHashes(std::shared_ptr<EthereumPeer> _peer);

	/// Request subchain, no-op for pv60
	virtual void requestSubchain(std::shared_ptr<EthereumPeer> /*_peer*/) {}

	/// Abort syncing
	void abortSync();

	/// Reset hash chain syncing
	void resetSync();

	bool invariants() const override;

	h256s m_syncingNeededBlocks;				///< The blocks that we should download from this peer.
	h256 m_syncingLastReceivedHash;				///< Hash most recently received from peer.
	h256 m_syncingLatestHash;					///< Latest block's hash of the peer we are syncing to, as of the current sync.
	u256 m_syncingTotalDifficulty;				///< Latest block's total difficulty of the peer we aresyncing to, as of the current sync.
	std::weak_ptr<EthereumPeer> m_syncer;		///< Peer we are currently syncing with
};

/**
 * @brief Syncrhonization over PV61. Selects a single peer and requests every c_hashSubchainSize hash, splitting the hashchain into subchains and downloading each subchain in parallel.
 * Syncs to peers and keeps up to date
 */
class PV61Sync: public PV60Sync
{
public:
	PV61Sync(EthereumHost& _host);

protected:
	void restartSync() override;
	void completeSync() override;
	void requestSubchain(std::shared_ptr<EthereumPeer> _peer) override;
	void syncHashes(std::shared_ptr<EthereumPeer> _peer) override;
	void onPeerHashes(std::shared_ptr<EthereumPeer> _peer, h256s const& _hashes) override;
	void onPeerAborting() override;
	SyncStatus status() const override;
	bool invariants() const override;

private:
	/// Called when subchain is complete. Check if if hashchain is fully downloaded and proceed to downloading blocks
	void completeSubchain(std::shared_ptr<EthereumPeer> _peer, unsigned _n);
	/// Find a subchain for peers to downloading
	void requestSubchains();
	/// Check if downloading hashes in parallel
	bool isPV61Syncing() const;

	struct SubChain
	{
		h256s hashes;	///< List of subchain hashes
		h256 lastHash;	///< Last requested subchain hash
	};

	std::map<unsigned, SubChain> m_completeChainMap;		///< Fully downloaded subchains
	std::map<unsigned, SubChain> m_readyChainMap;			///< Subchains ready for download
	std::map<unsigned, SubChain> m_downloadingChainMap;		///< Subchains currently being downloading. In sync with m_chainSyncPeers
	std::map<std::weak_ptr<EthereumPeer>, unsigned, std::owner_less<std::weak_ptr<EthereumPeer>>> m_chainSyncPeers; ///< Peers to m_downloadingSubchain number map
	h256Hash m_knownHashes;									///< Subchain start markers. Used to track suchain completion
	unsigned m_syncingBlockNumber = 0;						///< Current subchain marker
	bool m_hashScanComplete = false;						///< True if leading peer completed hashchain scan and we have a list of subchains ready
};

std::ostream& operator<<(std::ostream& _out, SyncStatus const& _sync);

}
}
