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
/** @file EthereumHost.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <memory>
#include <utility>
#include <thread>

#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libdevcore/RangeMask.h>
#include <libethcore/Common.h>
#include <libp2p/Common.h>
#include "CommonNet.h"
#include "EthereumPeer.h" //TODO: forward decl
#include "DownloadMan.h"


namespace dev
{

class RLPStream;

namespace eth
{

class EthereumHost;
class BlockQueue;

/**
 * @brief BlockChain synchronization strategy class
 * @doWork Syncs to peers and sends new blocks and transactions.
 */
class BlockChainSync: public HasInvariants
{
public:
	BlockChainSync(EthereumHost& _host);

	/// Will block on network process events.
	virtual ~BlockChainSync();
	void abortSync();

	DownloadMan const& downloadMan() const;
	DownloadMan& downloadMan();
	virtual bool isSyncing() const = 0;
	virtual void onPeerStatus(EthereumPeer* _peer); ///< Called by peer to report status
	virtual void onPeerBlocks(EthereumPeer* _peer, RLP const& _r) = 0; ///< Called by peer once it has new blocks during syn
	virtual void onPeerNewBlock(EthereumPeer* _peer, RLP const& _r) = 0; ///< Called by peer once it has new blocks
	virtual void onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes) = 0; ///< Called by peer once it has new hashes
	virtual void onPeerHashes(EthereumPeer* _peer, h256s const& _hashes) = 0; ///< Called by peer once it has another sequential block of hashes during sync
	virtual void onPeerAborting(EthereumPeer* _peer) = 0; ///< Called by peer when it is disconnecting
	virtual SyncStatus status() const = 0;

	static char const* stateName(SyncState _s) { return s_stateNames[static_cast<int>(_s)]; }

private:
	static char const* const s_stateNames[static_cast<int>(SyncState::Size)];

	void setState(SyncState _s);

	bool invariants() const override = 0;

	EthereumHost& m_host;
	Handler m_bqRoomAvailable;
	HashDownloadMan m_hashMan;

protected:

	EthereumHost& host() { return m_host; }
	EthereumHost const& host() const { return m_host; }
	unsigned estimateHashes();

	mutable RecursiveMutex x_sync;
	SyncState m_state = SyncState::Idle;			///< Current sync state
	SyncState m_lastActiveState = SyncState::Idle; 	///< Saved state before entering waiting queue mode
	unsigned m_estimatedHashes = 0;					///< Number of estimated hashes for the last peer over PV60. Used for status reporting only.
};

class PV60Sync: public BlockChainSync
{
public:

	PV60Sync(EthereumHost& _host);

	bool isSyncing() const override { return !!m_syncer; }
	void onPeerStatus(EthereumPeer* _peer) override; ///< Called by peer to report status
	void onPeerBlocks(EthereumPeer* _peer, RLP const& _r) override; ///< Called by peer once it has new blocks during syn
	void onPeerNewBlock(EthereumPeer* _peer, RLP const& _r) override; ///< Called by peer once it has new blocks
	void onPeerNewHashes(EthereumPeer* _peer, h256s const& _hashes) override; ///< Called by peer once it has new hashes
	void onPeerHashes(EthereumPeer* _peer, h256s const& _hashes) override; ///< Called by peer once it has another sequential block of hashes during sync
	void onPeerAborting(EthereumPeer* _peer) override; ///< Called by peer when it is disconnecting
	SyncStatus status() const override;

	void transition(EthereumPeer* _peer, SyncState _s, bool _force = false, bool _needHelp = true);
	void resetNeedsSyncing(EthereumPeer* _peer) { setNeedsSyncing(_peer, h256(), 0); }
	bool needsSyncing(EthereumPeer* _peer) const { return !!_peer->m_latestHash; }

	void setNeedsSyncing(EthereumPeer* _peer, h256 _latestHash, u256 _td);
	bool shouldGrabBlocks(EthereumPeer* _peer) const;
	void attemptSync(EthereumPeer* _peer);
	void setState(EthereumPeer* _peer, SyncState _s, bool _isSyncing = false, bool _needHelp = false);
	bool isSyncing(EthereumPeer* _peer) const;
	void noteNeedsSyncing(EthereumPeer* _who);
	void changeSyncer(EthereumPeer* _syncer, bool _needHelp);
	void noteDoneBlocks(EthereumPeer* _who, bool _clemency);
	void abortSync(EthereumPeer* _peer);
	void requestBlocks(EthereumPeer* _peer);


private:
	bool invariants() const override;

	h256s m_knownHashes;						///< List of block hashes we need to download.

	h256s m_syncingNeededBlocks;				///< The blocks that we should download from this peer.
	h256 m_syncingLastReceivedHash;				///< Hash most recently received from peer.
	h256 m_syncingLatestHash;					///< Peer's latest block's hash, as of the current sync.
	u256 m_syncingTotalDifficulty;				///< Peer's latest block's total difficulty, as of the current sync.
	EthereumPeer* m_syncer = nullptr;	// TODO: switch to weak_ptr

};
}
}
