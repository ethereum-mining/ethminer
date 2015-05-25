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
/** @file EthereumPeer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <array>
#include <memory>
#include <utility>

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include <libdevcore/RangeMask.h>
#include <libethcore/Common.h>
#include <libp2p/Capability.h>
#include "CommonNet.h"
#include "DownloadMan.h"

namespace dev
{
namespace eth
{

/**
 * @brief The EthereumPeer class
 * @todo Document fully.
 * @todo make state transitions thread-safe.
 */
class EthereumPeer: public p2p::Capability
{
	friend class EthereumHost;

public:
	/// Basic constructor.
	EthereumPeer(p2p::Session* _s, p2p::HostCapabilityFace* _h, unsigned _i);

	/// Basic destructor.
	virtual ~EthereumPeer();

	/// What is our name?
	static std::string name() { return "eth"; }

	/// What is our version?
	static u256 version() { return c_protocolVersion; }

	/// How many message types do we have?
	static unsigned messageCount() { return PacketCount; }

	/// What is the ethereum subprotocol host object.
	EthereumHost* host() const;

	void setIdle();
	void requestState();
	void requestHashes();
	void requestHashes(h256 const& _lastHash);
	void requestBlocks();

private:
	using p2p::Capability::sealAndSend;

	/// Interpret an incoming message.
	virtual bool interpret(unsigned _id, RLP const& _r);

	/// Abort the sync operation.
	void abortSync();

	/// Clear all known transactions.
	void clearKnownTransactions() { std::lock_guard<std::mutex> l(x_knownTransactions); m_knownTransactions.clear(); }

	/// Update our asking state.
	void setAsking(Asking _g);

	/// Update our syncing requirements state.
	void setNeedsSyncing(h256 _latestHash, u256 _td);
	void resetNeedsSyncing() { setNeedsSyncing(h256(), 0); }

	/// Do we presently need syncing with this peer?
	bool needsSyncing() const { return !!m_latestHash; }

	/// Are we presently syncing with this peer?
	bool isSyncing() const;

	/// Check whether the session should bother grabbing the peer's blocks.
	bool shouldGrabBlocks() const;

	/// Runs period checks to check up on the peer.
	void tick();

	/// Peer's protocol version.
	unsigned m_protocolVersion;
	/// Peer's network id.
	u256 m_networkId;

	/// What, if anything, we last asked the other peer for.
	Asking m_asking = Asking::Nothing;
	/// When we asked for it. Allows a time out.
	std::chrono::system_clock::time_point m_lastAsk;

	/// Whether this peer is in the process of syncing or not. Only one peer can be syncing at once.
	bool m_isSyncing = false;

	/// These are determined through either a Status message or from NewBlock.
	h256 m_latestHash;						///< Peer's latest block's hash that we know about or default null value if no need to sync.
	u256 m_totalDifficulty;					///< Peer's latest block's total difficulty.
	h256 m_genesisHash;						///< Peer's genesis hash
	/// Once a sync is started on this peer, they are cleared and moved into m_syncing*.

	/// This is built as we ask for hashes. Once no more hashes are given, we present this to the
	/// host who initialises the DownloadMan and m_sub becomes active for us to begin asking for blocks.
	h256s m_syncingNeededBlocks;				///< The blocks that we should download from this peer.
	h256 m_syncingLastReceivedHash;				///< Hash most recently received from peer.
	h256 m_syncingLatestHash;					///< Peer's latest block's hash, as of the current sync.
	u256 m_syncingTotalDifficulty;				///< Peer's latest block's total difficulty, as of the current sync.
	unsigned m_expectedHashes = 0;				///< Estimated Upper bound of hashes to expect from this peer.
	unsigned m_syncHashNumber = 0;

	/// Once we're asking for blocks, this becomes in use.
	DownloadSub m_sub;
	HashDownloadSub m_hashSub;

	/// Have we received a GetTransactions packet that we haven't yet answered?
	bool m_requireTransactions = false;

	Mutex x_knownBlocks;
	h256Hash m_knownBlocks;					///< Blocks that the peer already knows about (that don't need to be sent to them).
	Mutex x_knownTransactions;
	h256Hash m_knownTransactions;			///< Transactions that the peer already knows of.

};

}
}
