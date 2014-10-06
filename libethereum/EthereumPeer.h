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
#include <set>
#include <memory>
#include <utility>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include <libdevcore/RangeMask.h>
#include <libethcore/CommonEth.h>
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
 */
class EthereumPeer: public p2p::Capability
{
	friend class EthereumHost;

public:
	EthereumPeer(p2p::Session* _s, p2p::HostCapabilityFace* _h);
	virtual ~EthereumPeer();

	static std::string name() { return "eth"; }

	EthereumHost* host() const;

private:
	virtual bool interpret(RLP const& _r);

	void sendStatus();
	void startInitialSync();

	void tryGrabbingHashChain();

	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void ensureAskingBlocks();
	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void continueSync();

	void finishSync();

	void clearKnownTransactions() { std::lock_guard<std::mutex> l(x_knownTransactions); m_knownTransactions.clear(); }
	void setAsking(Asking _g, bool _isSyncing);

	void setNeedsSyncing(h256 _latestHash, u256 _td) { m_latestHash = _latestHash; m_totalDifficulty = _td; }
	bool needsSyncing() const { return !!m_latestHash; }
	bool isSyncing() const { return m_isSyncing; }
	
	/// Peer's protocol version.
	unsigned m_protocolVersion;
	/// Peer's network id.
	u256 m_networkId;

	/// What, if anything, we last asked the other peer for.
	Asking m_asking;

	/// Whether this peer is in the process of syncing or not. Only one peer can be syncing at once.
	bool m_isSyncing = false;

	/// These are determined through either a Status message or from NewBlock.
	h256 m_latestHash;						///< Peer's latest block's hash.
	u256 m_totalDifficulty;					///< Peer's latest block's total difficulty.
	/// Once a sync is started on this peer, they are cleared.

	/// This is built as we ask for hashes. Once no more hashes are given, we present this to the
	/// host who initialises the DownloadMan and m_sub becomes active for us to begin asking for blocks.
	h256s m_neededBlocks;					///< The blocks that we should download from this peer.

	/// Once we're asking for blocks, this becomes in use.
	DownloadSub m_sub;

	/// Have we received a GetTransactions packet that we haven't yet answered?
	bool m_requireTransactions;

	Mutex x_knownBlocks;
	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
	std::mutex x_knownTransactions;

};

}
}
