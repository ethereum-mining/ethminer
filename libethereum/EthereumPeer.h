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
	void ensureGettingChain();
	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void continueGettingChain();

	void giveUpOnFetch();

	void clearKnownTransactions() { std::lock_guard<std::mutex> l(x_knownTransactions); m_knownTransactions.clear(); }
	void setAsking(Asking _g, bool _helping = false);
	void setHelping(bool _helping = false) { setAsking(m_asking, _helping); }
	
	unsigned m_protocolVersion;
	u256 m_networkId;

	Asking m_asking;
	Syncing m_syncing;

	h256 m_latestHash;						///< Peer's latest block's hash.
	u256 m_totalDifficulty;					///< Peer's latest block's total difficulty.
	h256s m_neededBlocks;					///< The blocks that we should download from this peer.

	bool m_requireTransactions;

	Mutex x_knownBlocks;
	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
	std::mutex x_knownTransactions;

	DownloadSub m_sub;
};

}
}
