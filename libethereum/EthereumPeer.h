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
#include <libethential/RLP.h>
#include <libethcore/CommonEth.h>
#include <libethnet/Common.h>
#include "CommonNet.h"

namespace eth
{

class HostCapabilityFace;

/**
 * @brief The EthereumPeer class
 * @todo Document fully.
 */
class EthereumPeer: public PeerCapability
{
	friend class EthereumHost;

public:
	EthereumPeer(PeerSession* _s, HostCapabilityFace* _h);
	virtual ~EthereumPeer();

	static std::string name() { return "eth"; }

	EthereumHost* host() const;

private:
	virtual bool interpret(RLP const& _r);

	void sendStatus();
	void startInitialSync();

	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void ensureGettingChain();
	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void continueGettingChain();
	/// Now getting a different chain so we need to make sure we restart.
	void restartGettingChain();

	void giveUpOnFetch();

	uint m_protocolVersion;
	u256 m_networkId;

	h256 m_latestHash;						///< Peer's latest block's hash.
	u256 m_totalDifficulty;					///< Peer's latest block's total difficulty.
	h256s m_neededBlocks;					///< The blocks that we should download from this peer.
	h256Set m_failedBlocks;					///< Blocks that the peer doesn't seem to have.

	h256Set m_askedBlocks;					///< The blocks for which we sent the last GetBlocks for but haven't received a corresponding Blocks.
	bool m_askedBlocksChanged = true;

	bool m_requireTransactions;

	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
};

}
