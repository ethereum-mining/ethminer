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
/** @file EthereumSession.h
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
#include "CommonNet.h"

namespace eth
{

/**
 * @brief The EthereumSession class
 * @todo Document fully.
 */
class EthereumSession: public std::enable_shared_from_this<EthereumSession>
{
	friend class EthereumHost;

public:
	EthereumSession(EthereumHost* _server, bi::tcp::socket _socket, u256 _rNId, bi::address _peerAddress, unsigned short _peerPort = 0);
	~EthereumSession();

	void start();
	void disconnect(int _reason);

	void ping();

	bool isOpen() const { return m_socket.is_open(); }

	bi::tcp::endpoint endpoint() const;	///< for other peers to connect to.

private:
	void startInitialSync();
	void getPeers();

	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void ensureGettingChain();
	/// Ensure that we are waiting for a bunch of blocks from our peer.
	void continueGettingChain();
	/// Now getting a different chain so we need to make sure we restart.
	void restartGettingChain();

	void giveUpOnFetch();

	void dropped();
	void doRead();
	void doWrite(std::size_t length);
	bool interpret(RLP const& _r);

	/// @returns true iff the _msg forms a valid message for sending or receiving on the network.
	static bool checkPacket(bytesConstRef _msg);

	static RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void sendDestroy(bytes& _msg);
	void send(bytesConstRef _msg);
	void writeImpl(bytes& _buffer);
	void write();
	EthereumHost* m_server;

	std::recursive_mutex m_writeLock;
	std::deque<bytes> m_writeQueue;

	bi::tcp::socket m_socket;
	std::array<byte, 65536> m_data;
	PeerInfo m_info;
	Public m_id;

	bytes m_incoming;
	uint m_protocolVersion;
	u256 m_networkId;
	u256 m_reqNetworkId;
	unsigned short m_listenPort;			///< Port that the remote client is listening on for connections. Useful for giving to peers.
	uint m_caps;

	h256 m_latestHash;						///< Peer's latest block's hash.
	u256 m_totalDifficulty;					///< Peer's latest block's total difficulty.
	h256s m_neededBlocks;					///< The blocks that we should download from this peer.
	h256Set m_failedBlocks;					///< Blocks that the peer doesn't seem to have.

	h256Set m_askedBlocks;					///< The blocks for which we sent the last GetBlocks for but haven't received a corresponding Blocks.
	bool m_askedBlocksChanged = true;

	std::chrono::steady_clock::time_point m_ping;
	std::chrono::steady_clock::time_point m_connect;
	std::chrono::steady_clock::time_point m_disconnect;

	uint m_rating;
	bool m_requireTransactions;

	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;

	bool m_willBeDeleted = false;			///< True if we already posted a deleter on the strand.
};

}
