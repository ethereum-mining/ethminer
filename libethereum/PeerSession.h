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
/** @file PeerSession.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <array>
#include <set>
#include <memory>
#include <utility>
#include <libethcore/RLP.h>
#include "CommonEth.h"
#include "PeerNetwork.h"

namespace eth
{

class PeerSession: public std::enable_shared_from_this<PeerSession>
{
	friend class PeerServer;

public:
	PeerSession(PeerServer* _server, bi::tcp::socket _socket, uint _rNId, bi::address _peerAddress, unsigned short _peerPort = 0);
	~PeerSession();

	void start();
	void disconnect(int _reason);

	void ping();

	bool isOpen() const { return m_socket.is_open(); }

	bi::tcp::endpoint endpoint() const;	///< for other peers to connect to.

private:
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
	PeerServer* m_server;

	bi::tcp::socket m_socket;
	std::array<byte, 65536> m_data;
	PeerInfo m_info;
	Public m_id;

	bytes m_incoming;
	uint m_protocolVersion;
	uint m_networkId;
	uint m_reqNetworkId;
	unsigned short m_listenPort;			///< Port that the remote client is listening on for connections. Useful for giving to peers.
	uint m_caps;

	std::chrono::steady_clock::time_point m_ping;
	std::chrono::steady_clock::time_point m_connect;
	std::chrono::steady_clock::time_point m_disconnect;

	uint m_rating;
	bool m_requireTransactions;

	std::set<h256> m_knownBlocks;
	std::set<h256> m_knownTransactions;
};

}
