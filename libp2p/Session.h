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
/** @file Session.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <array>
#include <deque>
#include <set>
#include <memory>
#include <utility>
#include <libdevcore/RLP.h>
#include "Common.h"

namespace dev
{

namespace p2p
{

/**
 * @brief The Session class
 * @todo Document fully.
 */
class Session: public std::enable_shared_from_this<Session>
{
	friend class Host;
	friend class HostCapabilityFace;

public:
	Session(Host* _server, bi::tcp::socket _socket, bi::address _peerAddress, unsigned short _peerPort = 0);
	virtual ~Session();

	void start();
	void disconnect(int _reason);

	void ping();

	bool isOpen() const { return m_socket.is_open(); }

	h512 id() const { return m_id; }
	unsigned socketId() const { return m_socket.native_handle(); }

	bi::tcp::endpoint endpoint() const;	///< for other peers to connect to.

	template <class PeerCap>
	std::shared_ptr<PeerCap> cap() const { try { return std::static_pointer_cast<PeerCap>(m_capabilities.at(PeerCap::name())); } catch (...) { return nullptr; } }

	static RLPStream& prep(RLPStream& _s);
	void sealAndSend(RLPStream& _s);
	void sendDestroy(bytes& _msg);
	void send(bytesConstRef _msg);

	void addRating(unsigned _r) { m_rating += _r; }

	void addNote(std::string const& _k, std::string const& _v) { m_info.notes[_k] = _v; }

private:
	void dropped();
	void doRead();
	void doWrite(std::size_t length);
	void writeImpl(bytes& _buffer);
	void write();

	void getPeers();
	bool interpret(RLP const& _r);

	/// @returns true iff the _msg forms a valid message for sending or receiving on the network.
	static bool checkPacket(bytesConstRef _msg);

	Host* m_server;

	std::mutex m_writeLock;
	std::deque<bytes> m_writeQueue;

	mutable bi::tcp::socket m_socket;	///< Mutable to ask for native_handle().
	std::array<byte, 65536> m_data;
	PeerInfo m_info;
	h512 m_id;

	bytes m_incoming;
	unsigned m_protocolVersion;
	unsigned short m_listenPort;			///< Port that the remote client is listening on for connections. Useful for giving to peers.

	std::chrono::steady_clock::time_point m_ping;
	std::chrono::steady_clock::time_point m_connect;
	std::chrono::steady_clock::time_point m_disconnect;

	unsigned m_rating;

	std::map<std::string, std::shared_ptr<Capability>> m_capabilities;

	bool m_willBeDeleted = false;			///< True if we already posted a deleter on the strand.
};

}
}
