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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * Miscellanea required for the Host/Session/NodeTable classes.
 */

#pragma once

#include <string>
#include <set>
#include <vector>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <chrono>
#include <libdevcrypto/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{

class RLP;
class RLPStream;

namespace p2p
{

using NodeId = h512;

bool isPrivateAddress(bi::address const& _addressToCheck);
bool isLocalHostAddress(bi::address const& _addressToCheck);

class UPnP;
class Capability;
class Host;
class Session;

struct NetworkStartRequired: virtual dev::Exception {};

struct NetWarn: public LogChannel { static const char* name() { return "!N!"; } static const int verbosity = 0; };
struct NetNote: public LogChannel { static const char* name() { return "*N*"; } static const int verbosity = 1; };
struct NetMessageSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 2; };
struct NetConnect: public LogChannel { static const char* name() { return "+N+"; } static const int verbosity = 10; };
struct NetMessageDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 5; };
struct NetTriviaSummary: public LogChannel { static const char* name() { return "-N-"; } static const int verbosity = 10; };
struct NetTriviaDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 11; };
struct NetAllDetail: public LogChannel { static const char* name() { return "=N="; } static const int verbosity = 13; };
struct NetRight: public LogChannel { static const char* name() { return ">N>"; } static const int verbosity = 14; };
struct NetLeft: public LogChannel { static const char* name() { return "<N<"; } static const int verbosity = 15; };

enum PacketType
{
	HelloPacket = 0,
	DisconnectPacket,
	PingPacket,
	PongPacket,
	GetPeersPacket,
	PeersPacket,
	UserPacket = 0x10
};

enum DisconnectReason
{
	DisconnectRequested = 0,
	TCPError,
	BadProtocol,
	UselessPeer,
	TooManyPeers,
	DuplicatePeer,
	IncompatibleProtocol,
	NullIdentity,
	ClientQuit,
	UnexpectedIdentity,
	LocalIdentity,
	PingTimeout,
	UserReason = 0x10,
	NoDisconnect = 0xffff
};

inline bool isPermanentProblem(DisconnectReason _r)
{
	switch (_r)
	{
	case DuplicatePeer:
	case IncompatibleProtocol:
	case NullIdentity:
	case UnexpectedIdentity:
	case LocalIdentity:
		return true;
	default:
		return false;
	}
}

/// @returns the string form of the given disconnection reason.
std::string reasonOf(DisconnectReason _r);

using CapDesc = std::pair<std::string, u256>;
using CapDescSet = std::set<CapDesc>;
using CapDescs = std::vector<CapDesc>;

/*
 * Used by Host to pass negotiated information about a connection to a
 * new Peer Session; PeerSessionInfo is then maintained by Session and can
 * be queried for point-in-time status information via Host.
 */
struct PeerSessionInfo
{
	NodeId id;
	std::string clientVersion;
	std::string host;
	unsigned short port;
	std::chrono::steady_clock::duration lastPing;
	std::set<CapDesc> caps;
	unsigned socket;
	std::map<std::string, std::string> notes;
};

using PeerSessionInfos = std::vector<PeerSessionInfo>;

/**
 * @brief IPv4,UDP/TCP endpoints.
 */
struct NodeIPEndpoint
{
	NodeIPEndpoint(): udp(bi::udp::endpoint()), tcp(bi::tcp::endpoint()) {}
	NodeIPEndpoint(bi::udp::endpoint _udp): udp(_udp) {}
	NodeIPEndpoint(bi::tcp::endpoint _tcp): tcp(_tcp) {}
	NodeIPEndpoint(bi::udp::endpoint _udp, bi::tcp::endpoint _tcp): udp(_udp), tcp(_tcp) {}

	bi::udp::endpoint udp;
	bi::tcp::endpoint tcp;
	
	operator bool() const { return !udp.address().is_unspecified() || !tcp.address().is_unspecified(); }
};

struct Node
{
	Node(): endpoint(NodeIPEndpoint()) {};
	Node(Public _pubk, NodeIPEndpoint _ip, bool _required = false): id(_pubk), endpoint(_ip), required(_required) {}
	Node(Public _pubk, bi::udp::endpoint _udp, bool _required = false): Node(_pubk, NodeIPEndpoint(_udp), _required) {}
	
	virtual NodeId const& address() const { return id; }
	virtual Public const& publicKey() const { return id; }
	
	NodeId id;
	
	/// Endpoints by which we expect to reach node.
	NodeIPEndpoint endpoint;
	
	/// If true, node will not be removed from Node list.
	// TODO: p2p implement
	bool required = false;
	
	virtual operator bool() const { return (bool)id; }
};

}
}
