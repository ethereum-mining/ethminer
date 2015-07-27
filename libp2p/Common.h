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

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <chrono>
#include <libdevcrypto/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{

class RLP;
class RLPStream;

namespace p2p
{

/// Peer network protocol version.
extern const unsigned c_protocolVersion;
extern const unsigned c_defaultIPPort;

struct NodeIPEndpoint;
struct Node;
extern const NodeIPEndpoint UnspecifiedNodeIPEndpoint;
extern const Node UnspecifiedNode;

using NodeId = h512;

bool isPrivateAddress(bi::address const& _addressToCheck);
bool isPrivateAddress(std::string const& _addressToCheck);
bool isLocalHostAddress(bi::address const& _addressToCheck);
bool isLocalHostAddress(std::string const& _addressToCheck);
bool isPublicAddress(bi::address const& _addressToCheck);
bool isPublicAddress(std::string const& _addressToCheck);

class UPnP;
class Capability;
class Host;
class Session;

struct NetworkStartRequired: virtual dev::Exception {};
struct InvalidPublicIPAddress: virtual dev::Exception {};
struct InvalidHostIPAddress: virtual dev::Exception {};

struct NetWarn: public LogChannel { static const char* name(); static const int verbosity = 0; };
struct NetNote: public LogChannel { static const char* name(); static const int verbosity = 1; };
struct NetImpolite: public LogChannel { static const char* name(); static const int verbosity = 2; };
struct NetMessageSummary: public LogChannel { static const char* name(); static const int verbosity = 3; };
struct NetConnect: public LogChannel { static const char* name(); static const int verbosity = 10; };
struct NetMessageDetail: public LogChannel { static const char* name(); static const int verbosity = 5; };
struct NetTriviaSummary: public LogChannel { static const char* name(); static const int verbosity = 10; };
struct NetTriviaDetail: public LogChannel { static const char* name(); static const int verbosity = 11; };
struct NetAllDetail: public LogChannel { static const char* name(); static const int verbosity = 13; };
struct NetRight: public LogChannel { static const char* name(); static const int verbosity = 14; };
struct NetLeft: public LogChannel { static const char* name(); static const int verbosity = 15; };
struct NetP2PWarn: public LogChannel { static const char* name(); static const int verbosity = 2; };
struct NetP2PNote: public LogChannel { static const char* name(); static const int verbosity = 6; };
struct NetP2PConnect: public LogChannel { static const char* name(); static const int verbosity = 10; };

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
	NodeId const id;
	std::string const clientVersion;
	std::string const host;
	unsigned short const port;
	std::chrono::steady_clock::duration lastPing;
	std::set<CapDesc> const caps;
	unsigned socketId;
	std::map<std::string, std::string> notes;
	unsigned const protocolVersion;
};

using PeerSessionInfos = std::vector<PeerSessionInfo>;

/**
 * @brief IPv4,UDP/TCP endpoints.
 */
struct NodeIPEndpoint
{
	enum RLPAppend
	{
		StreamList,
		StreamInline
	};
	
	/// Setting true causes isAllowed to return true for all addresses. (Used by test fixtures)
	static bool test_allowLocal;

	NodeIPEndpoint() = default;
	NodeIPEndpoint(bi::address _addr, uint16_t _udp, uint16_t _tcp): address(_addr), udpPort(_udp), tcpPort(_tcp) {}
	NodeIPEndpoint(RLP const& _r) { interpretRLP(_r); }

	bi::address address = bi::address();
	uint16_t udpPort = 0;
	uint16_t tcpPort = 0;
	
	operator bi::udp::endpoint() const { return bi::udp::endpoint(address, udpPort); }
	operator bi::tcp::endpoint() const { return bi::tcp::endpoint(address, tcpPort); }
	
	operator bool() const { return !address.is_unspecified() && udpPort > 0 && tcpPort > 0; }
	
	bool isAllowed() const { return NodeIPEndpoint::test_allowLocal ? !address.is_unspecified() : isPublicAddress(address); }
	
	void streamRLP(RLPStream& _s, RLPAppend _append = StreamList) const;
	void interpretRLP(RLP const& _r);
};
	
struct Node
{
	Node(Public _pubk, NodeIPEndpoint const& _ip, bool _required = false): id(_pubk), endpoint(_ip), required(_required) {}

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

class DeadlineOps
{
	class DeadlineOp
	{
	public:
		DeadlineOp(ba::io_service& _io, unsigned _msInFuture, std::function<void(boost::system::error_code const&)> const& _f): m_timer(new ba::deadline_timer(_io)) { m_timer->expires_from_now(boost::posix_time::milliseconds(_msInFuture)); m_timer->async_wait(_f); }
		~DeadlineOp() {}
		
		DeadlineOp(DeadlineOp&& _s): m_timer(_s.m_timer.release()) {}
		DeadlineOp& operator=(DeadlineOp&& _s) { m_timer.reset(_s.m_timer.release()); return *this; }
		
		bool expired() { Guard l(x_timer); return m_timer->expires_from_now().total_nanoseconds() <= 0; }
		void wait() { Guard l(x_timer); m_timer->wait(); }
		
	private:
		std::unique_ptr<ba::deadline_timer> m_timer;
		Mutex x_timer;
	};
	
public:
	DeadlineOps(ba::io_service& _io, unsigned _reapIntervalMs = 100): m_io(_io), m_reapIntervalMs(_reapIntervalMs), m_stopped(false) { reap(); }
	~DeadlineOps() { stop(); }

	void schedule(unsigned _msInFuture, std::function<void(boost::system::error_code const&)> const& _f) { if (m_stopped) return; DEV_GUARDED(x_timers) m_timers.emplace_back(m_io, _msInFuture, _f); }	

	void stop() { m_stopped = true; DEV_GUARDED(x_timers) m_timers.clear(); }
	
protected:
	void reap();
	
private:
	ba::io_service& m_io;
	unsigned m_reapIntervalMs;
	
	std::vector<DeadlineOp> m_timers;
	Mutex x_timers;
	
	std::atomic<bool> m_stopped;
};

}
	
/// Simple stream output for a NodeIPEndpoint.
std::ostream& operator<<(std::ostream& _out, dev::p2p::NodeIPEndpoint const& _ep);
}

/// std::hash for asio::adress
namespace std
{

template <> struct hash<bi::address>
{
	size_t operator()(bi::address const& _a) const
	{
		if (_a.is_v4())
			return std::hash<unsigned long>()(_a.to_v4().to_ulong());
		if (_a.is_v6())
		{
			auto const& range = _a.to_v6().to_bytes();
			return boost::hash_range(range.begin(), range.end());
		}
		if (_a.is_unspecified())
			return static_cast<size_t>(0x3487194039229152ull);  // Chosen by fair dice roll, guaranteed to be random
		return std::hash<std::string>()(_a.to_string());
	}
};

}
