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
/** @file NodeTable.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#pragma once

#include <boost/integer/static_log2.hpp>
#include <libdevcrypto/Common.h>
#include <libp2p/UDP.h>

namespace dev
{
namespace p2p
{

/**
 * Ping packet: Check if node is alive.
 * PingNode is cached and regenerated after expiration - t, where t is timeout.
 *
 * signature: Signature of message.
 * ipAddress: Our IP address.
 * port: Our port.
 * expiration: Triggers regeneration of packet. May also provide control over synchronization.
 *
 * Ping is used to implement evict. When a new node is seen for
 * a given bucket which is full, the least-responsive node is pinged.
 * If the pinged node doesn't respond then it is removed and the new
 * node is inserted.
 *
 * @todo uint128_t for ip address (<->integer ipv4/6, asio-address, asio-endpoint)
 *
 */
struct PingNode: RLPXDatagram
{
	using RLPXDatagram::RLPXDatagram;
	PingNode(bi::udp::endpoint _to, std::string _src, uint16_t _srcPort, std::chrono::seconds _expiration = std::chrono::seconds(60)): RLPXDatagram(_to), ipAddress(_src), port(_srcPort), expiration(fromNow(_expiration)) {}
	
	std::string ipAddress;
	uint16_t port;
	uint64_t expiration;

	void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << expiration; }
};

struct Pong: RLPXDatagram
{
	using RLPXDatagram::RLPXDatagram;
	
	h256 replyTo;	/// TBD
	
	void streamRLP(RLPStream& _s) const { _s.appendList(1); _s << replyTo; }
};

/**
 * FindNode Packet: Request k-nodes, closest to the target.
 * FindNode is cached and regenerated after expiration - t, where t is timeout.
 * FindNode implicitly results in finding neighbors of a given node.
 *
 * target: Address of NodeId. The responding node will send back nodes closest to the target.
 * expiration: Triggers regeneration of packet. May also provide control over synchronization.
 *
 */
struct FindNode: RLPXDatagram
{
	using RLPXDatagram::RLPXDatagram;
	FindNode(bi::udp::endpoint _to, Address _target, std::chrono::seconds _expiration = std::chrono::seconds(30)): RLPXDatagram(_to), target(_target), expiration(fromNow(_expiration)) {}
	
	h160 target;
	uint64_t expiration;

	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s << target << expiration; }
};

/**
 * Node Packet: Multiple node packets are sent in response to FindNode.
 */
struct Neighbors: RLPXDatagram
{
	using RLPXDatagram::RLPXDatagram;
	
	struct Node
	{
		bytes ipAddress;
		uint16_t port;
		NodeId node;
		void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << node; }
	};
	
	std::set<Node> nodes;
	h256 nonce;
	
	Signature signature;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s.appendList(nodes.size()); for (auto& n: nodes) n.streamRLP(_s); _s << nonce; }
};

/**
 * NodeTable using S/Kademlia system for node discovery and preference.
 * untouched buckets are refreshed if they have not been touched within an hour
 *
 * Thread-safety is ensured by modifying NodeEntry details via 
 * shared_ptr replacement instead of mutating values.
 *
 * [Interface]
 * @todo constructor support for m_node, m_secret
 * @todo don't try to evict node if node isRequired. (support for makeRequired)
 * @todo exclude bucket from refresh if we have node as peer (support for makeRequired)
 * @todo restore nodes
 * @todo std::shared_ptr<PingNode> m_cachedPingPacket;
 * @todo std::shared_ptr<FindNeighbors> m_cachedFindSelfPacket;
 *
 * [Networking]
 * @todo use eth/stun/ice/whatever for public-discovery
 *
 * [Protocol]
 * @todo optimize knowledge at opposite edges; eg, s_bitsPerStep lookups. (Can be done via pointers to NodeBucket)
 * @todo ^ s_bitsPerStep = 5; // Denoted by b in [Kademlia]. Bits by which address space is divided.
 * @todo optimize (use tree for state and/or custom compare for cache)
 * @todo reputation (aka universal siblings lists)
 * @todo dht (aka siblings)
 *
 * [Maintenance]
 * @todo pretty logs
 */
class NodeTable: UDPSocketEvents, public std::enable_shared_from_this<NodeTable>
{
	using NodeSocket = UDPSocket<NodeTable, 1280>;
	using TimePoint = std::chrono::steady_clock::time_point;
	using EvictionTimeout = std::pair<std::pair<Address,TimePoint>,Address>;

	struct NodeDefaultEndpoint
	{
		NodeDefaultEndpoint(bi::udp::endpoint _udp): udp(_udp) {}
		bi::udp::endpoint udp;
	};
	
	struct Node
	{
		Node(Address _id, Public _pubk, NodeDefaultEndpoint _udp): id(_id), pubk(_pubk), endpoint(_udp) {}
		Node(Address _id, Public _pubk, bi::udp::endpoint _udp): Node(_id, _pubk, NodeDefaultEndpoint(_udp)) {}
		
		virtual Address const& address() const { return id; }
		virtual Public const& publicKey() const { return pubk; }
		
		Address id;
		Public pubk;
		NodeDefaultEndpoint endpoint;
	};
	
	/**
	 * NodeEntry
	 * @todo Type of id will become template parameter.
	 */
	struct NodeEntry: public Node
	{
		NodeEntry(Node _src, Address _id, Public _pubk, NodeDefaultEndpoint _gw): Node(_id, _pubk, _gw), distance(dist(_src.id,_id)) {}
		NodeEntry(Node _src, Address _id, Public _pubk, bi::udp::endpoint _udp): Node(_id, _pubk, NodeDefaultEndpoint(_udp)), distance(dist(_src.id,_id)) {}

		const unsigned distance;	///< Node's distance from _src (see constructor).
	};
	
	struct NodeBucket
	{
		unsigned distance;
		TimePoint modified;
		std::list<std::weak_ptr<NodeEntry>> nodes;
	};
	
public:
	
	/// Constants for Kademlia, mostly derived from address space.
	
	static constexpr unsigned s_addressByteSize = sizeof(NodeEntry::id);		///< Size of address type in bytes.
	static constexpr unsigned s_bits = 8 * s_addressByteSize;					///< Denoted by n in [Kademlia].
	static constexpr unsigned s_bins = s_bits - 1;								///< Size of m_state (excludes root, which is us).
	static constexpr unsigned s_maxSteps = boost::static_log2<s_bits>::value;	///< Max iterations of discovery. (doFindNode)
	
	/// Chosen constants
	
	static constexpr unsigned s_bucketSize = 16;		///< Denoted by k in [Kademlia]. Number of nodes stored in each bucket.
	static constexpr unsigned s_alpha = 3;				///< Denoted by \alpha in [Kademlia]. Number of concurrent FindNode requests.
	static constexpr uint16_t s_defaultPort = 30300;	///< Default port to listen on.
	
	/// Intervals
	
	static constexpr unsigned s_evictionCheckInterval = 75;							///< Interval at which eviction timeouts are checked.
	std::chrono::milliseconds const c_reqTimeout = std::chrono::milliseconds(300);		///< How long to wait for requests (evict, find iterations).
	std::chrono::seconds const c_bucketRefresh = std::chrono::seconds(3600);			///< Refresh interval prevents bucket from becoming stale. [Kademlia]
	
	static unsigned dist(Address const& _a, Address const& _b) { u160 d = _a ^ _b; unsigned ret; for (ret = 0; d >>= 1; ++ret) {}; return ret; }

	NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _port = s_defaultPort);
	~NodeTable();
	
	void join();
	
	std::list<Address> nodes() const;
	
	NodeEntry operator[](Address _id);
	
protected:
	/// Repeatedly sends s_alpha concurrent requests to nodes nearest to target, for nodes nearest to target, up to .
	void doFindNode(Address _node, unsigned _round = 0, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried = std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>>());

	/// Returns nodes nearest to target.
	std::vector<std::shared_ptr<NodeEntry>> findNearest(Address _target);
	
	void ping(bi::udp::endpoint _to) const;
	
	void ping(NodeEntry* _n) const;
	
	void evict(std::shared_ptr<NodeEntry> _leastSeen, std::shared_ptr<NodeEntry> _new);
	
	void noteNode(Public _pubk, bi::udp::endpoint _endpoint);
	
	void noteNode(std::shared_ptr<NodeEntry> _n);
	
	void dropNode(std::shared_ptr<NodeEntry> _n);
	
	NodeBucket const& bucket(NodeEntry* _n) const;
	
	/// Network Events
	
	void onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet);
	
	void onDisconnected(UDPSocketFace*) {};
	
	/// Tasks
	
	void doCheckEvictions(boost::system::error_code const& _ec);
	
	void doRefreshBuckets(boost::system::error_code const& _ec);

#ifndef BOOST_AUTO_TEST_SUITE
private:
#else
protected:
#endif
	/// Sends s_alpha concurrent FindNeighbor requests to nodes closest to target until
	void requestNeighbors(NodeEntry const& _node, Address _target) const;

	Node m_node;												///< This node.
	Secret m_secret;											///< This nodes secret key.

	mutable Mutex x_nodes;									///< Mutable for thread-safe copy in nodes() const.
	std::map<Address, std::shared_ptr<NodeEntry>> m_nodes;		///< Address -> Node table (most common lookup path)

	Mutex x_state;
	std::array<NodeBucket, s_bins> m_state;					///< State table of binned nodes.

	Mutex x_evictions;
	std::deque<EvictionTimeout> m_evictions;					///< Eviction timeouts.
	
	std::shared_ptr<NodeSocket> m_socket;						///< Shared pointer for our UDPSocket; ASIO requires shared_ptr.
	NodeSocket* m_socketPtr;									///< Set to m_socket.get().
	ba::io_service& m_io;										///< Used by bucket refresh timer.
	boost::asio::deadline_timer m_bucketRefreshTimer;			///< Timer which schedules and enacts bucket refresh.
	boost::asio::deadline_timer m_evictionCheckTimer;			///< Timer for handling node evictions.
};
	
struct NodeTableWarn: public LogChannel { static const char* name() { return "!P!"; } static const int verbosity = 0; };
struct NodeTableNote: public LogChannel { static const char* name() { return "*P*"; } static const int verbosity = 1; };

}
}