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

#include <algorithm>
#include <boost/integer/static_log2.hpp>
#include <libdevcrypto/Common.h>
#include <libp2p/UDP.h>

namespace dev
{
namespace p2p
{

/**
 * NodeTable using S/Kademlia system for node discovery and preference.
 * untouched buckets are refreshed if they have not been touched within an hour
 *
 * Thread-safety is ensured by modifying NodeEntry details via 
 * shared_ptr replacement instead of mutating values.
 *
 * [Integration]
 * @todo deadline-timer which maintains tcp/peer connections
 * @todo restore nodes: affects refreshbuckets
 * @todo TCP endpoints
 * @todo makeRequired: don't try to evict node if node isRequired.
 * @todo makeRequired: exclude bucket from refresh if we have node as peer.
 *
 * [Optimization]
 * @todo encapsulate doFindNode into NetworkAlgorithm (task)
 * @todo Pong to include ip:port where ping was received
 * @todo expiration and sha3(id) 'to' for messages which are replies (prevents replay)
 * @todo std::shared_ptr<PingNode> m_cachedPingPacket;
 * @todo std::shared_ptr<FindNeighbours> m_cachedFindSelfPacket;
 * @todo store root node in table?
 *
 * [Networking]
 * @todo TCP endpoints
 * @todo eth/upnp/natpmp/stun/ice/etc for public-discovery
 * @todo firewall
 *
 * [Protocol]
 * @todo post-eviction pong
 * @todo optimize knowledge at opposite edges; eg, s_bitsPerStep lookups. (Can be done via pointers to NodeBucket)
 * @todo ^ s_bitsPerStep = 5; // Denoted by b in [Kademlia]. Bits by which address space is divided.
 * @todo optimize (use tree for state and/or custom compare for cache)
 * @todo reputation (aka universal siblings lists)
 */
class NodeTable: UDPSocketEvents, public std::enable_shared_from_this<NodeTable>
{
	friend struct Neighbours;
	using NodeSocket = UDPSocket<NodeTable, 1280>;
	using TimePoint = std::chrono::steady_clock::time_point;
	using EvictionTimeout = std::pair<std::pair<NodeId,TimePoint>,NodeId>;	///< First NodeId may be evicted and replaced with second NodeId.

	struct NodeDefaultEndpoint
	{
		NodeDefaultEndpoint(bi::udp::endpoint _udp): udp(_udp) {}
		bi::udp::endpoint udp;
	};
	
	struct Node
	{
		Node(Public _pubk, NodeDefaultEndpoint _udp): id(_pubk), endpoint(_udp) {}
		Node(Public _pubk, bi::udp::endpoint _udp): Node(_pubk, NodeDefaultEndpoint(_udp)) {}
		
		virtual NodeId const& address() const { return id; }
		virtual Public const& publicKey() const { return id; }
		
		NodeId id;
		NodeDefaultEndpoint endpoint;
	};
	
	/**
	 * NodeEntry
	 * @todo Type of id will become template parameter.
	 */
	struct NodeEntry: public Node
	{
		NodeEntry(Node _src, Public _pubk, NodeDefaultEndpoint _gw): Node(_pubk, _gw), distance(dist(_src.id,_pubk)) {}
		NodeEntry(Node _src, Public _pubk, bi::udp::endpoint _udp): Node(_pubk, NodeDefaultEndpoint(_udp)), distance(dist(_src.id,_pubk)) {}

		const unsigned distance;	///< Node's distance from _src (see constructor).
	};
	
	struct NodeBucket
	{
		unsigned distance;
		TimePoint modified;
		std::list<std::weak_ptr<NodeEntry>> nodes;
	};
	
public:
	NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _port = 30300);
	~NodeTable();
	
	/// Constants for Kademlia, mostly derived from address space.
	
	static unsigned const s_addressByteSize = sizeof(NodeId);				///< Size of address type in bytes.
	static unsigned const s_bits = 8 * s_addressByteSize;					///< Denoted by n in [Kademlia].
	static unsigned const s_bins = s_bits - 1;								///< Size of m_state (excludes root, which is us).
	static unsigned const s_maxSteps = boost::static_log2<s_bits>::value;	///< Max iterations of discovery. (doFindNode)
	
	/// Chosen constants
	
	static unsigned const s_bucketSize = 16;		///< Denoted by k in [Kademlia]. Number of nodes stored in each bucket.
	static unsigned const s_alpha = 3;				///< Denoted by \alpha in [Kademlia]. Number of concurrent FindNode requests.
	
	/// Intervals
	
	boost::posix_time::milliseconds const c_evictionCheckInterval = boost::posix_time::milliseconds(75);	///< Interval at which eviction timeouts are checked.
	std::chrono::milliseconds const c_reqTimeout = std::chrono::milliseconds(300);						///< How long to wait for requests (evict, find iterations).
	std::chrono::seconds const c_bucketRefresh = std::chrono::seconds(3600);							///< Refresh interval prevents bucket from becoming stale. [Kademlia]
	
	static unsigned dist(NodeId const& _a, NodeId const& _b) { u512 d = _a ^ _b; unsigned ret; for (ret = 0; d >>= 1; ++ret) {}; return ret; }
	
	void join();
	
	NodeEntry root() const { return NodeEntry(m_node, m_node.publicKey(), m_node.endpoint.udp); }
	std::list<NodeId> nodes() const;
	std::list<NodeEntry> state() const;
	
	NodeEntry operator[](NodeId _id);
	
	
protected:
	/// Repeatedly sends s_alpha concurrent requests to nodes nearest to target, for nodes nearest to target, up to s_maxSteps rounds.
	void doFindNode(NodeId _node, unsigned _round = 0, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried = std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>>());

	/// Returns nodes nearest to target.
	std::vector<std::shared_ptr<NodeEntry>> findNearest(NodeId _target);
	
	void ping(bi::udp::endpoint _to) const;
	
	void ping(NodeEntry* _n) const;
	
	void evict(std::shared_ptr<NodeEntry> _leastSeen, std::shared_ptr<NodeEntry> _new);
	
	void noteNode(Public const& _pubk, bi::udp::endpoint const& _endpoint);
	
	void noteNode(std::shared_ptr<NodeEntry> _n);
	
	void dropNode(std::shared_ptr<NodeEntry> _n);
	
	NodeBucket& bucket(NodeEntry const* _n);

	/// General Network Events
	
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
	/// Sends FindNeighbor packet. See doFindNode.
	void requestNeighbours(NodeEntry const& _node, NodeId _target) const;

	Node m_node;												///< This node.
	Secret m_secret;											///< This nodes secret key.

	mutable Mutex x_nodes;									///< Mutable for thread-safe copy in nodes() const.
	std::map<NodeId, std::shared_ptr<NodeEntry>> m_nodes;		///< NodeId -> Node table (most common lookup path)

	mutable Mutex x_state;
	std::array<NodeBucket, s_bins> m_state;					///< State table of binned nodes.

	Mutex x_evictions;
	std::deque<EvictionTimeout> m_evictions;					///< Eviction timeouts.
	
	std::shared_ptr<NodeSocket> m_socket;						///< Shared pointer for our UDPSocket; ASIO requires shared_ptr.
	NodeSocket* m_socketPtr;									///< Set to m_socket.get().
	ba::io_service& m_io;										///< Used by bucket refresh timer.
	boost::asio::deadline_timer m_bucketRefreshTimer;			///< Timer which schedules and enacts bucket refresh.
	boost::asio::deadline_timer m_evictionCheckTimer;			///< Timer for handling node evictions.
};
	
inline std::ostream& operator<<(std::ostream& _out, NodeTable const& _nodeTable)
{
	_out << _nodeTable.root().address() << "\t" << "0\t" << _nodeTable.root().endpoint.udp.address() << ":" << _nodeTable.root().endpoint.udp.port() << std::endl;
	auto s = _nodeTable.state();
	for (auto n: s)
		_out << n.address() << "\t" << n.distance << "\t" << n.endpoint.udp.address() << ":" << n.endpoint.udp.port() << std::endl;
	return _out;
}

/**
 * Ping packet: Check if node is alive.
 * PingNode is cached and regenerated after expiration - t, where t is timeout.
 *
 * RLP Encoded Items: 3
 * Minimum Encoded Size: 18 bytes
 * Maximum Encoded Size:  bytes // todo after u128 addresses
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
struct PingNode: RLPXDatagram<PingNode>
{
	PingNode(bi::udp::endpoint _ep): RLPXDatagram<PingNode>(_ep) {}
	PingNode(bi::udp::endpoint _ep, std::string _src, uint16_t _srcPort, std::chrono::seconds _expiration = std::chrono::seconds(60)): RLPXDatagram<PingNode>(_ep), ipAddress(_src), port(_srcPort), expiration(futureFromEpoch(_expiration)) {}

	std::string ipAddress;
	unsigned port;
	unsigned expiration;

	void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << expiration; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); ipAddress = r[0].toString(); port = r[1].toInt<unsigned>(); expiration = r[2].toInt<unsigned>(); }
};

/**
 * Pong packet: response to ping
 *
 * RLP Encoded Items: 1
 * Minimum Encoded Size: 33 bytes
 * Maximum Encoded Size: 33 bytes
 *
 * @todo expiration
 * @todo value of replyTo
 * @todo create from PingNode (reqs RLPXDatagram verify flag)
 */
struct Pong: RLPXDatagram<Pong>
{
	Pong(bi::udp::endpoint _ep): RLPXDatagram<Pong>(_ep) {}

	h256 replyTo; // hash of rlp of PingNode
	unsigned expiration;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(1); _s << replyTo; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); replyTo = (h256)r[0]; }
};

/**
 * FindNode Packet: Request k-nodes, closest to the target.
 * FindNode is cached and regenerated after expiration - t, where t is timeout.
 * FindNode implicitly results in finding neighbours of a given node.
 *
 * RLP Encoded Items: 2
 * Minimum Encoded Size: 21 bytes
 * Maximum Encoded Size: 30 bytes
 *
 * target: NodeId of node. The responding node will send back nodes closest to the target.
 * expiration: Triggers regeneration of packet. May also provide control over synchronization.
 *
 */
struct FindNode: RLPXDatagram<FindNode>
{
	FindNode(bi::udp::endpoint _ep): RLPXDatagram<FindNode>(_ep) {}
	FindNode(bi::udp::endpoint _ep, NodeId _target, std::chrono::seconds _expiration = std::chrono::seconds(30)): RLPXDatagram<FindNode>(_ep), target(_target), expiration(futureFromEpoch(_expiration)) {}
	
	h512 target;
	unsigned expiration;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s << target << expiration; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); target = r[0].toHash<h512>(); expiration = r[1].toInt<unsigned>(); }
};

/**
 * Node Packet: Multiple node packets are sent in response to FindNode.
 *
 * RLP Encoded Items: 2 (first item is list)
 * Minimum Encoded Size: 10 bytes
 *
 * @todo nonce: Should be replaced with expiration.
 */
struct Neighbours: RLPXDatagram<Neighbours>
{
	struct Node
	{
		Node() = default;
		Node(RLP const& _r) { interpretRLP(_r); }
		std::string ipAddress;
		unsigned port;
		NodeId node;
		void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << node; }
		void interpretRLP(RLP const& _r) { ipAddress = _r[0].toString(); port = _r[1].toInt<unsigned>(); node = h512(_r[2].toBytes()); }
	};
	
	Neighbours(bi::udp::endpoint _ep): RLPXDatagram<Neighbours>(_ep) {}
	Neighbours(bi::udp::endpoint _to, std::vector<std::shared_ptr<NodeTable::NodeEntry>> const& _nearest, unsigned _offset = 0, unsigned _limit = 0): RLPXDatagram<Neighbours>(_to)
	{
		auto limit = _limit ? std::min(_nearest.size(), (size_t)(_offset + _limit)) : _nearest.size();
		for (auto i = _offset; i < limit; i++)
		{
			Node node;
			node.ipAddress = _nearest[i]->endpoint.udp.address().to_string();
			node.port = _nearest[i]->endpoint.udp.port();
			node.node = _nearest[i]->publicKey();
			nodes.push_back(node);
		}
	}
	
	std::list<Node> nodes;
	unsigned expiration = 1;

	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s.appendList(nodes.size()); for (auto& n: nodes) n.streamRLP(_s); _s << expiration; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); for (auto n: r[0]) nodes.push_back(Node(n)); expiration = r[1].toInt<unsigned>(); }
};

struct NodeTableWarn: public LogChannel { static const char* name() { return "!P!"; } static const int verbosity = 0; };
struct NodeTableNote: public LogChannel { static const char* name() { return "*P*"; } static const int verbosity = 1; };
struct NodeTableMessageSummary: public LogChannel { static const char* name() { return "-P-"; } static const int verbosity = 2; };
struct NodeTableConnect: public LogChannel { static const char* name() { return "+P+"; } static const int verbosity = 10; };
struct NodeTableMessageDetail: public LogChannel { static const char* name() { return "=P="; } static const int verbosity = 5; };
struct NodeTableTriviaSummary: public LogChannel { static const char* name() { return "-P-"; } static const int verbosity = 10; };
struct NodeTableTriviaDetail: public LogChannel { static const char* name() { return "=P="; } static const int verbosity = 11; };
struct NodeTableAllDetail: public LogChannel { static const char* name() { return "=P="; } static const int verbosity = 13; };
struct NodeTableEgress: public LogChannel { static const char* name() { return ">>P"; } static const int verbosity = 14; };
struct NodeTableIngress: public LogChannel { static const char* name() { return "<<P"; } static const int verbosity = 15; };

}
}