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
#include <deque>
#include <boost/integer/static_log2.hpp>
#include <libp2p/UDP.h>
#include "Common.h"

namespace dev
{
namespace p2p
{

/**
 * NodeEntry
 * @brief Entry in Node Table
 */
struct NodeEntry: public Node
{
	NodeEntry(Node _src, Public _pubk, NodeIPEndpoint _gw);
	NodeEntry(Node _src, Public _pubk, bi::udp::endpoint _udp);

	unsigned const distance;	///< Node's distance (xor of _src as integer).
};

enum NodeTableEventType {
	NodeEntryAdded,
	NodeEntryRemoved
};
class NodeTable;
class NodeTableEventHandler
{
	friend class NodeTable;
public:
	virtual void processEvent(NodeId const& _n, NodeTableEventType const& _e) = 0;
	
protected:
	/// Called by NodeTable on behalf of an implementation (Host) to process new events without blocking nodetable.
	void processEvents()
	{
		std::list<std::pair<NodeId, NodeTableEventType>> events;
		{
			Guard l(x_events);
			if (!m_nodeEventHandler.size())
				return;
			m_nodeEventHandler.unique();
			for (auto const& n: m_nodeEventHandler)
				events.push_back(std::make_pair(n,m_events[n]));
			m_nodeEventHandler.clear();
			m_events.clear();
		}
		for (auto const& e: events)
			processEvent(e.first, e.second);
	}
	
	/// Called by NodeTable to append event.
	virtual void appendEvent(NodeId _n, NodeTableEventType _e) { Guard l(x_events); m_nodeEventHandler.push_back(_n); m_events[_n] = _e; }
	
	Mutex x_events;
	std::list<NodeId> m_nodeEventHandler;
	std::map<NodeId, NodeTableEventType> m_events;
};

class NodeTable;
inline std::ostream& operator<<(std::ostream& _out, NodeTable const& _nodeTable);
	
/**
 * NodeTable using modified kademlia for node discovery and preference.
 * Node table requires an IO service, creates a socket for incoming 
 * UDP messages and implements a kademlia-like protocol. Node requests and
 * responses are used to build a node table which can be queried to
 * obtain a list of potential nodes to connect to, and, passes events to
 * Host whenever a node is added or removed to/from the table.
 *
 * Thread-safety is ensured by modifying NodeEntry details via 
 * shared_ptr replacement instead of mutating values.
 *
 * NodeTable accepts a port for UDP and will listen to the port on all available
 * interfaces.
 *
 * [Integration]
 * @todo restore nodes: affects refreshbuckets
 * @todo TCP endpoints
 * @todo makeRequired: don't try to evict node if node isRequired.
 * @todo makeRequired: exclude bucket from refresh if we have node as peer.
 *
 * [Optimization]
 * @todo serialize evictions per-bucket
 * @todo store evictions in map, unit-test eviction logic
 * @todo store root node in table
 * @todo encapsulate discover into NetworkAlgorithm (task)
 * @todo Pong to include ip:port where ping was received
 * @todo expiration and sha3(id) 'to' for messages which are replies (prevents replay)
 * @todo cache Ping and FindSelf
 *
 * [Networking]
 * @todo node-endpoint updates
 * @todo TCP endpoints
 * @todo eth/upnp/natpmp/stun/ice/etc for public-discovery
 * @todo firewall
 *
 * [Protocol]
 * @todo optimize knowledge at opposite edges; eg, s_bitsPerStep lookups. (Can be done via pointers to NodeBucket)
 * @todo ^ s_bitsPerStep = 8; // Denoted by b in [Kademlia]. Bits by which address space is divided.
 */
class NodeTable: UDPSocketEvents, public std::enable_shared_from_this<NodeTable>
{
	friend std::ostream& operator<<(std::ostream& _out, NodeTable const& _nodeTable);
	using NodeSocket = UDPSocket<NodeTable, 1280>;
	using TimePoint = std::chrono::steady_clock::time_point;
	using EvictionTimeout = std::pair<std::pair<NodeId, TimePoint>, NodeId>;	///< First NodeId may be evicted and replaced with second NodeId.
	
public:
	NodeTable(ba::io_service& _io, KeyPair _alias, uint16_t _udpPort = 30303);
	~NodeTable();
	
	/// Returns distance based on xor metric two node ids. Used by NodeEntry and NodeTable.
	static unsigned distance(NodeId const& _a, NodeId const& _b) { u512 d = _a ^ _b; unsigned ret; for (ret = 0; d >>= 1; ++ret) {}; return ret; }
	
	/// Set event handler for NodeEntryAdded and NodeEntryRemoved events.
	void setEventHandler(NodeTableEventHandler* _handler) { m_nodeEventHandler.reset(_handler); }
	
	/// Called by implementation which provided handler to process NodeEntryAdded/NodeEntryRemoved events. Events are coalesced by type whereby old events are ignored.
	void processEvents();
	
	/// Add node. Node will be pinged if it's not already known.
	std::shared_ptr<NodeEntry> addNode(Public const& _pubk, bi::udp::endpoint const& _udp, bi::tcp::endpoint const& _tcp);
	
	/// Add node. Node will be pinged if it's not already known.
	std::shared_ptr<NodeEntry> addNode(Node const& _node);

	/// To be called when node table is empty. Runs node discovery with m_node.id as the target in order to populate node-table.
	void discover();
	
	/// Returns list of node ids active in node table.
	std::list<NodeId> nodes() const;
	
	/// Returns node count.
	unsigned count() const { return m_nodes.size(); }
	
	/// Returns snapshot of table.
	std::list<NodeEntry> snapshot() const;
	
	/// Returns true if node id is in node table.
	bool haveNode(NodeId const& _id) { Guard l(x_nodes); return m_nodes.count(_id) > 0; }
	
	/// Returns the Node to the corresponding node id or the empty Node if that id is not found.
	Node node(NodeId const& _id);
	
#ifndef BOOST_AUTO_TEST_SUITE
private:
#else
protected:
#endif
	
	/// Constants for Kademlia, derived from address space.
	
	static unsigned const s_addressByteSize = sizeof(NodeId);				///< Size of address type in bytes.
	static unsigned const s_bits = 8 * s_addressByteSize;					///< Denoted by n in [Kademlia].
	static unsigned const s_bins = s_bits - 1;								///< Size of m_state (excludes root, which is us).
	static unsigned const s_maxSteps = boost::static_log2<s_bits>::value;	///< Max iterations of discovery. (discover)
	
	/// Chosen constants
	
	static unsigned const s_bucketSize = 16;			///< Denoted by k in [Kademlia]. Number of nodes stored in each bucket.
	static unsigned const s_alpha = 3;				///< Denoted by \alpha in [Kademlia]. Number of concurrent FindNode requests.
	
	/// Intervals
	
	/* todo: replace boost::posix_time; change constants to upper camelcase */
	boost::posix_time::milliseconds const c_evictionCheckInterval = boost::posix_time::milliseconds(75);	///< Interval at which eviction timeouts are checked.
	std::chrono::milliseconds const c_reqTimeout = std::chrono::milliseconds(300);						///< How long to wait for requests (evict, find iterations).
	std::chrono::seconds const c_bucketRefresh = std::chrono::seconds(3600);							///< Refresh interval prevents bucket from becoming stale. [Kademlia]
	
	struct NodeBucket
	{
		unsigned distance;
		TimePoint modified;
		std::list<std::weak_ptr<NodeEntry>> nodes;
		void touch() { modified = std::chrono::steady_clock::now(); }
	};
	
	/// Used to ping endpoint.
	void ping(bi::udp::endpoint _to) const;
	
	/// Used ping known node. Used by node table when refreshing buckets and as part of eviction process (see evict).
	void ping(NodeEntry* _n) const;
	
	/// Returns center node entry which describes this node and used with dist() to calculate xor metric for node table nodes.
	NodeEntry center() const { return NodeEntry(m_node, m_node.publicKey(), m_node.endpoint.udp); }
	
	/// Used by asynchronous operations to return NodeEntry which is active and managed by node table.
	std::shared_ptr<NodeEntry> nodeEntry(NodeId _id);
	
	/// Used to discovery nodes on network which are close to the given target.
	/// Sends s_alpha concurrent requests to nodes nearest to target, for nodes nearest to target, up to s_maxSteps rounds.
	void discover(NodeId _target, unsigned _round = 0, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried = std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>>());

	/// Returns nodes from node table which are closest to target.
	std::vector<std::shared_ptr<NodeEntry>> nearestNodeEntries(NodeId _target);
	
	/// Asynchronously drops _leastSeen node if it doesn't reply and adds _new node, otherwise _new node is thrown away.
	void evict(std::shared_ptr<NodeEntry> _leastSeen, std::shared_ptr<NodeEntry> _new);
	
	/// Called whenever activity is received from an unknown node in order to maintain node table.
	void noteActiveNode(Public const& _pubk, bi::udp::endpoint const& _endpoint);

	/// Used to drop node when timeout occurs or when evict() result is to keep previous node.
	void dropNode(std::shared_ptr<NodeEntry> _n);
	
	/// Returns references to bucket which corresponds to distance of node id.
	/// @warning Only use the return reference locked x_state mutex.
	// TODO p2p: Remove this method after removing offset-by-one functionality.
	NodeBucket& bucket_UNSAFE(NodeEntry const* _n);

	/// General Network Events
	
	/// Called by m_socket when packet is received.
	void onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet);
	
	/// Called by m_socket when socket is disconnected.
	void onDisconnected(UDPSocketFace*) {}
	
	/// Tasks
	
	/// Called by evict() to ensure eviction check is scheduled to run and terminates when no evictions remain. Asynchronous.
	void doCheckEvictions(boost::system::error_code const& _ec);

	/// Purges and pings nodes for any buckets which haven't been touched for c_bucketRefresh seconds.
	void doRefreshBuckets(boost::system::error_code const& _ec);

	std::unique_ptr<NodeTableEventHandler> m_nodeEventHandler;	///< Event handler for node events.
	
	Node m_node;												///< This node.
	Secret m_secret;											///< This nodes secret key.

	mutable Mutex x_nodes;									///< LOCK x_state first if both locks are required. Mutable for thread-safe copy in nodes() const.
	std::map<NodeId, std::shared_ptr<NodeEntry>> m_nodes;		///< Nodes

	mutable Mutex x_state;									///< LOCK x_state first if both x_nodes and x_state locks are required.
	std::array<NodeBucket, s_bins> m_state;					///< State of p2p node network.

	Mutex x_evictions;										///< LOCK x_nodes first if both x_nodes and x_evictions locks are required.
	std::deque<EvictionTimeout> m_evictions;					///< Eviction timeouts.

	ba::io_service& m_io;										///< Used by bucket refresh timer.
	std::shared_ptr<NodeSocket> m_socket;						///< Shared pointer for our UDPSocket; ASIO requires shared_ptr.
	NodeSocket* m_socketPointer;								///< Set to m_socket.get(). Socket is created in constructor and disconnected in destructor to ensure access to pointer is safe.

	boost::asio::deadline_timer m_bucketRefreshTimer;			///< Timer which schedules and enacts bucket refresh.
	boost::asio::deadline_timer m_evictionCheckTimer;			///< Timer for handling node evictions.
};

inline std::ostream& operator<<(std::ostream& _out, NodeTable const& _nodeTable)
{
	_out << _nodeTable.center().address() << "\t" << "0\t" << _nodeTable.center().endpoint.udp.address() << ":" << _nodeTable.center().endpoint.udp.port() << std::endl;
	auto s = _nodeTable.snapshot();
	for (auto n: s)
		_out << n.address() << "\t" << n.distance << "\t" << n.endpoint.udp.address() << ":" << n.endpoint.udp.port() << std::endl;
	return _out;
}

/**
 * Ping packet: Sent to check if node is alive.
 * PingNode is cached and regenerated after expiration - t, where t is timeout.
 *
 * Ping is used to implement evict. When a new node is seen for
 * a given bucket which is full, the least-responsive node is pinged.
 * If the pinged node doesn't respond, then it is removed and the new
 * node is inserted.
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
 * @todo uint128_t for ip address (<->integer ipv4/6, asio-address, asio-endpoint)
 *
 */
struct PingNode: RLPXDatagram<PingNode>
{
	PingNode(bi::udp::endpoint _ep): RLPXDatagram<PingNode>(_ep) {}
	PingNode(bi::udp::endpoint _ep, std::string _src, uint16_t _srcPort, std::chrono::seconds _expiration = std::chrono::seconds(60)): RLPXDatagram<PingNode>(_ep), ipAddress(_src), port(_srcPort), expiration(futureFromEpoch(_expiration)) {}

	static const uint8_t type = 1;
	
	unsigned version = 1;
	std::string ipAddress;
	unsigned port;
	unsigned expiration;

	void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << expiration; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); ipAddress = r[0].toString(); port = r[1].toInt<unsigned>(); expiration = r[2].toInt<unsigned>(); }
};

/**
 * Pong packet: Sent in response to ping
 *
 * RLP Encoded Items: 2
 * Minimum Encoded Size: 33 bytes
 * Maximum Encoded Size: 33 bytes
 */
struct Pong: RLPXDatagram<Pong>
{
	Pong(bi::udp::endpoint _ep): RLPXDatagram<Pong>(_ep), expiration(futureFromEpoch(std::chrono::seconds(60))) {}

	static const uint8_t type = 2;

	h256 echo;				///< MCD of PingNode
	unsigned expiration;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s << echo << expiration; }
	void interpretRLP(bytesConstRef _bytes) { RLP r(_bytes); echo = (h256)r[0]; expiration = r[1].toInt<unsigned>(); }
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

	static const uint8_t type = 3;
	
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
	
	Neighbours(bi::udp::endpoint _ep): RLPXDatagram<Neighbours>(_ep), expiration(futureFromEpoch(std::chrono::seconds(30))) {}
	Neighbours(bi::udp::endpoint _to, std::vector<std::shared_ptr<NodeEntry>> const& _nearest, unsigned _offset = 0, unsigned _limit = 0): RLPXDatagram<Neighbours>(_to), expiration(futureFromEpoch(std::chrono::seconds(30)))
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
	
	static const uint8_t type = 4;
	std::vector<Node> nodes;
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
