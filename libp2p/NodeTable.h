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
 */
struct PingNode: RLPDatagram
{
	bytes ipAddress;
	uint16_t port;
	uint64_t expiration;

	Signature signature;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(3); _s << ipAddress << port << expiration; }
};

struct Pong: RLPDatagram
{
	// todo: weak-signed pong
	Address from;
	uint64_t replyTo;	/// expiration from PingNode
	
	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s << from << replyTo; }
};

/**
 * FindNeighbors Packet: Request k-nodes, closest to the target.
 * FindNeighbors is cached and regenerated after expiration - t, where t is timeout.
 *
 * signature: Signature of message.
 * target: Address of NodeId. The responding node will send back nodes closest to the target.
 * expiration: Triggers regeneration of packet. May also provide control over synchronization.
 *
 */
struct FindNeighbors: RLPDatagram
{
	h160 target;
	uint64_t expiration;
	
	Signature signature;
	
	void streamRLP(RLPStream& _s) const { _s.appendList(2); _s << target << expiration; }
};

/**
 * Node Packet: Multiple node packets are sent in response to FindNeighbors.
 */
struct Neighbors: RLPDatagram
{
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
 * @todo don't try to evict node if node isRequired. (support for makeRequired)
 * @todo optimize (use tree for state (or set w/custom compare for cache))
 * @todo constructor support for m_node, m_secret
 * @todo use s_bitsPerStep for find and refresh/ping
 * @todo exclude bucket from refresh if we have node as peer
 * @todo restore nodes
 */
class NodeTable: UDPSocketEvents, public std::enable_shared_from_this<NodeTable>
{
	using nodeSocket = UDPSocket<NodeTable, 1024>;
	using timePoint = std::chrono::steady_clock::time_point;
	
	static unsigned const s_bucketSize = 16;			// Denoted by k in [Kademlia]. Number of nodes stored in each bucket.
//	const unsigned s_bitsPerStep = 5;					// @todo Denoted by b in [Kademlia]. Bits by which address space will be divided for find responses.
	static unsigned const s_alpha = 3;				// Denoted by \alpha in [Kademlia]. Number of concurrent FindNeighbors requests.
	const unsigned s_findTimout = 300;				// How long to wait between find queries.
//	const unsigned s_siblings = 5;					// @todo Denoted by s in [S/Kademlia]. User-defined by sub-protocols.
	const unsigned s_bucketRefresh = 3600;				// Refresh interval prevents bucket from becoming stale. [Kademlia]
	static unsigned const s_bits = 8 * Address::size;	// Denoted by n.
	static unsigned const s_bins = s_bits - 1;			//
	const unsigned s_evictionCheckInterval = 75;		// Interval by which eviction timeouts are checked.
	const unsigned s_pingTimeout = 500;
	
public:
	static unsigned dist(Address const& _a, Address const& _b) { u160 d = _a ^ _b; unsigned ret; for (ret = 0; d >>= 1; ++ret) {}; return ret; }

	struct NodeDefaultEndpoint
	{
		NodeDefaultEndpoint(bi::udp::endpoint _udp): udp(_udp) {}
		bi::udp::endpoint udp;
	};
	
	struct NodeEntry
	{
		NodeEntry(Address _id, Public _pubk, bi::udp::endpoint _udp): id(_id), pubk(_pubk), endpoint(NodeDefaultEndpoint(_udp)), distance(0) {}
		NodeEntry(NodeEntry _src, Address _id, Public _pubk, bi::udp::endpoint _udp): id(_id), pubk(_pubk), endpoint(NodeDefaultEndpoint(_udp)), distance(dist(_src.id,_id)) {}
		NodeEntry(NodeEntry _src, Address _id, Public _pubk, NodeDefaultEndpoint _gw): id(_id), pubk(_pubk), endpoint(_gw), distance(dist(_src.id,_id)) {}
		Address id;
		Public pubk;
		NodeDefaultEndpoint endpoint;		///< How we've previously connected to this node. (must match node's reported endpoint)
		const unsigned distance;
		timePoint activePing;
	};
	
	struct NodeBucket
	{
		unsigned distance;
		timePoint modified;
		std::list<std::weak_ptr<NodeEntry>> nodes;
	};
	
	using EvictionTimeout = std::pair<std::pair<Address,timePoint>,Address>;

	NodeTable(ba::io_service& _io);
	~NodeTable();
	
	void join();
	
	std::list<Address> nodes() const;
	
	NodeEntry operator[](Address _id);
	
protected:
	void requestNeighbors(NodeEntry const& _node, Address _target) const;
	
	/// Sends requests to other nodes requesting nodes "near" to us in order to populate node table such that connected nodes form centrality.
	void doFindNode(Address _node, unsigned _round = 0, std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>> _tried = std::shared_ptr<std::set<std::shared_ptr<NodeEntry>>>());

	std::vector<std::shared_ptr<NodeEntry>> findNearest(Address _target);
	
	void ping(bi::address _address, unsigned _port) const;
	
	void ping(NodeEntry* _n) const;
	
	void evict(std::shared_ptr<NodeEntry> _leastSeen, std::shared_ptr<NodeEntry> _new);
	
	void noteNode(Public _pubk, bi::udp::endpoint _endpoint);
	
	void noteNode(std::shared_ptr<NodeEntry> _n);
	
	void dropNode(std::shared_ptr<NodeEntry> _n);
	
	NodeBucket const& bucket(NodeEntry* _n) const;
	
	void onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packet);
	
	void onDisconnected(UDPSocketFace*) {};
	
	void doCheckEvictions(boost::system::error_code const& _ec);
	
	void doRefreshBuckets(boost::system::error_code const& _ec);
	
private:
	NodeEntry m_node;										///< This node.
	Secret m_secret;											///< This nodes secret key.

	mutable Mutex x_nodes;									///< Mutable for thread-safe copy in nodes() const.
	std::map<Address, std::shared_ptr<NodeEntry>> m_nodes;		///< Address -> Node table (most common lookup path)

	Mutex x_state;
	std::array<NodeBucket, s_bins> m_state;					///< State table of binned nodes.

	Mutex x_evictions;
	std::deque<EvictionTimeout> m_evictions;					///< Eviction timeouts.
	
	std::shared_ptr<nodeSocket> m_socket;							///< Shared pointer for our UDPSocket; ASIO requires shared_ptr.
	nodeSocket* m_socketPtr;									///< Set to m_socket.get().
	ba::io_service& m_io;										///< Used by bucket refresh timer.
	boost::asio::deadline_timer m_bucketRefreshTimer;			///< Timer which schedules and enacts bucket refresh.
	boost::asio::deadline_timer m_evictionCheckTimer;			///< Timer for handling node evictions.
};

}
}