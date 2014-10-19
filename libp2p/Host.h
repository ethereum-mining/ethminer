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
/** @file EthereumHost.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <utility>
#include <thread>
#include <chrono>
#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libdevcore/RangeMask.h>
#include <libdevcrypto/Common.h>
#include "HostCapability.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{

class RLPStream;

namespace p2p
{

class Host;

enum class Origin
{
	Unknown,
	Self,
	SelfThird,
	PerfectThird,
	Perfect,
};

struct Node
{
	NodeId id;										///< Their id/public key.
	unsigned index;									///< Index into m_nodesList
	bi::tcp::endpoint address;						///< As reported from the node itself.
	int score = 0;									///< All time cumulative.
	int rating = 0;									///< Trending.
	bool dead = false;								///< If true, we believe this node is permanently dead - forget all about it.
	std::chrono::system_clock::time_point lastConnected;
	std::chrono::system_clock::time_point lastAttempted;
	unsigned failedAttempts = 0;
	DisconnectReason lastDisconnect = NoDisconnect;	///< Reason for disconnect that happened last.

	Origin idOrigin = Origin::Unknown;				///< How did we get to know this node's id?

	int secondsSinceLastConnected() const { return lastConnected == std::chrono::system_clock::time_point() ? -1 : (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - lastConnected).count(); }
	int secondsSinceLastAttempted() const { return lastAttempted == std::chrono::system_clock::time_point() ? -1 : (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - lastAttempted).count(); }

	unsigned fallbackSeconds() const;
	bool shouldReconnect() const;

	bool isOffline() const { return lastAttempted > lastConnected; }
	bool operator<(Node const& _n) const
	{
		if (isOffline() != _n.isOffline())
			return isOffline();
		else if (isOffline())
			if (lastAttempted == _n.lastAttempted)
				return failedAttempts < _n.failedAttempts;
			else
				return lastAttempted < _n.lastAttempted;
		else
			if (score == _n.score)
				if (rating == _n.rating)
					return failedAttempts < _n.failedAttempts;
				else
					return rating < _n.rating;
			else
				return score < _n.score;
	}
};

using Nodes = std::vector<Node>;

struct NetworkPreferences
{
	NetworkPreferences(unsigned short p = 30303, std::string i = std::string(), bool u = true, bool l = false): listenPort(p), publicIP(i), upnp(u), localNetworking(l) {}

	unsigned short listenPort = 30303;
	std::string publicIP;
	bool upnp = true;
	bool localNetworking = false;
};

/**
 * @brief The Host class
 * Capabilities should be registered prior to startNetwork, since m_capabilities is not thread-safe.
 */
class Host: public Worker
{
	friend class Session;
	friend class HostCapabilityFace;
	friend struct Node;

public:
	/// Start server, listening for connections on the given port.
	Host(std::string const& _clientVersion, NetworkPreferences const& _n = NetworkPreferences(), bool _start = false);

	/// Will block on network process events.
	virtual ~Host();

	/// Closes all peers.
	void disconnectPeers();

	/// Basic peer network protocol version.
	unsigned protocolVersion() const;

	/// Register a peer-capability; all new peer connections will have this capability.
	template <class T> std::shared_ptr<T> registerCapability(T* _t) { _t->m_host = this; auto ret = std::shared_ptr<T>(_t); m_capabilities[std::make_pair(T::staticName(), T::staticVersion())] = ret; return ret; }

	bool haveCapability(CapDesc const& _name) const { return m_capabilities.count(_name) != 0; }
	CapDescs caps() const { CapDescs ret; for (auto const& i: m_capabilities) ret.push_back(i.first); return ret; }
	template <class T> std::shared_ptr<T> cap() const { try { return std::static_pointer_cast<T>(m_capabilities.at(std::make_pair(T::staticName(), T::staticVersion()))); } catch (...) { return nullptr; } }

	/// Connect to a peer explicitly.
	static std::string pocHost();
	void connect(std::string const& _addr, unsigned short _port = 30303) noexcept;
	void connect(bi::tcp::endpoint const& _ep);
	void connect(std::shared_ptr<Node> const& _n);

	/// @returns true iff we have the a peer of the given id.
	bool havePeer(NodeId _id) const;

	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	/// Get peer information.
	PeerInfos peers(bool _updatePing = false) const;

	/// Get number of peers connected; equivalent to, but faster than, peers().size().
	size_t peerCount() const { RecursiveGuard l(x_peers); return m_peers.size(); }

	/// Ping the peers, to update the latency information.
	void pingAll();

	/// Get the port we're listening on currently.
	unsigned short listenPort() const { return m_public.port(); }

	/// Serialise the set of known peers.
	bytes saveNodes() const;

	/// Deserialise the data and populate the set of known peers.
	void restoreNodes(bytesConstRef _b);

	Nodes nodes() const { RecursiveGuard l(x_peers); Nodes ret; for (auto const& i: m_nodes) ret.push_back(*i.second); return ret; }

	void setNetworkPreferences(NetworkPreferences const& _p) { stop(); m_netPrefs = _p; start(); }

	void start();
	void stop();
	bool isStarted() const { return isWorking(); }

	void quit();

	NodeId id() const { return m_key.pub(); }

	void registerPeer(std::shared_ptr<Session> _s, CapDescs const& _caps);

	std::shared_ptr<Node> node(NodeId _id) const { if (m_nodes.count(_id)) return m_nodes.at(_id); return std::shared_ptr<Node>(); }

private:
	void seal(bytes& _b);
	void populateAddresses();
	void determinePublic(std::string const& _publicAddress, bool _upnp);
	void ensureAccepting();

	void growPeers();
	void prunePeers();

	virtual void startedWorking();

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread or otherwise asynchronously, but you get this call every 100ms or so anyway.
	/// This won't touch alter the blockchain.
	virtual void doWork();

	std::shared_ptr<Node> noteNode(NodeId _id, bi::tcp::endpoint _a, Origin _o, bool _ready, NodeId _oldId = h256());
	Nodes potentialPeers(RangeMask<unsigned> const& _known);

	std::string m_clientVersion;											///< Our version string.

	NetworkPreferences m_netPrefs;											///< Network settings.

	static const int NetworkStopped = -1;									///< The value meaning we're not actually listening.
	int m_listenPort = NetworkStopped;										///< What port are we listening on?

	std::unique_ptr<ba::io_service> m_ioService;							///< IOService for network stuff.
	bi::tcp::acceptor m_acceptor;											///< Listening acceptor.
	bi::tcp::socket m_socket;												///< Listening socket.

	UPnP* m_upnp = nullptr;													///< UPnP helper.
	bi::tcp::endpoint m_public;												///< Our public listening endpoint.
	KeyPair m_key;															///< Our unique ID.

	bool m_hadNewNodes = false;

	mutable RecursiveMutex x_peers;

	/// The nodes to which we are currently connected.
	/// Mutable because we flush zombie entries (null-weakptrs) as regular maintenance from a const method.
	mutable std::map<NodeId, std::weak_ptr<Session>> m_peers;

	/// Nodes to which we may connect (or to which we have connected).
	/// TODO: does this need a lock?
	std::map<NodeId, std::shared_ptr<Node> > m_nodes;

	/// A list of node IDs. This contains every index from m_nodes; the order is guaranteed to remain the same.
	std::vector<NodeId> m_nodesList;

	RangeMask<unsigned> m_ready;											///< Indices into m_nodesList over to which nodes we are not currently connected, connecting or otherwise ignoring.
	RangeMask<unsigned> m_private;											///< Indices into m_nodesList over to which nodes are private.

	unsigned m_idealPeerCount = 5;											///< Ideal number of peers to be connected to.

	// Our addresses.
	std::vector<bi::address> m_addresses;									///< Addresses for us.
	std::vector<bi::address> m_peerAddresses;								///< Addresses that peers (can) know us by.

	// Our capabilities.
	std::map<CapDesc, std::shared_ptr<HostCapabilityFace>> m_capabilities;	///< Each of the capabilities we support.

	std::chrono::steady_clock::time_point m_lastPing;						///< Time we sent the last ping to all peers.

	bool m_accepting = false;
};

}
}
