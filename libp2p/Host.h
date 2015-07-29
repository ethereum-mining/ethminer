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
/** @file Host.h
 * @author Alex Leverington <nessence@gmail.com>
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
#include <libdevcrypto/Common.h>
#include <libdevcrypto/ECDHE.h>
#include "NodeTable.h"
#include "HostCapability.h"
#include "Network.h"
#include "Peer.h"
#include "RLPXSocket.h"
#include "RLPXFrameCoder.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

namespace std
{
template<> struct hash<pair<dev::p2p::NodeId, string>>
{
	size_t operator()(pair<dev::p2p::NodeId, string> const& _value) const
	{
		size_t ret = hash<dev::p2p::NodeId>()(_value.first);
		return ret ^ (hash<string>()(_value.second) + 0x9e3779b9 + (ret << 6) + (ret >> 2));
	}
};
}

namespace dev
{

namespace p2p
{

class Host;

class HostNodeTableHandler: public NodeTableEventHandler
{
public:
	HostNodeTableHandler(Host& _host);

	Host const& host() const { return m_host; }

private:
	virtual void processEvent(NodeId const& _n, NodeTableEventType const& _e);

	Host& m_host;
};

struct SubReputation
{
	bool isRude = false;
	int utility = 0;
	bytes data;
};

struct Reputation
{
	std::unordered_map<std::string, SubReputation> subs;
};

class ReputationManager
{
public:
	ReputationManager();

	void noteRude(Session const& _s, std::string const& _sub = std::string());
	bool isRude(Session const& _s, std::string const& _sub = std::string()) const;
	void setData(Session const& _s, std::string const& _sub, bytes const& _data);
	bytes data(Session const& _s, std::string const& _subs) const;

private:
	std::unordered_map<std::pair<p2p::NodeId, std::string>, Reputation> m_nodes;	///< Nodes that were impolite while syncing. We avoid syncing from these if possible.
	SharedMutex mutable x_nodes;
};

struct NodeInfo
{
	NodeInfo() = default;
	NodeInfo(NodeId const& _id, std::string const& _address, unsigned _port, std::string const& _version):
		id(_id), address(_address), port(_port), version(_version) {}

	std::string enode() const { return "enode://" + id.hex() + "@" + address + ":" + toString(port); }

	NodeId id;
	std::string address;
	unsigned port;
	std::string version;
};

/**
 * @brief The Host class
 * Capabilities should be registered prior to startNetwork, since m_capabilities is not thread-safe.
 *
 * @todo determinePublic: ipv6, udp
 * @todo per-session keepalive/ping instead of broadcast; set ping-timeout via median-latency
 */
class Host: public Worker
{
	friend class HostNodeTableHandler;
	friend class RLPXHandshake;
	
	friend class Session;
	friend class HostCapabilityFace;

public:
	/// Start server, listening for connections on the given port.
	Host(
		std::string const& _clientVersion,
		NetworkPreferences const& _n = NetworkPreferences(),
		bytesConstRef _restoreNetwork = bytesConstRef()
	);

	/// Will block on network process events.
	virtual ~Host();

	/// Default host for current version of client.
	static std::string pocHost();

	static std::unordered_map<Public, std::string> const& pocHosts();

	/// Register a peer-capability; all new peer connections will have this capability.
	template <class T> std::shared_ptr<T> registerCapability(T* _t) { _t->m_host = this; std::shared_ptr<T> ret(_t); m_capabilities[std::make_pair(T::staticName(), T::staticVersion())] = ret; return ret; }
	template <class T> void addCapability(std::shared_ptr<T> const & _p, std::string const& _name, u256 const& _version) { m_capabilities[std::make_pair(_name, _version)] = _p; }

	bool haveCapability(CapDesc const& _name) const { return m_capabilities.count(_name) != 0; }
	CapDescs caps() const { CapDescs ret; for (auto const& i: m_capabilities) ret.push_back(i.first); return ret; }
	template <class T> std::shared_ptr<T> cap() const { try { return std::static_pointer_cast<T>(m_capabilities.at(std::make_pair(T::staticName(), T::staticVersion()))); } catch (...) { return nullptr; } }

	/// Add node as a peer candidate. Node is added if discovery ping is successful and table has capacity.
	void addNode(NodeId const& _node, NodeIPEndpoint const& _endpoint);
	
	/// Create Peer and attempt keeping peer connected.
	void requirePeer(NodeId const& _node, NodeIPEndpoint const& _endpoint);

	/// Create Peer and attempt keeping peer connected.
	void requirePeer(NodeId const& _node, bi::address const& _addr, unsigned short _udpPort, unsigned short _tcpPort) { requirePeer(_node, NodeIPEndpoint(_addr, _udpPort, _tcpPort)); }

	/// Note peer as no longer being required.
	void relinquishPeer(NodeId const& _node);
	
	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	/// Get peer information.
	PeerSessionInfos peerSessionInfo() const;

	/// Get number of peers connected.
	size_t peerCount() const;

	/// Get the address we're listening on currently.
	std::string listenAddress() const { return m_netPrefs.listenIPAddress.empty() ? "0.0.0.0" : m_netPrefs.listenIPAddress; }

	/// Get the port we're listening on currently.
	unsigned short listenPort() const { return m_netPrefs.listenPort; }

	/// Serialise the set of known peers.
	bytes saveNetwork() const;

	// TODO: P2P this should be combined with peers into a HostStat object of some kind; coalesce data, as it's only used for status information.
	Peers getPeers() const { RecursiveGuard l(x_sessions); Peers ret; for (auto const& i: m_peers) ret.push_back(*i.second); return ret; }

	NetworkPreferences const& networkPreferences() const { return m_netPrefs; }

	void setNetworkPreferences(NetworkPreferences const& _p, bool _dropPeers = false) { m_dropPeers = _dropPeers; auto had = isStarted(); if (had) stop(); m_netPrefs = _p; if (had) start(); }

	/// Start network. @threadsafe
	void start();

	/// Stop network. @threadsafe
	/// Resets acceptor, socket, and IO service. Called by deallocator.
	void stop();

	/// @returns if network has been started.
	bool isStarted() const { return isWorking(); }

	/// @returns our reputation manager.
	ReputationManager& repMan() { return m_repMan; }

	/// @returns if network is started and interactive.
	bool haveNetwork() const { return m_run && !!m_nodeTable; }
	
	/// Validates and starts peer session, taking ownership of _io. Disconnects and returns false upon error.
	void startPeerSession(Public const& _id, RLP const& _hello, RLPXFrameCoder* _io, std::shared_ptr<RLPXSocket> const& _s);

	/// Get session by id
	std::shared_ptr<Session> peerSession(NodeId const& _id) { RecursiveGuard l(x_sessions); return m_sessions.count(_id) ? m_sessions[_id].lock() : std::shared_ptr<Session>(); }

	/// Get our current node ID.
	NodeId id() const { return m_alias.pub(); }

	/// Get the public TCP endpoint.
	bi::tcp::endpoint const& tcpPublic() const { return m_tcpPublic; }

	/// Get the public endpoint information.
	std::string enode() const { return "enode://" + id().hex() + "@" + (networkPreferences().publicIPAddress.empty() ? m_tcpPublic.address().to_string() : networkPreferences().publicIPAddress) + ":" + toString(m_tcpPublic.port()); }

	/// Get the node information.
	p2p::NodeInfo nodeInfo() const { return NodeInfo(id(), (networkPreferences().publicIPAddress.empty() ? m_tcpPublic.address().to_string() : networkPreferences().publicIPAddress), m_tcpPublic.port(), m_clientVersion); }

protected:
	void onNodeTableEvent(NodeId const& _n, NodeTableEventType const& _e);

	/// Deserialise the data and populate the set of known peers.
	void restoreNetwork(bytesConstRef _b);

private:
	enum PeerSlotRatio { Egress = 1, Ingress = 4 };
	
	bool havePeerSession(NodeId const& _id) { return !!peerSession(_id); }

	/// Determines and sets m_tcpPublic to publicly advertised address.
	void determinePublic();

	void connect(std::shared_ptr<Peer> const& _p);

	/// Returns true if pending and connected peer count is less than maximum
	bool peerSlotsAvailable(PeerSlotRatio _type) { Guard l(x_pendingNodeConns); return peerCount() + m_pendingPeerConns.size() < _type * m_idealPeerCount; }
	
	/// Ping the peers to update the latency information and disconnect peers which have timed out.
	void keepAlivePeers();

	/// Disconnect peers which didn't respond to keepAlivePeers ping prior to c_keepAliveTimeOut.
	void disconnectLatePeers();

	/// Called only from startedWorking().
	void runAcceptor();

	/// Called by Worker. Not thread-safe; to be called only by worker.
	virtual void startedWorking();
	/// Called by startedWorking. Not thread-safe; to be called only be Worker.
	void run(boost::system::error_code const& error);			///< Run network. Called serially via ASIO deadline timer. Manages connection state transitions.

	/// Run network. Not thread-safe; to be called only by worker.
	virtual void doWork();

	/// Shutdown network. Not thread-safe; to be called only by worker.
	virtual void doneWorking();

	/// Get or create host identifier (KeyPair).
	static KeyPair networkAlias(bytesConstRef _b);

	bytes m_restoreNetwork;										///< Set by constructor and used to set Host key and restore network peers & nodes.

	bool m_run = false;													///< Whether network is running.
	std::mutex x_runTimer;												///< Start/stop mutex.

	std::string m_clientVersion;											///< Our version string.

	NetworkPreferences m_netPrefs;										///< Network settings.

	/// Interface addresses (private, public)
	std::set<bi::address> m_ifAddresses;								///< Interface addresses.

	int m_listenPort = -1;												///< What port are we listening on. -1 means binding failed or acceptor hasn't been initialized.

	ba::io_service m_ioService;											///< IOService for network stuff.
	bi::tcp::acceptor m_tcp4Acceptor;										///< Listening acceptor.

	std::unique_ptr<boost::asio::deadline_timer> m_timer;					///< Timer which, when network is running, calls scheduler() every c_timerInterval ms.
	static const unsigned c_timerInterval = 100;							///< Interval which m_timer is run when network is connected.

	std::set<Peer*> m_pendingPeerConns;									/// Used only by connect(Peer&) to limit concurrently connecting to same node. See connect(shared_ptr<Peer>const&).
	Mutex x_pendingNodeConns;

	bi::tcp::endpoint m_tcpPublic;											///< Our public listening endpoint.
	KeyPair m_alias;															///< Alias for network communication. Network address is k*G. k is key material. TODO: Replace KeyPair.
	std::shared_ptr<NodeTable> m_nodeTable;									///< Node table (uses kademlia-like discovery).

	/// Shared storage of Peer objects. Peers are created or destroyed on demand by the Host. Active sessions maintain a shared_ptr to a Peer;
	std::unordered_map<NodeId, std::shared_ptr<Peer>> m_peers;
	
	/// Peers we try to connect regardless of p2p network.
	std::set<NodeId> m_requiredPeers;
	Mutex x_requiredPeers;

	/// The nodes to which we are currently connected. Used by host to service peer requests and keepAlivePeers and for shutdown. (see run())
	/// Mutable because we flush zombie entries (null-weakptrs) as regular maintenance from a const method.
	mutable std::unordered_map<NodeId, std::weak_ptr<Session>> m_sessions;
	mutable RecursiveMutex x_sessions;
	
	std::list<std::weak_ptr<RLPXHandshake>> m_connecting;					///< Pending connections.
	Mutex x_connecting;													///< Mutex for m_connecting.

	unsigned m_idealPeerCount = 11;										///< Ideal number of peers to be connected to.

	std::map<CapDesc, std::shared_ptr<HostCapabilityFace>> m_capabilities;	///< Each of the capabilities we support.
	
	/// Deadline timers used for isolated network events. GC'd by run.
	std::list<std::shared_ptr<boost::asio::deadline_timer>> m_timers;
	Mutex x_timers;

	std::chrono::steady_clock::time_point m_lastPing;						///< Time we sent the last ping to all peers.
	bool m_accepting = false;
	bool m_dropPeers = false;

	ReputationManager m_repMan;
};

}
}
