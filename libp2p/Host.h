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
#include <libdevcore/RangeMask.h>
#include <libdevcrypto/Common.h>
#include "NodeTable.h"
#include "HostCapability.h"
#include "Network.h"
#include "Peer.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

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

/**
 * @brief The Host class
 * Capabilities should be registered prior to startNetwork, since m_capabilities is not thread-safe.
 *
 * @todo exceptions when nodeTable not set (prior to start)
 * @todo onNodeTableEvent: move peer-connection logic into ensurePeers
 * @todo handshake: gracefully disconnect peer if peer already connected
 * @todo abstract socket -> IPConnection
 * @todo determinePublic: ipv6, udp
 * @todo handle conflict if addNode/requireNode called and Node already exists w/conflicting tcp or udp port
 * @todo write host identifier to disk w/nodes
 * @todo per-session keepalive/ping instead of broadcast; set ping-timeout via median-latency
 * @todo configuration-management (NetworkPrefs+Keys+Topology)
 */
class Host: public Worker
{
	friend class HostNodeTableHandler;
	friend class Session;
	friend class HostCapabilityFace;
	
public:
	/// Start server, listening for connections on the given port.
	Host(std::string const& _clientVersion, NetworkPreferences const& _n = NetworkPreferences(), bytesConstRef _restoreNetwork = bytesConstRef());

	/// Will block on network process events.
	virtual ~Host();
	
	/// Interval at which Host::run will call keepAlivePeers to ping peers.
	std::chrono::seconds const c_keepAliveInterval = std::chrono::seconds(30);

	/// Disconnect timeout after failure to respond to keepAlivePeers ping.
	std::chrono::milliseconds const c_keepAliveTimeOut = std::chrono::milliseconds(1000);
	
	/// Default host for current version of client.
	static std::string pocHost();
	
	/// Basic peer network protocol version.
	unsigned protocolVersion() const;

	/// Register a peer-capability; all new peer connections will have this capability.
	template <class T> std::shared_ptr<T> registerCapability(T* _t) { _t->m_host = this; auto ret = std::shared_ptr<T>(_t); m_capabilities[std::make_pair(T::staticName(), T::staticVersion())] = ret; return ret; }

	bool haveCapability(CapDesc const& _name) const { return m_capabilities.count(_name) != 0; }
	CapDescs caps() const { CapDescs ret; for (auto const& i: m_capabilities) ret.push_back(i.first); return ret; }
	template <class T> std::shared_ptr<T> cap() const { try { return std::static_pointer_cast<T>(m_capabilities.at(std::make_pair(T::staticName(), T::staticVersion()))); } catch (...) { return nullptr; } }
	
	bool havePeerSession(NodeId _id) { RecursiveGuard l(x_sessions); return m_sessions.count(_id) ? !!m_sessions[_id].lock() : false; }
	
	void addNode(NodeId const& _node, std::string const& _addr, unsigned short _tcpPort, unsigned short _udpPort);
	
	/// Set ideal number of peers.
	void setIdealPeerCount(unsigned _n) { m_idealPeerCount = _n; }

	/// Get peer information.
	PeerSessionInfos peerSessionInfo() const;
	
	/// Get number of peers connected.
	size_t peerCount() const;
	
	/// Get the address we're listening on currently.
	std::string listenAddress() const { return m_tcpPublic.address().to_string(); }
	
	/// Get the port we're listening on currently.
	unsigned short listenPort() const { return m_tcpPublic.port(); }

	/// Serialise the set of known peers.
	bytes saveNetwork() const;

	// TODO: P2P this should be combined with peers into a HostStat object of some kind; coalesce data, as it's only used for status information.
	Peers getPeers() const { RecursiveGuard l(x_sessions); Peers ret; for (auto const& i: m_peers) ret.push_back(*i.second); return ret; }

	void setNetworkPreferences(NetworkPreferences const& _p) { auto had = isStarted(); if (had) stop(); m_netPrefs = _p; if (had) start(); }

	/// Start network. @threadsafe
	void start();
	
	/// Stop network. @threadsafe
	/// Resets acceptor, socket, and IO service. Called by deallocator.
	void stop();
	
	/// @returns if network is running.
	bool isStarted() const { return m_run; }

	NodeId id() const { return m_alias.pub(); }

	void registerPeer(std::shared_ptr<Session> _s, CapDescs const& _caps);

protected:
	void onNodeTableEvent(NodeId const& _n, NodeTableEventType const& _e);

	/// Deserialise the data and populate the set of known peers.
	void restoreNetwork(bytesConstRef _b);
	
private:
	/// Populate m_peerAddresses with available public addresses.
	void determinePublic(std::string const& _publicAddress, bool _upnp);
	
	void connect(std::shared_ptr<Peer> const& _p);
	
	/// Ping the peers to update the latency information and disconnect peers which have timed out.
	void keepAlivePeers();
	
	/// Disconnect peers which didn't respond to keepAlivePeers ping prior to c_keepAliveTimeOut.
	void disconnectLatePeers();
	
	/// Called only from startedWorking().
	void runAcceptor();
	
	/// Handler for verifying handshake siganture before creating session. _nodeId is passed for outbound connections. If successful, socket is moved to Session via std::move.
	void doHandshake(bi::tcp::socket* _socket, NodeId _nodeId = NodeId());
	
	void seal(bytes& _b);

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
	std::vector<bi::address> m_ifAddresses;								///< Interface addresses.

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
	std::map<NodeId, std::shared_ptr<Peer>> m_peers;
	
	/// The nodes to which we are currently connected. Used by host to service peer requests and keepAlivePeers and for shutdown. (see run())
	/// Mutable because we flush zombie entries (null-weakptrs) as regular maintenance from a const method.
	mutable std::map<NodeId, std::weak_ptr<Session>> m_sessions;
	mutable RecursiveMutex x_sessions;

	unsigned m_idealPeerCount = 5;										///< Ideal number of peers to be connected to.
	
	std::set<bi::address> m_peerAddresses;									///< Public addresses that peers (can) know us by.

	std::map<CapDesc, std::shared_ptr<HostCapabilityFace>> m_capabilities;	///< Each of the capabilities we support.

	std::chrono::steady_clock::time_point m_lastPing;						///< Time we sent the last ping to all peers.

	bool m_accepting = false;
};
	
}
}
