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
/** @file Host.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <set>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/Assertions.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/StructuredLogger.h>
#include <libethcore/Exceptions.h>
#include <libdevcrypto/FileSystem.h>
#include "Session.h"
#include "Common.h"
#include "Capability.h"
#include "UPnP.h"
#include "RLPxHandshake.h"
#include "Host.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

/// Interval at which Host::run will call keepAlivePeers to ping peers.
std::chrono::seconds const c_keepAliveInterval = std::chrono::seconds(30);

/// Disconnect timeout after failure to respond to keepAlivePeers ping.
std::chrono::milliseconds const c_keepAliveTimeOut = std::chrono::milliseconds(1000);

HostNodeTableHandler::HostNodeTableHandler(Host& _host): m_host(_host) {}

void HostNodeTableHandler::processEvent(NodeId const& _n, NodeTableEventType const& _e)
{
	m_host.onNodeTableEvent(_n, _e);
}

Host::Host(std::string const& _clientVersion, NetworkPreferences const& _n, bytesConstRef _restoreNetwork):
	Worker("p2p", 0),
	m_restoreNetwork(_restoreNetwork.toBytes()),
	m_clientVersion(_clientVersion),
	m_netPrefs(_n),
	m_ifAddresses(Network::getInterfaceAddresses()),
	m_ioService(2),
	m_tcp4Acceptor(m_ioService),
	m_alias(networkAlias(_restoreNetwork)),
	m_lastPing(chrono::steady_clock::time_point::min())
{
	for (auto address: m_ifAddresses)
		if (address.is_v4())
			clog(NetNote) << "IP Address: " << address << " = " << (isPrivateAddress(address) ? "[LOCAL]" : "[PEER]");

	clog(NetNote) << "Id:" << id();
}

Host::~Host()
{
	stop();
}

void Host::start()
{
	startWorking();
}

void Host::stop()
{
	// called to force io_service to kill any remaining tasks it might have -
	// such tasks may involve socket reads from Capabilities that maintain references
	// to resources we're about to free.

	{
		// Although m_run is set by stop() or start(), it effects m_runTimer so x_runTimer is used instead of a mutex for m_run.
		// when m_run == false, run() will cause this::run() to stop() ioservice
		Guard l(x_runTimer);
		// ignore if already stopped/stopping
		if (!m_run)
			return;
		m_run = false;
	}

	// wait for m_timer to reset (indicating network scheduler has stopped)
	while (!!m_timer)
		this_thread::sleep_for(chrono::milliseconds(50));

	// stop worker thread
	stopWorking();
}

void Host::doneWorking()
{
	// reset ioservice (allows manually polling network, below)
	m_ioService.reset();

	// shutdown acceptor
	m_tcp4Acceptor.cancel();
	if (m_tcp4Acceptor.is_open())
		m_tcp4Acceptor.close();

	// There maybe an incoming connection which started but hasn't finished.
	// Wait for acceptor to end itself instead of assuming it's complete.
	// This helps ensure a peer isn't stopped at the same time it's starting
	// and that socket for pending connection is closed.
	while (m_accepting)
		m_ioService.poll();

	// stop capabilities (eth: stops syncing or block/tx broadcast)
	for (auto const& h: m_capabilities)
		h.second->onStopping();

	// disconnect pending handshake, before peers, as a handshake may create a peer
	for (unsigned n = 0;; n = 0)
	{
		{
			Guard l(x_connecting);
			for (auto i: m_connecting)
				if (auto h = i.lock())
				{
					h->cancel();
					n++;
				}
		}
		if (!n)
			break;
		m_ioService.poll();
	}
	
	// disconnect peers
	for (unsigned n = 0;; n = 0)
	{
		{
			RecursiveGuard l(x_sessions);
			for (auto i: m_sessions)
				if (auto p = i.second.lock())
					if (p->isConnected())
					{
						p->disconnect(ClientQuit);
						n++;
					}
		}
		if (!n)
			break;

		// poll so that peers send out disconnect packets
		m_ioService.poll();
	}

	// stop network (again; helpful to call before subsequent reset())
	m_ioService.stop();

	// reset network (allows reusing ioservice in future)
	m_ioService.reset();

	// finally, clear out peers (in case they're lingering)
	RecursiveGuard l(x_sessions);
	m_sessions.clear();
}

void Host::startPeerSession(Public const& _id, RLP const& _rlp, RLPXFrameIO* _io, bi::tcp::endpoint _endpoint)
{
	shared_ptr<Peer> p;
	if (!m_peers.count(_id))
	{
		p.reset(new Peer());
		p->id = _id;
	}
	else
		p = m_peers[_id];
	p->m_lastDisconnect = NoDisconnect;
	if (p->isOffline())
		p->m_lastConnected = std::chrono::system_clock::now();
	p->m_failedAttempts = 0;
	p->endpoint.tcp.address(_endpoint.address());

	auto protocolVersion = _rlp[0].toInt<unsigned>();
	auto clientVersion = _rlp[1].toString();
	auto caps = _rlp[2].toVector<CapDesc>();
	auto listenPort = _rlp[3].toInt<unsigned short>();
	
	// clang error (previously: ... << hex << caps ...)
	// "'operator<<' should be declared prior to the call site or in an associated namespace of one of its arguments"
	stringstream capslog;
	for (auto cap: caps)
		capslog << "(" << cap.first << "," << dec << cap.second << ")";
	clog(NetMessageSummary) << "Hello: " << clientVersion << "V[" << protocolVersion << "]" << _id.abridged() << showbase << capslog.str() << dec << listenPort;
	
	// create session so disconnects are managed
	auto ps = make_shared<Session>(this, _io, p, PeerSessionInfo({_id, clientVersion, _endpoint.address().to_string(), listenPort, chrono::steady_clock::duration(), _rlp[2].toSet<CapDesc>(), 0, map<string, string>()}));
	if (protocolVersion != dev::p2p::c_protocolVersion)
	{
		ps->disconnect(IncompatibleProtocol);
		return;
	}
	
	{
		RecursiveGuard l(x_sessions);
		if (m_sessions.count(_id) && !!m_sessions[_id].lock())
			if (auto s = m_sessions[_id].lock())
				if(s->isConnected())
				{
					// Already connected.
					clog(NetWarn) << "Session already exists for peer with id" << _id.abridged();
					ps->disconnect(DuplicatePeer);
					return;
				}
		m_sessions[_id] = ps;
	}
	ps->start();
	unsigned o = (unsigned)UserPacket;
	for (auto const& i: caps)
		if (haveCapability(i))
		{
			ps->m_capabilities[i] = shared_ptr<Capability>(m_capabilities[i]->newPeerCapability(ps.get(), o));
			o += m_capabilities[i]->messageCount();
		}
	clog(NetNote) << "p2p.host.peer.register" << _id.abridged();
	StructuredLogger::p2pConnected(_id.abridged(), ps->m_peer->peerEndpoint(), ps->m_peer->m_lastConnected, clientVersion, peerCount());
}

void Host::onNodeTableEvent(NodeId const& _n, NodeTableEventType const& _e)
{

	if (_e == NodeEntryAdded)
	{
		clog(NetNote) << "p2p.host.nodeTable.events.nodeEntryAdded " << _n;

		auto n = m_nodeTable->node(_n);
		if (n)
		{
			shared_ptr<Peer> p;
			{
				RecursiveGuard l(x_sessions);
				if (m_peers.count(_n))
					p = m_peers[_n];
				else
				{
					// TODO p2p: construct peer from node
					p.reset(new Peer());
					p->id = _n;
					p->endpoint = NodeIPEndpoint(n.endpoint.udp, n.endpoint.tcp);
					p->required = n.required;
					m_peers[_n] = p;

					clog(NetNote) << "p2p.host.peers.events.peersAdded " << _n << "udp:" << p->endpoint.udp.address() << "tcp:" << p->endpoint.tcp.address();
				}
				p->endpoint.tcp = n.endpoint.tcp;
			}

			// TODO: Implement similar to discover. Attempt connecting to nodes
			//       until ideal peer count is reached; if all nodes are tried,
			//       repeat. Notably, this is an integrated process such that
			//       when onNodeTableEvent occurs we should also update +/-
			//       the list of nodes to be tried. Thus:
			//       1) externalize connection attempts
			//       2) attempt copies potentialPeer list
			//       3) replace this logic w/maintenance of potentialPeers
			if (peerCount() < m_idealPeerCount)
				connect(p);
		}
	}
	else if (_e == NodeEntryDropped)
	{
		clog(NetNote) << "p2p.host.nodeTable.events.NodeEntryDropped " << _n;

		RecursiveGuard l(x_sessions);
		m_peers.erase(_n);
	}
}

void Host::determinePublic()
{
	// set m_tcpPublic := listenIP (if public) > public > upnp > unspecified address.
	
	auto ifAddresses = Network::getInterfaceAddresses();
	auto laddr = m_netPrefs.listenIPAddress.empty() ? bi::address() : bi::address::from_string(m_netPrefs.listenIPAddress);
	auto lset = !laddr.is_unspecified();
	auto paddr = m_netPrefs.publicIPAddress.empty() ? bi::address() : bi::address::from_string(m_netPrefs.publicIPAddress);
	auto pset = !paddr.is_unspecified();
	
	bool listenIsPublic = lset && isPublicAddress(laddr);
	bool publicIsHost = !lset && pset && ifAddresses.count(paddr);
	
	bi::tcp::endpoint ep(bi::address(), m_netPrefs.listenPort);
	if (m_netPrefs.traverseNAT && listenIsPublic)
	{
		clog(NetNote) << "Listen address set to Public address:" << laddr << ". UPnP disabled.";
		ep.address(laddr);
	}
	else if (m_netPrefs.traverseNAT && publicIsHost)
	{
		clog(NetNote) << "Public address set to Host configured address:" << paddr << ". UPnP disabled.";
		ep.address(paddr);
	}
	else if (m_netPrefs.traverseNAT)
	{
		bi::address natIFAddr;
		if (lset && ifAddresses.count(laddr))
			ep = Network::traverseNAT(std::set<bi::address>({laddr}), m_netPrefs.listenPort, natIFAddr);
		else
			ep = Network::traverseNAT(ifAddresses, m_netPrefs.listenPort, natIFAddr);
		
		if (lset && natIFAddr != laddr)
			// if listen address is set we use it, even if upnp returns different
			clog(NetWarn) << "Listen address" << laddr << "differs from local address" << natIFAddr << "returned by UPnP!";
		
		if (pset && ep.address() != paddr)
		{
			// if public address is set we advertise it, even if upnp returns different
			clog(NetWarn) << "Specified public address" << paddr << "differs from external address" << ep.address() << "returned by UPnP!";
			ep.address(paddr);
		}
	}
	else if (pset)
		ep.address(paddr);

	m_tcpPublic = ep;
}

void Host::runAcceptor()
{
	assert(m_listenPort > 0);

	if (m_run && !m_accepting)
	{
		clog(NetConnect) << "Listening on local port " << m_listenPort << " (public: " << m_tcpPublic << ")";
		m_accepting = true;

		auto socket = make_shared<RLPXSocket>(new bi::tcp::socket(m_ioService));
		m_tcp4Acceptor.async_accept(socket->ref(), [=](boost::system::error_code ec)
		{
			// if no error code
			bool success = false;
			if (!ec)
			{
				try
				{
					// incoming connection; we don't yet know nodeid
					auto handshake = make_shared<RLPXHandshake>(this, socket);
					m_connecting.push_back(handshake);
					handshake->start();
					success = true;
				}
				catch (Exception const& _e)
				{
					clog(NetWarn) << "ERROR: " << diagnostic_information(_e);
				}
				catch (std::exception const& _e)
				{
					clog(NetWarn) << "ERROR: " << _e.what();
				}
			}

			if (!success)
				socket->ref().close();
			
			m_accepting = false;
			if (ec.value() < 1)
				runAcceptor();
		});
	}
}

string Host::pocHost()
{
	vector<string> strs;
	boost::split(strs, dev::Version, boost::is_any_of("."));
	return "poc-" + strs[1] + ".ethdev.com";
}

void Host::addNode(NodeId const& _node, std::string const& _addr, unsigned short _tcpPeerPort, unsigned short _udpNodePort)
{
	// TODO: p2p clean this up (bring tested acceptor code over from network branch)
	while (isWorking() && !m_run)
		this_thread::sleep_for(chrono::milliseconds(50));
	if (!m_run)
		return;

	if (_tcpPeerPort < 30300 || _tcpPeerPort > 30305)
		cwarn << "Non-standard port being recorded: " << _tcpPeerPort;

	if (_tcpPeerPort >= /*49152*/32768)
	{
		cwarn << "Private port being recorded - setting to 0";
		_tcpPeerPort = 0;
	}

	boost::system::error_code ec;
	bi::address addr = bi::address::from_string(_addr, ec);
	if (ec)
	{
		bi::tcp::resolver *r = new bi::tcp::resolver(m_ioService);
		r->async_resolve({_addr, toString(_tcpPeerPort)}, [=](boost::system::error_code const& _ec, bi::tcp::resolver::iterator _epIt)
		{
			if (!_ec)
			{
				bi::tcp::endpoint tcp = *_epIt;
				if (m_nodeTable) m_nodeTable->addNode(Node(_node, NodeIPEndpoint(bi::udp::endpoint(tcp.address(), _udpNodePort), tcp)));
			}
			delete r;
		});
	}
	else
		if (m_nodeTable) m_nodeTable->addNode(Node(_node, NodeIPEndpoint(bi::udp::endpoint(addr, _udpNodePort), bi::tcp::endpoint(addr, _tcpPeerPort))));
}

void Host::relinquishPeer(NodeId const& _node)
{
	Guard l(x_requiredPeers);
	if (m_requiredPeers.count(_node))
		m_requiredPeers.erase(_node);
}

void Host::requirePeer(NodeId const& _n, std::string const& _udpAddr, unsigned short _udpPort, std::string const& _tcpAddr, unsigned short _tcpPort)
{
	auto naddr = bi::address::from_string(_udpAddr);
	auto paddr = _tcpAddr.empty() ? naddr : bi::address::from_string(_tcpAddr);
	auto udp = bi::udp::endpoint(naddr, _udpPort);
	auto tcp = bi::tcp::endpoint(paddr, _tcpPort ? _tcpPort : _udpPort);
	Node node(_n, NodeIPEndpoint(udp, tcp));
	if (_n)
	{
		// add or replace peer
		shared_ptr<Peer> p;
		{
			RecursiveGuard l(x_sessions);
			if (m_peers.count(_n))
				p = m_peers[_n];
			else
			{
				p.reset(new Peer());
				p->id = _n;
				p->required = true;
				m_peers[_n] = p;
			}
			p->endpoint.udp = node.endpoint.udp;
			p->endpoint.tcp = node.endpoint.tcp;
		}
		connect(p);
	}
	else if (m_nodeTable)
	{
		shared_ptr<boost::asio::deadline_timer> t(new boost::asio::deadline_timer(m_ioService));
		m_timers.push_back(t);
		
		m_nodeTable->addNode(node);
		t->expires_from_now(boost::posix_time::milliseconds(600));
		t->async_wait([this, _n](boost::system::error_code const& _ec)
		{
			if (!_ec && m_nodeTable)
				if (auto n = m_nodeTable->node(_n))
					requirePeer(n.id, n.endpoint.udp.address().to_string(), n.endpoint.udp.port(), n.endpoint.tcp.address().to_string(), n.endpoint.tcp.port());
		});
	}
}

void Host::connect(std::shared_ptr<Peer> const& _p)
{
	if (!m_run)
		return;

	_p->m_lastAttempted = std::chrono::system_clock::now();
	
	if (havePeerSession(_p->id))
	{
		clog(NetConnect) << "Aborted connect. Node already connected.";
		return;
	}

	if (!m_nodeTable->haveNode(_p->id))
	{
		clog(NetWarn) << "Aborted connect. Node not in node table.";
		m_nodeTable->addNode(*_p.get());
		return;
	}

	// prevent concurrently connecting to a node
	Peer *nptr = _p.get();
	{
		Guard l(x_pendingNodeConns);
		if (m_pendingPeerConns.count(nptr))
			return;
		m_pendingPeerConns.insert(nptr);
	}

	clog(NetConnect) << "Attempting connection to node" << _p->id.abridged() << "@" << _p->peerEndpoint() << "from" << id().abridged();
	auto socket = make_shared<RLPXSocket>(new bi::tcp::socket(m_ioService));
	socket->ref().async_connect(_p->peerEndpoint(), [=](boost::system::error_code const& ec)
	{
		if (ec)
		{
			clog(NetConnect) << "Connection refused to node" << _p->id.abridged() << "@" << _p->peerEndpoint() << "(" << ec.message() << ")";
			_p->m_lastDisconnect = TCPError;
			_p->m_lastAttempted = std::chrono::system_clock::now();
			_p->m_failedAttempts++;
		}
		else
		{
			clog(NetConnect) << "Connecting to" << _p->id.abridged() << "@" << _p->peerEndpoint();
			auto handshake = make_shared<RLPXHandshake>(this, socket, _p->id);
			{
				Guard l(x_connecting);
				m_connecting.push_back(handshake);
			}
			
			// preempt setting failedAttempts; this value is cleared upon success
			_p->m_failedAttempts++;
			handshake->start();
		}
		
		Guard l(x_pendingNodeConns);
		m_pendingPeerConns.erase(nptr);
	});
}

PeerSessionInfos Host::peerSessionInfo() const
{
	if (!m_run)
		return PeerSessionInfos();

	std::vector<PeerSessionInfo> ret;
	RecursiveGuard l(x_sessions);
	for (auto& i: m_sessions)
		if (auto j = i.second.lock())
			if (j->isConnected())
				ret.push_back(j->m_info);
	return ret;
}

size_t Host::peerCount() const
{
	unsigned retCount = 0;
	RecursiveGuard l(x_sessions);
	for (auto& i: m_sessions)
		if (std::shared_ptr<Session> j = i.second.lock())
			if (j->isConnected())
				retCount++;
	return retCount;
}

void Host::run(boost::system::error_code const&)
{
	if (!m_run)
	{
		// reset NodeTable
		m_nodeTable.reset();

		// stopping io service allows running manual network operations for shutdown
		// and also stops blocking worker thread, allowing worker thread to exit
		m_ioService.stop();

		// resetting timer signals network that nothing else can be scheduled to run
		m_timer.reset();
		return;
	}

	m_nodeTable->processEvents();

	// cleanup zombies
	{
		Guard l(x_connecting);
		m_connecting.remove_if([](std::weak_ptr<RLPXHandshake> h){ return h.lock(); });
	}
	
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
			pp->serviceNodesRequest();

	keepAlivePeers();
	
	// At this time peers will be disconnected based on natural TCP timeout.
	// disconnectLatePeers needs to be updated for the assumption that Session
	// is always live and to ensure reputation and fallback timers are properly
	// updated. // disconnectLatePeers();

	auto openSlots = m_idealPeerCount - peerCount();
	if (openSlots > 0)
	{
		list<shared_ptr<Peer>> toConnect;
		{
			RecursiveGuard l(x_sessions);
			for (auto p: m_peers)
				if (p.second->shouldReconnect() && !havePeerSession(p.second->id))
					toConnect.push_back(p.second);
		}
		
		for (auto p: toConnect)
			if (openSlots--)
				connect(p);
			else
				break;
		
		m_nodeTable->discover();
	}

	auto runcb = [this](boost::system::error_code const& error) { run(error); };
	m_timer->expires_from_now(boost::posix_time::milliseconds(c_timerInterval));
	m_timer->async_wait(runcb);
}

void Host::startedWorking()
{
	asserts(!m_timer);

	{
		// prevent m_run from being set to true at same time as set to false by stop()
		// don't release mutex until m_timer is set so in case stop() is called at same
		// time, stop will wait on m_timer and graceful network shutdown.
		Guard l(x_runTimer);
		// create deadline timer
		m_timer.reset(new boost::asio::deadline_timer(m_ioService));
		m_run = true;
	}

	// start capability threads (ready for incoming connections)
	for (auto const& h: m_capabilities)
		h.second->onStarting();
	
	// try to open acceptor (todo: ipv6)
	m_listenPort = Network::tcp4Listen(m_tcp4Acceptor, m_netPrefs);

	// determine public IP, but only if we're able to listen for connections
	// todo: GUI when listen is unavailable in UI
	if (m_listenPort)
	{
		determinePublic();

		if (m_listenPort > 0)
			runAcceptor();
	}
	else
		clog(NetNote) << "p2p.start.notice id:" << id().abridged() << "Listen port is invalid or unavailable. Node Table using default port (30303).";

	// this doesn't work unless local-networking is enabled because the port is -1
	m_nodeTable.reset(new NodeTable(m_ioService, m_alias, bi::address::from_string(listenAddress()), listenPort() > 0 ? listenPort() : 30303));
	m_nodeTable->setEventHandler(new HostNodeTableHandler(*this));
	restoreNetwork(&m_restoreNetwork);

	clog(NetNote) << "p2p.started id:" << id().abridged();

	run(boost::system::error_code());
}

void Host::doWork()
{
	if (m_run)
		m_ioService.run();
}

void Host::keepAlivePeers()
{
	if (chrono::steady_clock::now() - c_keepAliveInterval < m_lastPing)
		return;

	RecursiveGuard l(x_sessions);
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
				pp->ping();

	m_lastPing = chrono::steady_clock::now();
}

void Host::disconnectLatePeers()
{
	auto now = chrono::steady_clock::now();
	if (now - c_keepAliveTimeOut < m_lastPing)
		return;

	RecursiveGuard l(x_sessions);
	for (auto p: m_sessions)
		if (auto pp = p.second.lock())
			if (now - c_keepAliveTimeOut > m_lastPing && pp->m_lastReceived < m_lastPing)
				pp->disconnect(PingTimeout);
}

bytes Host::saveNetwork() const
{
	if (!m_nodeTable)
		return bytes();

	std::list<Peer> peers;
	{
		RecursiveGuard l(x_sessions);
		for (auto p: m_peers)
			if (p.second)
				peers.push_back(*p.second);
	}
	peers.sort();

	RLPStream network;
	int count = 0;
	{
		RecursiveGuard l(x_sessions);
		for (auto const& p: peers)
		{
			// TODO: alpha: Figure out why it ever shares these ports.//p.address.port() >= 30300 && p.address.port() <= 30305 &&
			// TODO: alpha: if/how to save private addresses
			// Only save peers which have connected within 2 days, with properly-advertised port and public IP address
			if (chrono::system_clock::now() - p.m_lastConnected < chrono::seconds(3600 * 48) && p.peerEndpoint().port() > 0 && p.peerEndpoint().port() < /*49152*/32768 && p.id != id() && !isPrivateAddress(p.endpoint.udp.address()) && !isPrivateAddress(p.endpoint.tcp.address()))
			{
				network.appendList(10);
				if (p.peerEndpoint().address().is_v4())
					network << p.peerEndpoint().address().to_v4().to_bytes();
				else
					network << p.peerEndpoint().address().to_v6().to_bytes();
				// TODO: alpha: replace 0 with trust-state of node
				network << p.peerEndpoint().port() << p.id << 0
					<< chrono::duration_cast<chrono::seconds>(p.m_lastConnected.time_since_epoch()).count()
					<< chrono::duration_cast<chrono::seconds>(p.m_lastAttempted.time_since_epoch()).count()
					<< p.m_failedAttempts << (unsigned)p.m_lastDisconnect << p.m_score << p.m_rating;
				count++;
			}
		}
	}

	if (!!m_nodeTable)
	{
		auto state = m_nodeTable->snapshot();
		state.sort();
		for (auto const& s: state)
		{
			network.appendList(3);
			if (s.endpoint.tcp.address().is_v4())
				network << s.endpoint.tcp.address().to_v4().to_bytes();
			else
				network << s.endpoint.tcp.address().to_v6().to_bytes();
			network << s.endpoint.tcp.port() << s.id;
			count++;
		}
	}

	RLPStream ret(3);
	ret << dev::p2p::c_protocolVersion << m_alias.secret();
	ret.appendList(count).appendRaw(network.out(), count);
	return ret.out();
}

void Host::restoreNetwork(bytesConstRef _b)
{
	// nodes can only be added if network is added
	if (!isStarted())
		BOOST_THROW_EXCEPTION(NetworkStartRequired());

	RecursiveGuard l(x_sessions);
	RLP r(_b);
	if (r.itemCount() > 0 && r[0].isInt() && r[0].toInt<unsigned>() == dev::p2p::c_protocolVersion)
	{
		// r[0] = version
		// r[1] = key
		// r[2] = nodes

		for (auto i: r[2])
		{
			bi::tcp::endpoint tcp;
			bi::udp::endpoint udp;
			if (i[0].itemCount() == 4)
			{
				tcp = bi::tcp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
				udp = bi::udp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
			}
			else
			{
				tcp = bi::tcp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
				udp = bi::udp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
			}
			
			// skip private addresses
			// todo: to support private addresseses entries must be stored
			//       and managed externally by host rather than nodetable.
			if (isPrivateAddress(tcp.address()) || isPrivateAddress(udp.address()))
				continue;
			
			auto id = (NodeId)i[2];
			if (i.itemCount() == 3)
				m_nodeTable->addNode(id, udp, tcp);
			else if (i.itemCount() == 10)
			{
				shared_ptr<Peer> p = make_shared<Peer>();
				p->id = id;
				p->m_lastConnected = chrono::system_clock::time_point(chrono::seconds(i[4].toInt<unsigned>()));
				p->m_lastAttempted = chrono::system_clock::time_point(chrono::seconds(i[5].toInt<unsigned>()));
				p->m_failedAttempts = i[6].toInt<unsigned>();
				p->m_lastDisconnect = (DisconnectReason)i[7].toInt<unsigned>();
				p->m_score = (int)i[8].toInt<unsigned>();
				p->m_rating = (int)i[9].toInt<unsigned>();
				p->endpoint.tcp = tcp;
				p->endpoint.udp = udp;
				m_peers[p->id] = p;
				m_nodeTable->addNode(*p.get());
			}
		}
	}
}

KeyPair Host::networkAlias(bytesConstRef _b)
{
	RLP r(_b);
	if (r.itemCount() == 3 && r[0].isInt() && r[0].toInt<int>() == 1)
		return move(KeyPair(move(Secret(r[1].toBytes()))));
	else
		return move(KeyPair::create());
}
