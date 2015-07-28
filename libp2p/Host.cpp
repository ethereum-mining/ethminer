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
#include <libdevcore/FileSystem.h>
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

ReputationManager::ReputationManager()
{
}

void ReputationManager::noteRude(Session const& _s, std::string const& _sub)
{
	DEV_WRITE_GUARDED(x_nodes)
		m_nodes[make_pair(_s.id(), _s.info().clientVersion)].subs[_sub].isRude = true;
}

bool ReputationManager::isRude(Session const& _s, std::string const& _sub) const
{
	DEV_READ_GUARDED(x_nodes)
	{
		auto nit = m_nodes.find(make_pair(_s.id(), _s.info().clientVersion));
		if (nit == m_nodes.end())
			return false;
		auto sit = nit->second.subs.find(_sub);
		bool ret = sit == nit->second.subs.end() ? false : sit->second.isRude;
		return _sub.empty() ? ret : (ret || isRude(_s));
	}
	return false;
}

void ReputationManager::setData(Session const& _s, std::string const& _sub, bytes const& _data)
{
	DEV_WRITE_GUARDED(x_nodes)
		m_nodes[make_pair(_s.id(), _s.info().clientVersion)].subs[_sub].data = _data;
}

bytes ReputationManager::data(Session const& _s, std::string const& _sub) const
{
	DEV_READ_GUARDED(x_nodes)
	{
		auto nit = m_nodes.find(make_pair(_s.id(), _s.info().clientVersion));
		if (nit == m_nodes.end())
			return bytes();
		auto sit = nit->second.subs.find(_sub);
		return sit == nit->second.subs.end() ? bytes() : sit->second.data;
	}
	return bytes();
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
	clog(NetNote) << "Id:" << id();
}

Host::~Host()
{
	stop();
}

void Host::start()
{
	startWorking();
	while (isWorking() && !haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(10));
	
	// network start failed!
	if (isWorking())
		return;

	clog(NetWarn) << "Network start failed!";
	doneWorking();
}

void Host::stop()
{
	// called to force io_service to kill any remaining tasks it might have -
	// such tasks may involve socket reads from Capabilities that maintain references
	// to resources we're about to free.

	{
		// Although m_run is set by stop() or start(), it effects m_runTimer so x_runTimer is used instead of a mutex for m_run.
		Guard l(x_runTimer);
		// ignore if already stopped/stopping
		if (!m_run)
			return;
		
		// signal run() to prepare for shutdown and reset m_timer
		m_run = false;
	}

	// wait for m_timer to reset (indicating network scheduler has stopped)
	while (!!m_timer)
		this_thread::sleep_for(chrono::milliseconds(50));

	// stop worker thread
	if (isWorking())
		stopWorking();
}

void Host::doneWorking()
{
	// reset ioservice (cancels all timers and allows manually polling network, below)
	m_ioService.reset();

	DEV_GUARDED(x_timers)
		m_timers.clear();
	
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
		DEV_GUARDED(x_connecting)
			for (auto const& i: m_connecting)
				if (auto h = i.lock())
				{
					h->cancel();
					n++;
				}
		if (!n)
			break;
		m_ioService.poll();
	}
	
	// disconnect peers
	for (unsigned n = 0;; n = 0)
	{
		DEV_RECURSIVE_GUARDED(x_sessions)
			for (auto i: m_sessions)
				if (auto p = i.second.lock())
					if (p->isConnected())
					{
						p->disconnect(ClientQuit);
						n++;
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

void Host::startPeerSession(Public const& _id, RLP const& _rlp, RLPXFrameCoder* _io, std::shared_ptr<RLPXSocket> const& _s)
{
	// session maybe ingress or egress so m_peers and node table entries may not exist
	shared_ptr<Peer> p;
	DEV_RECURSIVE_GUARDED(x_sessions)
	{
		if (m_peers.count(_id))
			p = m_peers[_id];
		else
		{
			// peer doesn't exist, try to get port info from node table
			if (Node n = m_nodeTable->node(_id))
				p.reset(new Peer(n));
			else
				p.reset(new Peer(Node(_id, UnspecifiedNodeIPEndpoint)));
			m_peers[_id] = p;
		}
	}
	if (p->isOffline())
		p->m_lastConnected = std::chrono::system_clock::now();
	p->endpoint.address = _s->remoteEndpoint().address();

	auto protocolVersion = _rlp[0].toInt<unsigned>();
	auto clientVersion = _rlp[1].toString();
	auto caps = _rlp[2].toVector<CapDesc>();
	auto listenPort = _rlp[3].toInt<unsigned short>();
	
	// clang error (previously: ... << hex << caps ...)
	// "'operator<<' should be declared prior to the call site or in an associated namespace of one of its arguments"
	stringstream capslog;

	if (caps.size() > 1)
		caps.erase(remove_if(caps.begin(), caps.end(), [&](CapDesc const& _r){ return any_of(caps.begin(), caps.end(), [&](CapDesc const& _o){ return _r.first == _o.first && _o.second > _r.second; }); }), caps.end());

	for (auto cap: caps)
		capslog << "(" << cap.first << "," << dec << cap.second << ")";
	clog(NetMessageSummary) << "Hello: " << clientVersion << "V[" << protocolVersion << "]" << _id << showbase << capslog.str() << dec << listenPort;
	
	// create session so disconnects are managed
	auto ps = make_shared<Session>(this, _io, _s, p, PeerSessionInfo({_id, clientVersion, p->endpoint.address.to_string(), listenPort, chrono::steady_clock::duration(), _rlp[2].toSet<CapDesc>(), 0, map<string, string>(), protocolVersion}));
	if (protocolVersion < dev::p2p::c_protocolVersion - 1)
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
					clog(NetWarn) << "Session already exists for peer with id" << _id;
					ps->disconnect(DuplicatePeer);
					return;
				}
		
		if (!peerSlotsAvailable(Ingress))
		{
			ps->disconnect(TooManyPeers);
			return;
		}
		
		// todo: mutex Session::m_capabilities and move for(:caps) out of mutex.
		unsigned o = (unsigned)UserPacket;
		for (auto const& i: caps)
			if (haveCapability(i))
			{
				ps->m_capabilities[i] = shared_ptr<Capability>(m_capabilities[i]->newPeerCapability(ps, o, i));
				o += m_capabilities[i]->messageCount();
			}
		ps->start();
		m_sessions[_id] = ps;
	}
	
	clog(NetP2PNote) << "p2p.host.peer.register" << _id;
	StructuredLogger::p2pConnected(_id.abridged(), ps->m_peer->endpoint, ps->m_peer->m_lastConnected, clientVersion, peerCount());
}

void Host::onNodeTableEvent(NodeId const& _n, NodeTableEventType const& _e)
{
	if (_e == NodeEntryAdded)
	{
		clog(NetP2PNote) << "p2p.host.nodeTable.events.nodeEntryAdded " << _n;
		// only add iff node is in node table
		if (Node n = m_nodeTable->node(_n))
		{
			shared_ptr<Peer> p;
			DEV_RECURSIVE_GUARDED(x_sessions)
			{
				if (m_peers.count(_n))
				{
					p = m_peers[_n];
					p->endpoint = n.endpoint;
				}
				else
				{
					p.reset(new Peer(n));
					m_peers[_n] = p;
					clog(NetP2PNote) << "p2p.host.peers.events.peerAdded " << _n << p->endpoint;
				}
			}
			if (peerSlotsAvailable(Egress))
				connect(p);
		}
	}
	else if (_e == NodeEntryDropped)
	{
		clog(NetP2PNote) << "p2p.host.nodeTable.events.NodeEntryDropped " << _n;
		RecursiveGuard l(x_sessions);
		if (m_peers.count(_n) && !m_peers[_n]->required)
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
		ep = Network::traverseNAT(lset && ifAddresses.count(laddr) ? std::set<bi::address>({laddr}) : ifAddresses, m_netPrefs.listenPort, natIFAddr);
		
		if (lset && natIFAddr != laddr)
			// if listen address is set, Host will use it, even if upnp returns different
			clog(NetWarn) << "Listen address" << laddr << "differs from local address" << natIFAddr << "returned by UPnP!";
		
		if (pset && ep.address() != paddr)
		{
			// if public address is set, Host will advertise it, even if upnp returns different
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
			m_accepting = false;
			if (ec || !m_run)
			{
				socket->close();
				return;
			}
			if (peerCount() > Ingress * m_idealPeerCount)
			{
				clog(NetConnect) << "Dropping incoming connect due to maximum peer count (" << Ingress << " * ideal peer count): " << socket->remoteEndpoint();
				socket->close();
				if (ec.value() < 1)
					runAcceptor();
				return;
			}
			
			bool success = false;
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

			if (!success)
				socket->ref().close();
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

std::unordered_map<Public, std::string> const& Host::pocHosts()
{
	static const std::unordered_map<Public, std::string> c_ret = {
		{ Public("487611428e6c99a11a9795a6abe7b529e81315ca6aad66e2a2fc76e3adf263faba0d35466c2f8f68d561dbefa8878d4df5f1f2ddb1fbeab7f42ffb8cd328bd4a"), "poc-9.ethdev.com:30303" },
		{ Public("a979fb575495b8d6db44f750317d0f4622bf4c2aa3365d6af7c284339968eef29b69ad0dce72a4d8db5ebb4968de0e3bec910127f134779fbcb0cb6d3331163c"), "52.16.188.185:30303" },
		{ Public("7f25d3eab333a6b98a8b5ed68d962bb22c876ffcd5561fca54e3c2ef27f754df6f7fd7c9b74cc919067abac154fb8e1f8385505954f161ae440abc355855e034"), "54.207.93.166:30303" }
	};
	return c_ret;
}

void Host::addNode(NodeId const& _node, NodeIPEndpoint const& _endpoint)
{
	// return if network is stopped while waiting on Host::run() or nodeTable to start
	while (!haveNetwork())
		if (isWorking())
			this_thread::sleep_for(chrono::milliseconds(50));
		else
			return;

	if (_endpoint.tcpPort < 30300 || _endpoint.tcpPort > 30305)
		clog(NetConnect) << "Non-standard port being recorded: " << _endpoint.tcpPort;

	if (m_nodeTable)
		m_nodeTable->addNode(Node(_node, _endpoint));
}

void Host::requirePeer(NodeId const& _n, NodeIPEndpoint const& _endpoint)
{
	if (!m_run)
		return;
	
	Node node(_n, _endpoint, true);
	if (_n)
	{
		// create or update m_peers entry
		shared_ptr<Peer> p;
		DEV_RECURSIVE_GUARDED(x_sessions)
			if (m_peers.count(_n))
			{
				p = m_peers[_n];
				p->endpoint = node.endpoint;
				p->required = true;
			}
			else
			{
				p.reset(new Peer(node));
				m_peers[_n] = p;
			}
	}
	else if (m_nodeTable)
	{
		m_nodeTable->addNode(node);
		shared_ptr<boost::asio::deadline_timer> t(new boost::asio::deadline_timer(m_ioService));
		t->expires_from_now(boost::posix_time::milliseconds(600));
		t->async_wait([this, _n](boost::system::error_code const& _ec)
		{
			if (!_ec)
				if (m_nodeTable)
					if (auto n = m_nodeTable->node(_n))
						requirePeer(n.id, n.endpoint);
		});
		DEV_GUARDED(x_timers)
			m_timers.push_back(t);
	}
}

void Host::relinquishPeer(NodeId const& _node)
{
	Guard l(x_requiredPeers);
	if (m_requiredPeers.count(_node))
		m_requiredPeers.erase(_node);
}

void Host::connect(std::shared_ptr<Peer> const& _p)
{
	if (!m_run)
		return;
	
	if (havePeerSession(_p->id))
	{
		clog(NetConnect) << "Aborted connect. Node already connected.";
		return;
	}

	if (!!m_nodeTable && !m_nodeTable->haveNode(_p->id))
	{
		// connect was attempted, so try again by adding to node table
		m_nodeTable->addNode(*_p.get());
		// abort unless peer is required
		if (!_p->required)
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

	_p->m_lastAttempted = std::chrono::system_clock::now();
	
	bi::tcp::endpoint ep(_p->endpoint);
	clog(NetConnect) << "Attempting connection to node" << _p->id << "@" << ep << "from" << id();
	auto socket = make_shared<RLPXSocket>(new bi::tcp::socket(m_ioService));
	socket->ref().async_connect(ep, [=](boost::system::error_code const& ec)
	{
		_p->m_lastAttempted = std::chrono::system_clock::now();
		_p->m_failedAttempts++;
		
		if (ec)
		{
			clog(NetConnect) << "Connection refused to node" << _p->id << "@" << ep << "(" << ec.message() << ")";
			// Manually set error (session not present)
			_p->m_lastDisconnect = TCPError;
		}
		else
		{
			clog(NetConnect) << "Connecting to" << _p->id << "@" << ep;
			auto handshake = make_shared<RLPXHandshake>(this, socket, _p->id);
			{
				Guard l(x_connecting);
				m_connecting.push_back(handshake);
			}

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
				DEV_GUARDED(j->x_info)
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
	DEV_GUARDED(x_connecting)
		m_connecting.remove_if([](std::weak_ptr<RLPXHandshake> h){ return h.expired(); });
	DEV_GUARDED(x_timers)
		m_timers.remove_if([](std::shared_ptr<boost::asio::deadline_timer> t)
		{
			return t->expires_from_now().total_milliseconds() < 0;
		});

	keepAlivePeers();
	
	// At this time peers will be disconnected based on natural TCP timeout.
	// disconnectLatePeers needs to be updated for the assumption that Session
	// is always live and to ensure reputation and fallback timers are properly
	// updated. // disconnectLatePeers();

	// todo: update peerSlotsAvailable()
	
	list<shared_ptr<Peer>> toConnect;
	unsigned reqConn = 0;
	{
		RecursiveGuard l(x_sessions);
		for (auto const& p: m_peers)
		{
			bool haveSession = havePeerSession(p.second->id);
			bool required = p.second->required;
			if (haveSession && required)
				reqConn++;
			else if (!haveSession && p.second->shouldReconnect() && (!m_netPrefs.pin || required))
				toConnect.push_back(p.second);
		}
	}

	for (auto p: toConnect)
		if (p->required && reqConn++ < m_idealPeerCount)
			connect(p);
	
	if (!m_netPrefs.pin)
	{
		unsigned pendingCount = 0;
		DEV_GUARDED(x_pendingNodeConns)
			pendingCount = m_pendingPeerConns.size();
		int openSlots = m_idealPeerCount - peerCount() - pendingCount + reqConn;
		if (openSlots > 0)
			for (auto p: toConnect)
				if (!p->required && openSlots--)
					connect(p);
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
		clog(NetP2PNote) << "p2p.start.notice id:" << id() << "TCP Listen port is invalid or unavailable.";

	shared_ptr<NodeTable> nodeTable(new NodeTable(m_ioService, m_alias, NodeIPEndpoint(bi::address::from_string(listenAddress()), listenPort(), listenPort()), m_netPrefs.discovery));
	nodeTable->setEventHandler(new HostNodeTableHandler(*this));
	m_nodeTable = nodeTable;
	restoreNetwork(&m_restoreNetwork);

	clog(NetP2PNote) << "p2p.started id:" << id();

	run(boost::system::error_code());
}

void Host::doWork()
{
	try
	{
		if (m_run)
			m_ioService.run();
	}
	catch (std::exception const& _e)
	{
		clog(NetP2PWarn) << "Exception in Network Thread:" << _e.what();
		clog(NetP2PWarn) << "Network Restart is Recommended.";
	}
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
	for (auto const& p: peers)
	{
		// todo: ipv6
		if (!p.endpoint.address.is_v4())
			continue;

		// Only save peers which have connected within 2 days, with properly-advertised port and public IP address
		if (chrono::system_clock::now() - p.m_lastConnected < chrono::seconds(3600 * 48) && !!p.endpoint && p.id != id() && (p.required || p.endpoint.isAllowed()))
		{
			network.appendList(11);
			p.endpoint.streamRLP(network, NodeIPEndpoint::StreamInline);
			network << p.id << p.required
				<< chrono::duration_cast<chrono::seconds>(p.m_lastConnected.time_since_epoch()).count()
				<< chrono::duration_cast<chrono::seconds>(p.m_lastAttempted.time_since_epoch()).count()
				<< p.m_failedAttempts << (unsigned)p.m_lastDisconnect << p.m_score << p.m_rating;
			count++;
		}
	}

	if (!!m_nodeTable)
	{
		auto state = m_nodeTable->snapshot();
		state.sort();
		for (auto const& entry: state)
		{
			network.appendList(4);
			entry.endpoint.streamRLP(network, NodeIPEndpoint::StreamInline);
			network << entry.id;
			count++;
		}
	}
	// else: TODO: use previous configuration if available

	RLPStream ret(3);
	ret << dev::p2p::c_protocolVersion << m_alias.secret().ref();
	ret.appendList(count);
	if (!!count)
		ret.appendRaw(network.out(), count);
	return ret.out();
}

void Host::restoreNetwork(bytesConstRef _b)
{
	if (!_b.size())
		return;
	
	// nodes can only be added if network is added
	if (!isStarted())
		BOOST_THROW_EXCEPTION(NetworkStartRequired());

	if (m_dropPeers)
		return;
	
	RecursiveGuard l(x_sessions);
	RLP r(_b);
	unsigned fileVersion = r[0].toInt<unsigned>();
	if (r.itemCount() > 0 && r[0].isInt() && fileVersion >= dev::p2p::c_protocolVersion - 1)
	{
		// r[0] = version
		// r[1] = key
		// r[2] = nodes

		for (auto i: r[2])
		{
			// todo: ipv6
			if (i[0].itemCount() != 4 && i[0].size() != 4)
				continue;

			if (i.itemCount() == 4 || i.itemCount() == 11)
			{
				Node n((NodeId)i[3], NodeIPEndpoint(i));
				if (i.itemCount() == 4 && n.endpoint.isAllowed())
					m_nodeTable->addNode(n);
				else if (i.itemCount() == 11)
				{
					n.required = i[4].toInt<bool>();
					if (!n.endpoint.isAllowed() && !n.required)
						continue;
					shared_ptr<Peer> p = make_shared<Peer>(n);
					p->m_lastConnected = chrono::system_clock::time_point(chrono::seconds(i[5].toInt<unsigned>()));
					p->m_lastAttempted = chrono::system_clock::time_point(chrono::seconds(i[6].toInt<unsigned>()));
					p->m_failedAttempts = i[7].toInt<unsigned>();
					p->m_lastDisconnect = (DisconnectReason)i[8].toInt<unsigned>();
					p->m_score = (int)i[9].toInt<unsigned>();
					p->m_rating = (int)i[10].toInt<unsigned>();
					m_peers[p->id] = p;
					if (p->required)
						requirePeer(p->id, n.endpoint);
					else
						m_nodeTable->addNode(*p.get(), NodeTable::NodeRelation::Known);
				}
			}
			else if (i.itemCount() == 3 || i.itemCount() == 10)
			{
				Node n((NodeId)i[2], NodeIPEndpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<uint16_t>(), i[1].toInt<uint16_t>()));
				if (i.itemCount() == 3 && n.endpoint.isAllowed())
					m_nodeTable->addNode(n);
				else if (i.itemCount() == 10)
				{
					n.required = i[3].toInt<bool>();
					if (!n.endpoint.isAllowed() && !n.required)
						continue;
					shared_ptr<Peer> p = make_shared<Peer>(n);
					p->m_lastConnected = chrono::system_clock::time_point(chrono::seconds(i[4].toInt<unsigned>()));
					p->m_lastAttempted = chrono::system_clock::time_point(chrono::seconds(i[5].toInt<unsigned>()));
					p->m_failedAttempts = i[6].toInt<unsigned>();
					p->m_lastDisconnect = (DisconnectReason)i[7].toInt<unsigned>();
					p->m_score = (int)i[8].toInt<unsigned>();
					p->m_rating = (int)i[9].toInt<unsigned>();
					m_peers[p->id] = p;
					if (p->required)
						requirePeer(p->id, n.endpoint);
					else
						m_nodeTable->addNode(*p.get(), NodeTable::NodeRelation::Known);
				}
			}
		}
	}
}

KeyPair Host::networkAlias(bytesConstRef _b)
{
	RLP r(_b);
	if (r.itemCount() == 3 && r[0].isInt() && r[0].toInt<unsigned>() >= 3)
		return KeyPair(Secret(r[1].toBytes()));
	else
		return KeyPair::create();
}
