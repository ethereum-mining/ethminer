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
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Eric Lombrozo <elombrozo@gmail.com> (Windows version of populateAddresses())
 * @date 2014
 */

#include "Host.h"

#include <sys/types.h>
#ifdef _WIN32
// winsock is already included
// #include <winsock.h>
#else
#include <ifaddrs.h>
#endif

#include <set>
#include <chrono>
#include <thread>
#include <mutex>
#include <boost/algorithm/string.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libethcore/Exceptions.h>
#include "Session.h"
#include "Common.h"
#include "Capability.h"
#include "UPnP.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;

std::vector<bi::address> Host::getInterfaceAddresses()
{
	std::vector<bi::address> addresses;
	
#ifdef _WIN32
	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0)
		BOOST_THROW_EXCEPTION(NoNetworking());
	
	char ac[80];
	if (gethostname(ac, sizeof(ac)) == SOCKET_ERROR)
	{
		clog(NetWarn) << "Error " << WSAGetLastError() << " when getting local host name.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}
	
	struct hostent* phe = gethostbyname(ac);
	if (phe == 0)
	{
		clog(NetWarn) << "Bad host lookup.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}
	
	for (int i = 0; phe->h_addr_list[i] != 0; ++i)
	{
		struct in_addr addr;
		memcpy(&addr, phe->h_addr_list[i], sizeof(struct in_addr));
		char *addrStr = inet_ntoa(addr);
		bi::address address(bi::address::from_string(addrStr));
		if (!isLocalHostAddress(address))
			addresses.push_back(address.to_v4());
	}
	
	WSACleanup();
#else
	ifaddrs* ifaddr;
	if (getifaddrs(&ifaddr) == -1)
		BOOST_THROW_EXCEPTION(NoNetworking());
	
	for (auto ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
	{
		if (!ifa->ifa_addr || string(ifa->ifa_name) == "lo0")
			continue;
		
		if (ifa->ifa_addr->sa_family == AF_INET)
		{
			in_addr addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
			boost::asio::ip::address_v4 address(boost::asio::detail::socket_ops::network_to_host_long(addr.s_addr));
			if (!isLocalHostAddress(address))
				addresses.push_back(address);
		}
		else if (ifa->ifa_addr->sa_family == AF_INET6)
		{
			sockaddr_in6* sockaddr = ((struct sockaddr_in6 *)ifa->ifa_addr);
			in6_addr addr = sockaddr->sin6_addr;
			boost::asio::ip::address_v6::bytes_type bytes;
			memcpy(&bytes[0], addr.s6_addr, 16);
			boost::asio::ip::address_v6 address(bytes, sockaddr->sin6_scope_id);
			if (!isLocalHostAddress(address))
				addresses.push_back(address);
		}
	}
	
	if (ifaddr!=NULL)
		freeifaddrs(ifaddr);
	
#endif
	
	return std::move(addresses);
}

int Host::listen4(bi::tcp::acceptor* _acceptor, unsigned short _listenPort)
{
	int retport = -1;
	for (unsigned i = 0; i < 2; ++i)
	{
		// try to connect w/listenPort, else attempt net-allocated port
		bi::tcp::endpoint endpoint(bi::tcp::v4(), i ? 0 : _listenPort);
		try
		{
			_acceptor->open(endpoint.protocol());
			_acceptor->set_option(ba::socket_base::reuse_address(true));
			_acceptor->bind(endpoint);
			_acceptor->listen();
			retport = _acceptor->local_endpoint().port();
			break;
		}
		catch (...)
		{
			if (i)
			{
				// both attempts failed
				cwarn << "Couldn't start accepting connections on host. Something very wrong with network?\n" << boost::current_exception_diagnostic_information();
			}
			
			// first attempt failed
			_acceptor->close();
			continue;
		}
	}
	return retport;
}

bi::tcp::endpoint Host::traverseNAT(std::vector<bi::address> const& _ifAddresses, unsigned short _listenPort, bi::address& o_upnpifaddr)
{
	asserts(_listenPort != 0);
	
	UPnP* upnp = nullptr;
	try
	{
		upnp = new UPnP;
	}
	// let m_upnp continue as null - we handle it properly.
	catch (NoUPnPDevice) {}
	
	bi::tcp::endpoint upnpep;
	if (upnp && upnp->isValid())
	{
		bi::address paddr;
		int extPort = 0;
		for (auto const& addr: _ifAddresses)
			if (addr.is_v4() && isPrivateAddress(addr) && (extPort = upnp->addRedirect(addr.to_string().c_str(), _listenPort)))
			{
				paddr = addr;
				break;
			}

		auto eip = upnp->externalIP();
		bi::address eipaddr(bi::address::from_string(eip));
		if (extPort && eip != string("0.0.0.0") && !isPrivateAddress(eipaddr))
		{
			clog(NetNote) << "Punched through NAT and mapped local port" << _listenPort << "onto external port" << extPort << ".";
			clog(NetNote) << "External addr:" << eip;
			o_upnpifaddr = paddr;
			upnpep = bi::tcp::endpoint(eipaddr, (unsigned short)extPort);
		}
		else
			clog(NetWarn) << "Couldn't punch through NAT (or no NAT in place).";
		
		if (upnp)
			delete upnp;
	}

	return upnpep;
}

Host::Host(std::string const& _clientVersion, NetworkPreferences const& _n, bool _start):
	Worker("p2p", 0),
	m_clientVersion(_clientVersion),
	m_netPrefs(_n),
	m_ifAddresses(getInterfaceAddresses()),
	m_ioService(new ba::io_service(2)),
	m_acceptor(new bi::tcp::acceptor(*m_ioService)),
	m_socket(new bi::tcp::socket(*m_ioService)),
	m_key(KeyPair::create())
{
	for (auto address: m_ifAddresses)
		if (address.is_v4())
			clog(NetNote) << "IP Address: " << address << " = " << (isPrivateAddress(address) ? "[LOCAL]" : "[PEER]");
	
	clog(NetNote) << "Id:" << id().abridged();
	if (_start)
		start();
}

Host::~Host()
{
	quit();
}

void Host::start()
{
	startWorking();
}

void Host::stop()
{
	{
		// prevent m_run from being set to false at same time as set to true by start()
		Guard l(x_runTimer);
		// once m_run is false the scheduler will shutdown network and stopWorking()
		m_run = false;
	}
	
	// we know shutdown is complete when m_timer is reset
	while (m_timer)
		this_thread::sleep_for(chrono::milliseconds(50));
	stopWorking();
}

void Host::quit()
{
	// called to force io_service to kill any remaining tasks it might have -
	// such tasks may involve socket reads from Capabilities that maintain references
	// to resources we're about to free.
	if (isWorking())
		stop();
	m_acceptor.reset();
	m_socket.reset();
	m_ioService.reset();
	// m_acceptor & m_socket are DANGEROUS now.
}

unsigned Host::protocolVersion() const
{
	return 2;
}

void Host::registerPeer(std::shared_ptr<Session> _s, CapDescs const& _caps)
{
	if (!_s->m_node || !_s->m_node->id)
	{
		cwarn << "Attempting to register a peer without node information!";
		return;
	}

	{
		RecursiveGuard l(x_peers);
		m_peers[_s->m_node->id] = _s;
	}
	unsigned o = (unsigned)UserPacket;
	for (auto const& i: _caps)
		if (haveCapability(i))
		{
			_s->m_capabilities[i] = shared_ptr<Capability>(m_capabilities[i]->newPeerCapability(_s.get(), o));
			o += m_capabilities[i]->messageCount();
		}
}

void Host::seal(bytes& _b)
{
	_b[0] = 0x22;
	_b[1] = 0x40;
	_b[2] = 0x08;
	_b[3] = 0x91;
	uint32_t len = (uint32_t)_b.size() - 8;
	_b[4] = (len >> 24) & 0xff;
	_b[5] = (len >> 16) & 0xff;
	_b[6] = (len >> 8) & 0xff;
	_b[7] = len & 0xff;
}

shared_ptr<Node> Host::noteNode(NodeId _id, bi::tcp::endpoint _a, Origin _o, bool _ready, NodeId _oldId)
{
	RecursiveGuard l(x_peers);
	if (_a.port() < 30300 || _a.port() > 30305)
		cwarn << "Weird port being recorded: " << _a.port();

	if (_a.port() >= /*49152*/32768)
	{
		cwarn << "Private port being recorded - setting to 0";
		_a = bi::tcp::endpoint(_a.address(), 0);
	}

//	cnote << "Node:" << _id.abridged() << _a << (_ready ? "ready" : "used") << _oldId.abridged() << (m_nodes.count(_id) ? "[have]" : "[NEW]");

	// First check for another node with the same connection credentials, and put it in oldId if found.
	if (!_oldId)
		for (pair<h512, shared_ptr<Node>> const& n: m_nodes)
			if (n.second->address == _a && n.second->id != _id)
			{
				_oldId = n.second->id;
				break;
			}

	unsigned i;
	if (!m_nodes.count(_id))
	{
		if (m_nodes.count(_oldId))
		{
			i = m_nodes[_oldId]->index;
			m_nodes.erase(_oldId);
			m_nodesList[i] = _id;
		}
		else
		{
			i = m_nodesList.size();
			m_nodesList.push_back(_id);
		}
		m_nodes[_id] = make_shared<Node>();
		m_nodes[_id]->id = _id;
		m_nodes[_id]->index = i;
		m_nodes[_id]->idOrigin = _o;
	}
	else
	{
		i = m_nodes[_id]->index;
		m_nodes[_id]->idOrigin = max(m_nodes[_id]->idOrigin, _o);
	}
	m_nodes[_id]->address = _a;
	m_ready.extendAll(i);
	m_private.extendAll(i);
	if (_ready)
		m_ready += i;
	else
		m_ready -= i;
	if (!_a.port() || (isPrivateAddress(_a.address()) && !m_netPrefs.localNetworking))
		m_private += i;
	else
		m_private -= i;

//	cnote << m_nodes[_id]->index << ":" << m_ready;

	m_hadNewNodes = true;

	return m_nodes[_id];
}

Nodes Host::potentialPeers(RangeMask<unsigned> const& _known)
{
	RecursiveGuard l(x_peers);
	Nodes ret;

	auto ns = (m_netPrefs.localNetworking ? _known : (m_private + _known)).inverted();
	for (auto i: ns)
		ret.push_back(*m_nodes[m_nodesList[i]]);
	return ret;
}

void Host::determinePublic(string const& _publicAddress, bool _upnp)
{
	m_peerAddresses.clear();
	
	// no point continuing if there are no interface addresses or valid listen port
	if (!m_ifAddresses.size() || m_listenPort < 1)
		return;

	// populate interfaces we'll listen on (eth listens on all interfaces); ignores local
	for (auto addr: m_ifAddresses)
		if ((m_netPrefs.localNetworking || !isPrivateAddress(addr)) && !isLocalHostAddress(addr))
			m_peerAddresses.insert(addr);
	
	// if user supplied address is a public address then we use it
	// if user supplied address is private, and localnetworking is enabled, we use it
	bi::address reqpublicaddr(bi::address(_publicAddress.empty() ? bi::address() : bi::address::from_string(_publicAddress)));
	bi::tcp::endpoint reqpublic(reqpublicaddr, m_listenPort);
	bool isprivate = isPrivateAddress(reqpublicaddr);
	bool ispublic = !isprivate && !isLocalHostAddress(reqpublicaddr);
	if (!reqpublicaddr.is_unspecified() && (ispublic || (isprivate && m_netPrefs.localNetworking)))
	{
		if (!m_peerAddresses.count(reqpublicaddr))
			m_peerAddresses.insert(reqpublicaddr);
		m_public = reqpublic;
		return;
	}
	
	// if address wasn't provided, then use first public ipv4 address found
	for (auto addr: m_peerAddresses)
		if (addr.is_v4() && !isPrivateAddress(addr))
		{
			m_public = bi::tcp::endpoint(*m_peerAddresses.begin(), m_listenPort);
			return;
		}
	
	// or find address via upnp
	if (_upnp)
	{
		bi::address upnpifaddr;
		bi::tcp::endpoint upnpep = traverseNAT(m_ifAddresses, m_listenPort, upnpifaddr);
		if (!upnpep.address().is_unspecified() && !upnpifaddr.is_unspecified())
		{
			if (!m_peerAddresses.count(upnpep.address()))
				m_peerAddresses.insert(upnpep.address());
			m_public = upnpep;
			return;
		}
	}

	// or if no address provided, use private ipv4 address if local networking is enabled
	if (reqpublicaddr.is_unspecified())
		if (m_netPrefs.localNetworking)
			for (auto addr: m_peerAddresses)
				if (addr.is_v4() && isPrivateAddress(addr))
				{
					m_public = bi::tcp::endpoint(addr, m_listenPort);
					return;
				}
	
	// otherwise address is unspecified
	m_public = bi::tcp::endpoint(bi::address(), m_listenPort);
}

void Host::ensureAccepting()
{
	// return if there's no io-server (quit called) or we're not listening
	if (!m_ioService || m_listenPort < 1)
		return;

	if (!m_accepting)
	{
		clog(NetConnect) << "Listening on local port " << m_listenPort << " (public: " << m_public << ")";
		m_accepting = true;
		m_acceptor->async_accept(*m_socket, [=](boost::system::error_code ec)
		{
			if (!ec)
			{
				try
				{
					try {
						clog(NetConnect) << "Accepted connection from " << m_socket->remote_endpoint();
					} catch (...){}
					bi::address remoteAddress = m_socket->remote_endpoint().address();
					// Port defaults to 0 - we let the hello tell us which port the peer listens to
					auto p = std::make_shared<Session>(this, std::move(*m_socket), bi::tcp::endpoint(remoteAddress, 0));
					p->start();
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
			m_accepting = false;
			if (ec.value() < 1)
				ensureAccepting();
		});
	}
}

string Host::pocHost()
{
	vector<string> strs;
	boost::split(strs, dev::Version, boost::is_any_of("."));
	return "poc-" + strs[1] + ".ethdev.com";
}

void Host::connect(std::string const& _addr, unsigned short _port) noexcept
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	for (int i = 0; i < 2; ++i)
	{
		try
		{
			if (i == 0)
			{
				bi::tcp::resolver r(*m_ioService);
				connect(r.resolve({_addr, toString(_port)})->endpoint());
			}
			else
				connect(bi::tcp::endpoint(bi::address::from_string(_addr), _port));
			break;
		}
		catch (Exception const& _e)
		{
			// Couldn't connect
			clog(NetConnect) << "Bad host " << _addr << "\n" << diagnostic_information(_e);
		}
		catch (exception const& e)
		{
			// Couldn't connect
			clog(NetConnect) << "Bad host " << _addr << " (" << e.what() << ")";
		}
	}
}

void Host::connect(bi::tcp::endpoint const& _ep)
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	clog(NetConnect) << "Attempting single-shot connection to " << _ep;
	bi::tcp::socket* s = new bi::tcp::socket(*m_ioService);
	s->async_connect(_ep, [=](boost::system::error_code const& ec)
	{
		if (ec)
			clog(NetConnect) << "Connection refused to " << _ep << " (" << ec.message() << ")";
		else
		{
			auto p = make_shared<Session>(this, std::move(*s), _ep);
			clog(NetConnect) << "Connected to " << _ep;
			p->start();
		}
		delete s;
	});
}

void Host::connect(std::shared_ptr<Node> const& _n)
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;
	
	// prevent concurrently connecting to a node; todo: better abstraction
	Node *nptr = _n.get();
	{
		Guard l(x_pendingNodeConns);
		if (m_pendingNodeConns.count(nptr))
			return;
		m_pendingNodeConns.insert(nptr);
	}
	
	clog(NetConnect) << "Attempting connection to node" << _n->id.abridged() << "@" << _n->address << "from" << id().abridged();
	_n->lastAttempted = std::chrono::system_clock::now();
	_n->failedAttempts++;
	m_ready -= _n->index;
	bi::tcp::socket* s = new bi::tcp::socket(*m_ioService);
	s->async_connect(_n->address, [=](boost::system::error_code const& ec)
	{
		if (ec)
		{
			clog(NetConnect) << "Connection refused to node" << _n->id.abridged() << "@" << _n->address << "(" << ec.message() << ")";
			_n->lastDisconnect = TCPError;
			_n->lastAttempted = std::chrono::system_clock::now();
			m_ready += _n->index;
		}
		else
		{
			clog(NetConnect) << "Connected to" << _n->id.abridged() << "@" << _n->address;
			_n->lastConnected = std::chrono::system_clock::now();
			auto p = make_shared<Session>(this, std::move(*s), node(_n->id), true);		// true because we don't care about ids matched for now. Once we have permenant IDs this will matter a lot more and we can institute a safer mechanism.
			p->start();
		}
		delete s;
		Guard l(x_pendingNodeConns);
		m_pendingNodeConns.erase(nptr);
	});
}

bool Host::havePeer(NodeId _id) const
{
	RecursiveGuard l(x_peers);

	// Remove dead peers from list.
	for (auto i = m_peers.begin(); i != m_peers.end();)
		if (i->second.lock().get())
			++i;
		else
			i = m_peers.erase(i);

	return !!m_peers.count(_id);
}

unsigned Node::fallbackSeconds() const
{
	switch (lastDisconnect)
	{
	case BadProtocol:
		return 30 * (failedAttempts + 1);
	case UselessPeer:
	case TooManyPeers:
	case ClientQuit:
		return 15 * (failedAttempts + 1);
	case NoDisconnect:
		return 0;
	default:
		if (failedAttempts < 5)
			return failedAttempts * 5;
		else if (failedAttempts < 15)
			return 25 + (failedAttempts - 5) * 10;
		else
			return 25 + 100 + (failedAttempts - 15) * 20;
	}
}

bool Node::shouldReconnect() const
{
	return chrono::system_clock::now() > lastAttempted + chrono::seconds(fallbackSeconds());
}

void Host::growPeers()
{
	RecursiveGuard l(x_peers);
	int morePeers = (int)m_idealPeerCount - m_peers.size();
	if (morePeers > 0)
	{
		auto toTry = m_ready;
		if (!m_netPrefs.localNetworking)
			toTry -= m_private;
		set<Node> ns;
		for (auto i: toTry)
			if (m_nodes[m_nodesList[i]]->shouldReconnect())
				ns.insert(*m_nodes[m_nodesList[i]]);

		if (ns.size())
			for (Node const& i: ns)
			{
				connect(m_nodes[i.id]);
				if (!--morePeers)
					return;
			}
		else
			for (auto const& i: m_peers)
				if (auto p = i.second.lock())
					p->ensureNodesRequested();
	}
}

void Host::prunePeers()
{
	RecursiveGuard l(x_peers);
	// We'll keep at most twice as many as is ideal, halfing what counts as "too young to kill" until we get there.
	set<NodeId> dc;
	for (unsigned old = 15000; m_peers.size() - dc.size() > m_idealPeerCount * 2 && old > 100; old /= 2)
		if (m_peers.size() - dc.size() > m_idealPeerCount)
		{
			// look for worst peer to kick off
			// first work out how many are old enough to kick off.
			shared_ptr<Session> worst;
			unsigned agedPeers = 0;
			for (auto i: m_peers)
				if (!dc.count(i.first))
					if (auto p = i.second.lock())
						if (/*(m_mode != NodeMode::Host || p->m_caps != 0x01) &&*/ chrono::steady_clock::now() > p->m_connect + chrono::milliseconds(old))	// don't throw off new peers; peer-servers should never kick off other peer-servers.
						{
							++agedPeers;
							if ((!worst || p->rating() < worst->rating() || (p->rating() == worst->rating() && p->m_connect > worst->m_connect)))	// kill older ones
								worst = p;
						}
			if (!worst || agedPeers <= m_idealPeerCount)
				break;
			dc.insert(worst->id());
			worst->disconnect(TooManyPeers);
		}

	// Remove dead peers from list.
	for (auto i = m_peers.begin(); i != m_peers.end();)
		if (i->second.lock().get())
			++i;
		else
			i = m_peers.erase(i);
}

PeerInfos Host::peers(bool _updatePing) const
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return PeerInfos();

	RecursiveGuard l(x_peers);
    if (_updatePing)
	{
		const_cast<Host*>(this)->pingAll();
		this_thread::sleep_for(chrono::milliseconds(200));
	}
	std::vector<PeerInfo> ret;
	for (auto& i: m_peers)
		if (auto j = i.second.lock())
			if (j->m_socket.is_open())
				ret.push_back(j->m_info);
	return ret;
}

void Host::run(boost::system::error_code const& error)
{
	m_lastTick += c_timerInterval;
	
	if (error || !m_ioService)
	{
		// timer died or io service went away, so stop here
		m_timer.reset();
		return;
	}

	// network running
	if (m_run)
	{
		if (m_lastTick >= c_timerInterval * 10)
		{
			growPeers();
			prunePeers();
			m_lastTick = 0;
		}
		
		if (m_hadNewNodes)
		{
			for (auto p: m_peers)
				if (auto pp = p.second.lock())
					pp->serviceNodesRequest();
			
			m_hadNewNodes = false;
		}
		
		if (chrono::steady_clock::now() - m_lastPing > chrono::seconds(30))	// ping every 30s.
		{
			for (auto p: m_peers)
				if (auto pp = p.second.lock())
					if (chrono::steady_clock::now() - pp->m_lastReceived > chrono::seconds(60))
						pp->disconnect(PingTimeout);
			pingAll();
		}
		
		auto runcb = [this](boost::system::error_code const& error) -> void { run(error); };
		m_timer->expires_from_now(boost::posix_time::milliseconds(c_timerInterval));
		m_timer->async_wait(runcb);
		
		return;
	}
	
	// network stopping
	if (!m_run)
	{
		// close acceptor
		if (m_acceptor->is_open())
		{
			if (m_accepting)
				m_acceptor->cancel();
			m_acceptor->close();
			m_accepting = false;
		}
		
		// stop capabilities (eth: stops syncing or block/tx broadcast)
		for (auto const& h: m_capabilities)
			h.second->onStopping();
		
		// disconnect peers
		for (unsigned n = 0;; n = 0)
		{
			{
				RecursiveGuard l(x_peers);
				for (auto i: m_peers)
					if (auto p = i.second.lock())
						if (p->isOpen())
						{
							p->disconnect(ClientQuit);
							n++;
						}
			}
			if (!n)
				break;
			this_thread::sleep_for(chrono::milliseconds(100));
		}
		
		if (m_socket->is_open())
			m_socket->close();
		
		// m_run is false, so we're stopping; kill timer
		m_lastTick = 0;
		
		// causes parent thread's stop() to continue which calls stopWorking()
		m_timer.reset();
		
		// stop ioservice (stops blocking worker thread, allowing thread to join)
		if (!!m_ioService)
			m_ioService->stop();
		return;
	}
}
			
void Host::startedWorking()
{
	if (!m_timer)
	{
		// no timer means this is first run and network must be started
		// (run once when host worker thread calls startedWorking())
		
		{
			// prevent m_run from being set to true at same time as set to false by stop()
			// don't release mutex until m_timer is set so in case stop() is called at same
			// time, stop will wait on m_timer and graceful network shutdown.
			Guard l(x_runTimer);
			// reset io service and create deadline timer
			m_timer.reset(new boost::asio::deadline_timer(*m_ioService));
			m_run = true;
		}
		m_ioService->reset();
		
		// try to open acceptor (todo: ipv6)
		m_listenPort = listen4(m_acceptor.get(), m_netPrefs.listenPort);
		
		// start capability threads
		for (auto const& h: m_capabilities)
			h.second->onStarting();
		
		// determine public IP, but only if we're able to listen for connections
		// todo: GUI when listen is unavailable in UI
		if (m_listenPort)
		{
			determinePublic(m_netPrefs.publicIP, m_netPrefs.upnp);
			ensureAccepting();
		}
		
		// if m_public address is valid then add us to node list
		// todo: abstract empty() and emplace logic
		if (!m_public.address().is_unspecified() && (m_nodes.empty() || m_nodes[m_nodesList[0]]->id != id()))
			noteNode(id(), m_public, Origin::Perfect, false);
		
		clog(NetNote) << "Id:" << id().abridged();
	}
	
	run(boost::system::error_code());
}

void Host::doWork()
{
	// no ioService means we've had quit() called - bomb out - we're not allowed in here.
	if (asserts(!!m_ioService))
		return;
	m_ioService->run();
}

void Host::pingAll()
{
	RecursiveGuard l(x_peers);
	for (auto& i: m_peers)
		if (auto j = i.second.lock())
			j->ping();
	m_lastPing = chrono::steady_clock::now();
}

bytes Host::saveNodes() const
{
	RLPStream nodes;
	int count = 0;
	{
		RecursiveGuard l(x_peers);
		for (auto const& i: m_nodes)
		{
			Node const& n = *(i.second);
			// TODO: PoC-7: Figure out why it ever shares these ports.//n.address.port() >= 30300 && n.address.port() <= 30305 &&
			if (!n.dead && chrono::system_clock::now() - n.lastConnected < chrono::seconds(3600 * 48) && n.address.port() > 0 && n.address.port() < /*49152*/32768 && n.id != id() && !isPrivateAddress(n.address.address()))
			{
				nodes.appendList(10);
				if (n.address.address().is_v4())
					nodes << n.address.address().to_v4().to_bytes();
				else
					nodes << n.address.address().to_v6().to_bytes();
				nodes << n.address.port() << n.id << (int)n.idOrigin
					<< chrono::duration_cast<chrono::seconds>(n.lastConnected.time_since_epoch()).count()
					<< chrono::duration_cast<chrono::seconds>(n.lastAttempted.time_since_epoch()).count()
					<< n.failedAttempts << (unsigned)n.lastDisconnect << n.score << n.rating;
				count++;
			}
		}
	}
	RLPStream ret(3);
	ret << 0 << m_key.secret();
	ret.appendList(count).appendRaw(nodes.out(), count);
	return ret.out();
}

void Host::restoreNodes(bytesConstRef _b)
{
	RecursiveGuard l(x_peers);
	RLP r(_b);
	if (r.itemCount() > 0 && r[0].isInt())
		switch (r[0].toInt<int>())
		{
		case 0:
		{
			auto oldId = id();
			m_key = KeyPair(r[1].toHash<Secret>());
			noteNode(id(), m_public, Origin::Perfect, false, oldId);

			for (auto i: r[2])
			{
				bi::tcp::endpoint ep;
				if (i[0].itemCount() == 4)
					ep = bi::tcp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
				else
					ep = bi::tcp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
				auto id = (NodeId)i[2];
				if (!m_nodes.count(id))
				{
					auto o = (Origin)i[3].toInt<int>();
					auto n = noteNode(id, ep, o, true);
					n->lastConnected = chrono::system_clock::time_point(chrono::seconds(i[4].toInt<unsigned>()));
					n->lastAttempted = chrono::system_clock::time_point(chrono::seconds(i[5].toInt<unsigned>()));
					n->failedAttempts = i[6].toInt<unsigned>();
					n->lastDisconnect = (DisconnectReason)i[7].toInt<unsigned>();
					n->score = (int)i[8].toInt<unsigned>();
					n->rating = (int)i[9].toInt<unsigned>();
				}
			}
		}
		default:;
		}
	else
		for (auto i: r)
		{
			auto id = (NodeId)i[2];
			if (!m_nodes.count(id))
			{
				bi::tcp::endpoint ep;
				if (i[0].itemCount() == 4)
					ep = bi::tcp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
				else
					ep = bi::tcp::endpoint(bi::address_v6(i[0].toArray<byte, 16>()), i[1].toInt<short>());
				auto n = noteNode(id, ep, Origin::Self, true);
			}
		}
}
