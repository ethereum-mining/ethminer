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

// Addresses we will skip during network interface discovery
// Use a vector as the list is small
// Why this and not names?
// Under MacOSX loopback (127.0.0.1) can be named lo0 and br0 are bridges (0.0.0.0)
static const set<bi::address> c_rejectAddresses = {
	{bi::address_v4::from_string("127.0.0.1")},
	{bi::address_v6::from_string("::1")},
	{bi::address_v4::from_string("0.0.0.0")},
	{bi::address_v6::from_string("::")}
};

Host::Host(std::string const& _clientVersion, NetworkPreferences const& _n, bool _start):
	Worker("p2p"),
	m_clientVersion(_clientVersion),
	m_netPrefs(_n),
	m_ioService(new ba::io_service),
	m_acceptor(*m_ioService),
	m_socket(*m_ioService),
	m_key(KeyPair::create())
{
	populateAddresses();
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
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	if (isWorking())
		stop();

	for (unsigned i = 0; i < 2; ++i)
	{
		bi::tcp::endpoint endpoint(bi::tcp::v4(), i ? 0 : m_netPrefs.listenPort);
		try
		{
			m_acceptor.open(endpoint.protocol());
			m_acceptor.set_option(ba::socket_base::reuse_address(true));
			m_acceptor.bind(endpoint);
			m_acceptor.listen();
			m_listenPort = i ? m_acceptor.local_endpoint().port() : m_netPrefs.listenPort;
			break;
		}
		catch (...)
		{
			if (i)
			{
				cwarn << "Couldn't start accepting connections on host. Something very wrong with network?\n" << boost::current_exception_diagnostic_information();
				return;
			}
			m_acceptor.close();
			continue;
		}
	}

	for (auto const& h: m_capabilities)
		h.second->onStarting();

	startWorking();
}

void Host::stop()
{
	for (auto const& h: m_capabilities)
		h.second->onStopping();

	stopWorking();

	if (m_acceptor.is_open())
	{
		if (m_accepting)
			m_acceptor.cancel();
		m_acceptor.close();
		m_accepting = false;
	}
	if (m_socket.is_open())
		m_socket.close();
	disconnectPeers();

	if (!!m_ioService)
	{
		m_ioService->stop();
		m_ioService->reset();
	}
}

void Host::quit()
{
	stop();
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

void Host::disconnectPeers()
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	for (unsigned n = 0;; n = 0)
	{
		{
			RecursiveGuard l(x_peers);
			for (auto i: m_peers)
				if (auto p = i.second.lock())
				{
					p->disconnect(ClientQuit);
					n++;
				}
		}
		if (!n)
			break;
		m_ioService->poll();
		this_thread::sleep_for(chrono::milliseconds(100));
	}

	delete m_upnp;
	m_upnp = nullptr;
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

void Host::determinePublic(string const& _publicAddress, bool _upnp)
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	if (_upnp)
		try
		{
			m_upnp = new UPnP;
		}
		catch (NoUPnPDevice) {}	// let m_upnp continue as null - we handle it properly.

	bi::tcp::resolver r(*m_ioService);
	if (m_upnp && m_upnp->isValid() && m_peerAddresses.size())
	{
		clog(NetNote) << "External addr:" << m_upnp->externalIP();
		int p;
		for (auto const& addr : m_peerAddresses)
			if ((p = m_upnp->addRedirect(addr.to_string().c_str(), m_listenPort)))
				break;
		if (p)
			clog(NetNote) << "Punched through NAT and mapped local port" << m_listenPort << "onto external port" << p << ".";
		else
		{
			// couldn't map
			clog(NetWarn) << "Couldn't punch through NAT (or no NAT in place). Assuming" << m_listenPort << "is local & external port.";
			p = m_listenPort;
		}

		auto eip = m_upnp->externalIP();
		if (eip == string("0.0.0.0") && _publicAddress.empty())
			m_public = bi::tcp::endpoint(bi::address(), (unsigned short)p);
		else
		{
			m_public = bi::tcp::endpoint(bi::address::from_string(_publicAddress.empty() ? eip : _publicAddress), (unsigned short)p);
			m_addresses.push_back(m_public.address());
		}
	}
	else
	{
		// No UPnP - fallback on given public address or, if empty, the assumed peer address.
		m_public = bi::tcp::endpoint(_publicAddress.size() ? bi::address::from_string(_publicAddress)
									: m_peerAddresses.size() ? m_peerAddresses[0]
									: bi::address(), m_listenPort);
		m_addresses.push_back(m_public.address());
	}
}

void Host::populateAddresses()
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

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
		bi::address ad(bi::address::from_string(addrStr));
		m_addresses.push_back(ad.to_v4());
		bool isLocal = std::find(c_rejectAddresses.begin(), c_rejectAddresses.end(), ad) != c_rejectAddresses.end();
		if (!isLocal)
			m_peerAddresses.push_back(ad.to_v4());
		clog(NetNote) << "Address: " << ac << " = " << m_addresses.back() << (isLocal ? " [LOCAL]" : " [PEER]");
	}

	WSACleanup();
#else
	ifaddrs* ifaddr;
	if (getifaddrs(&ifaddr) == -1)
		BOOST_THROW_EXCEPTION(NoNetworking());

	bi::tcp::resolver r(*m_ioService);

	for (ifaddrs* ifa = ifaddr; ifa; ifa = ifa->ifa_next)
	{
		if (!ifa->ifa_addr)
			continue;
		if (ifa->ifa_addr->sa_family == AF_INET)
		{
			char host[NI_MAXHOST];
			if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST))
				continue;
			try
			{
				auto it = r.resolve({host, "30303"});
				bi::tcp::endpoint ep = it->endpoint();
				bi::address ad = ep.address();
				m_addresses.push_back(ad.to_v4());
				bool isLocal = std::find(c_rejectAddresses.begin(), c_rejectAddresses.end(), ad) != c_rejectAddresses.end();
				if (!isLocal)
					m_peerAddresses.push_back(ad.to_v4());
				clog(NetNote) << "Address: " << host << " = " << m_addresses.back() << (isLocal ? " [LOCAL]" : " [PEER]");
			}
			catch (...)
			{
				clog(NetNote) << "Couldn't resolve: " << host;
			}
		}
		else if (ifa->ifa_addr->sa_family == AF_INET6)
		{
			char host[NI_MAXHOST];
			if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in6), host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST))
				continue;
			try
			{
				auto it = r.resolve({host, "30303"});
				bi::tcp::endpoint ep = it->endpoint();
				bi::address ad = ep.address();
				m_addresses.push_back(ad.to_v6());
				bool isLocal = std::find(c_rejectAddresses.begin(), c_rejectAddresses.end(), ad) != c_rejectAddresses.end();
				if (!isLocal)
					m_peerAddresses.push_back(ad);
				clog(NetNote) << "Address: " << host << " = " << m_addresses.back() << (isLocal ? " [LOCAL]" : " [PEER]");
			}
			catch (...)
			{
				clog(NetNote) << "Couldn't resolve: " << host;
			}
		}
	}

	freeifaddrs(ifaddr);
#endif
}

shared_ptr<Node> Host::noteNode(NodeId _id, bi::tcp::endpoint _a, Origin _o, bool _ready, NodeId _oldId)
{
	RecursiveGuard l(x_peers);
	if (_a.port() < 30300 && _a.port() > 30303)
		cwarn << "Wierd port being recorded!";

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

void Host::ensureAccepting()
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (!m_ioService)
		return;

	if (!m_accepting)
	{
		clog(NetConnect) << "Listening on local port " << m_listenPort << " (public: " << m_public << ")";
		m_accepting = true;
		m_acceptor.async_accept(m_socket, [=](boost::system::error_code ec)
		{
			if (!ec)
			{
				try
				{
					try {
						clog(NetConnect) << "Accepted connection from " << m_socket.remote_endpoint();
					} catch (...){}
					bi::address remoteAddress = m_socket.remote_endpoint().address();
					// Port defaults to 0 - we let the hello tell us which port the peer listens to
					auto p = std::make_shared<Session>(this, std::move(m_socket), bi::tcp::endpoint(remoteAddress, 0));
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
		{
			ensureAccepting();
			for (auto const& i: m_peers)
				if (auto p = i.second.lock())
					p->ensureNodesRequested();
		}
	}
}

void Host::prunePeers()
{
	RecursiveGuard l(x_peers);
	// We'll keep at most twice as many as is ideal, halfing what counts as "too young to kill" until we get there.
	for (unsigned old = 15000; m_peers.size() > m_idealPeerCount * 2 && old > 100; old /= 2)
		while (m_peers.size() > m_idealPeerCount)
		{
			// look for worst peer to kick off
			// first work out how many are old enough to kick off.
			shared_ptr<Session> worst;
			unsigned agedPeers = 0;
			for (auto i: m_peers)
				if (auto p = i.second.lock())
					if (/*(m_mode != NodeMode::Host || p->m_caps != 0x01) &&*/ chrono::steady_clock::now() > p->m_connect + chrono::milliseconds(old))	// don't throw off new peers; peer-servers should never kick off other peer-servers.
					{
						++agedPeers;
						if ((!worst || p->rating() < worst->rating() || (p->rating() == worst->rating() && p->m_connect > worst->m_connect)))	// kill older ones
							worst = p;
					}
			if (!worst || agedPeers <= m_idealPeerCount)
				break;
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

void Host::startedWorking()
{
	determinePublic(m_netPrefs.publicIP, m_netPrefs.upnp);
	ensureAccepting();

	if (!m_public.address().is_unspecified() && (m_nodes.empty() || m_nodes[m_nodesList[0]]->id != id()))
		noteNode(id(), m_public, Origin::Perfect, false);

	clog(NetNote) << "Id:" << id().abridged();
}

void Host::doWork()
{
	// if there's no ioService, it means we've had quit() called - bomb out - we're not allowed in here.
	if (asserts(!!m_ioService))
		return;

	growPeers();
	prunePeers();

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

	m_ioService->poll();
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
			if (!n.dead && n.address.port() > 0 && n.address.port() < /*49152*/32768 && n.id != id() && !isPrivateAddress(n.address.address()))
			{
				nodes.appendList(10);
				if (n.address.address().is_v4())
					nodes << n.address.address().to_v4().to_bytes();
				else
					nodes << n.address.address().to_v6().to_bytes();
				nodes << n.address.port() << n.id << (int)n.idOrigin
					<< std::chrono::duration_cast<std::chrono::seconds>(n.lastConnected.time_since_epoch()).count()
					<< std::chrono::duration_cast<std::chrono::seconds>(n.lastAttempted.time_since_epoch()).count()
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
