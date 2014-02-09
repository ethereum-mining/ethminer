/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file PeerNetwork.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Eric Lombrozo <elombrozo@gmail.com>
 * @date 2014
 */

#include <sys/types.h>
#ifdef _WIN32
// winsock is already included
// #include <winsock.h>
#else
#include <ifaddrs.h>
#endif

#include <chrono>
#include <thread>
#include "Exceptions.h"
#include "Common.h"
#include "BlockChain.h"
#include "BlockInfo.h"
#include "TransactionQueue.h"
#include "UPnP.h"
#include "PeerNetwork.h"
using namespace std;
using namespace eth;

#define clogS(X) eth::LogOutputStream<X, true>(false) << "| " << std::setw(2) << m_socket.native_handle() << "] "

static const eth::uint c_maxHashes = 256;		///< Maximum number of hashes GetChain will ever send.
static const eth::uint c_maxBlocks = 128;		///< Maximum number of blocks Blocks will ever send. BUG: if this gets too big (e.g. 2048) stuff starts going wrong.
static const eth::uint c_maxBlocksAsk = 2048;	///< Maximum number of blocks we ask to receive in Blocks (when using GetChain).

// Addresses we will skip during network interface discovery
// Use a vector as the list is small
// Why this and not names?
// Under MacOSX loopback (127.0.0.1) can be named lo0 and br0 are bridges (0.0.0.0)
static const vector<bi::address> c_rejectAddresses = {
	{bi::address_v4::from_string("127.0.0.1")},
	{bi::address_v6::from_string("::1")},
	{bi::address_v4::from_string("0.0.0.0")},
	{bi::address_v6::from_string("::")}
};

PeerSession::PeerSession(PeerServer* _s, bi::tcp::socket _socket, uint _rNId):
	m_server(_s),
	m_socket(std::move(_socket)),
	m_reqNetworkId(_rNId),
	m_rating(0)
{
	m_disconnect = std::chrono::steady_clock::time_point::max();
	m_connect = std::chrono::steady_clock::now();
}

PeerSession::~PeerSession()
{
	m_socket.close();
}

bi::tcp::endpoint PeerSession::endpoint() const
{
	if (m_socket.is_open())
		try {
			return bi::tcp::endpoint(m_socket.remote_endpoint().address(), m_listenPort);
		} catch (...){}

	return bi::tcp::endpoint();
}

// TODO: BUG! 256 -> work out why things start to break with big packet sizes -> g.t. ~370 blocks.

bool PeerSession::interpret(RLP const& _r)
{
	clogS(NetRight) << _r;
	switch (_r[0].toInt<unsigned>())
	{
	case HelloPacket:
	{
		m_protocolVersion = _r[1].toInt<uint>();
		m_networkId = _r[2].toInt<uint>();
		auto clientVersion = _r[3].toString();
		m_caps = _r[4].toInt<uint>();
		m_listenPort = _r[5].toInt<short>();
		m_id = _r[6].toHash<h512>();

		clogS(NetMessageSummary) << "Hello: " << clientVersion << "V[" << m_protocolVersion << "/" << m_networkId << "]" << asHex(m_id.ref().cropped(0, 4)) << showbase << hex << m_caps << dec << m_listenPort;

		if (m_server->m_peers.count(m_id) || !m_id)
		{
			// Already connected.
			disconnect(DuplicatePeer);
		}
		m_server->m_peers[m_id] = shared_from_this();

		if (m_protocolVersion != 2 || m_networkId != m_reqNetworkId)
		{
			disconnect(IncompatibleProtocol);
			return false;
		}
		try
			{ m_info = PeerInfo({clientVersion, m_socket.remote_endpoint().address().to_string(), (short)m_socket.remote_endpoint().port(), std::chrono::steady_clock::duration()}); }
		catch (...)
		{
			disconnect(BadProtocol);
			return false;
		}

		// Grab their block chain off them.
		{
			unsigned count = std::min<unsigned>(c_maxHashes, m_server->m_chain->details(m_server->m_latestBlockSent).number + 1);
			RLPStream s;
			prep(s).appendList(2 + count);
			s << GetChainPacket;
			auto h = m_server->m_latestBlockSent;
			for (unsigned i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
				s << h;
			s << c_maxBlocksAsk;
			sealAndSend(s);
			s.clear();
			prep(s).appendList(1);
			s << GetTransactionsPacket;
			sealAndSend(s);
		}
		break;
	}
	case DisconnectPacket:
	{
		string reason = "Unspecified";
		if (_r.itemCount() > 1 && _r[1].isInt())
			switch (_r[1].toInt<int>())
			{
			case DisconnectRequested: reason = "Disconnect was requested."; break;
			case TCPError: reason = "Low-level TCP communication error."; break;
			case BadProtocol: reason = "Data format error."; break;
			case UselessPeer: reason = "We had no use to peer."; break;
			case TooManyPeers: reason = "Peer had too many connections."; break;
			case DuplicatePeer: reason = "Peer was already connected."; break;
			case WrongGenesis: reason = "Disagreement over genesis block."; break;
			case IncompatibleProtocol: reason = "Peer protocol versions are incompatible."; break;
			case ClientQuit: reason = "Peer is exiting."; break;
			}

		clogS(NetMessageSummary) << "Disconnect (reason: " << reason << ")";
		if (m_socket.is_open())
			clogS(NetNote) << "Closing " << m_socket.remote_endpoint();
		else
			clogS(NetNote) << "Remote closed.";
		m_socket.close();
		return false;
	}
	case PingPacket:
	{
//		clogS(NetMessageSummary) << "Ping";
		RLPStream s;
		sealAndSend(prep(s).appendList(1) << PongPacket);
		break;
	}
	case PongPacket:
		m_info.lastPing = std::chrono::steady_clock::now() - m_ping;
//		clogS(NetMessageSummary) << "Latency: " << chrono::duration_cast<chrono::milliseconds>(m_lastPing).count() << " ms";
		break;
	case GetPeersPacket:
	{
		clogS(NetMessageSummary) << "GetPeers";
		auto peers = m_server->potentialPeers();
		RLPStream s;
		prep(s).appendList(peers.size() + 1);
		s << PeersPacket;
		for (auto i: peers)
		{
			clogS(NetMessageDetail) << "Sending peer " << asHex(i.first.ref().cropped(0, 4)) << i.second;
			s.appendList(3) << i.second.address().to_v4().to_bytes() << i.second.port() << i.first;
		}
		sealAndSend(s);
		break;
	}
	case PeersPacket:
		clogS(NetMessageSummary) << "Peers (" << dec << (_r.itemCount() - 1) << " entries)";
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto ep = bi::tcp::endpoint(bi::address_v4(_r[i][0].toArray<byte, 4>()), _r[i][1].toInt<short>());
			Public id;
			if (_r[i].itemCount() > 2)
				id = _r[i][2].toHash<Public>();

			clogS(NetAllDetail) << "Checking: " << ep << "(" << asHex(id.ref().cropped(0, 4)) << ")";

			// check that it's not us or one we already know:
			if (id && (m_server->m_key.pub() == id || m_server->m_peers.count(id) || m_server->m_incomingPeers.count(id)))
				goto CONTINUE;

			// check that we're not already connected to addr:
			if (!ep.port())
				goto CONTINUE;
			for (auto i: m_server->m_addresses)
				if (ep.address() == i && ep.port() == m_server->listenPort())
					goto CONTINUE;
			for (auto i: m_server->m_peers)
				if (shared_ptr<PeerSession> p = i.second.lock())
				{
					clogS(NetAllDetail) << "   ...against " << p->endpoint();
					if (p->m_socket.is_open() && p->endpoint() == ep)
						goto CONTINUE;
				}
			for (auto i: m_server->m_incomingPeers)
				if (i.second == ep)
					goto CONTINUE;
			m_server->m_incomingPeers.insert(make_pair(id, ep));
			clogS(NetMessageDetail) << "New peer: " << ep;
			CONTINUE:;
		}
		break;
	case TransactionsPacket:
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		clogS(NetMessageSummary) << "Transactions (" << dec << (_r.itemCount() - 1) << " entries)";
		m_rating += _r.itemCount() - 1;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			m_server->m_incomingTransactions.push_back(_r[i].data().toBytes());
			m_knownTransactions.insert(sha3(_r[i].data()));
		}
		break;
	case BlocksPacket:
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		clogS(NetMessageSummary) << "Blocks (" << dec << (_r.itemCount() - 1) << " entries)";
		m_rating += _r.itemCount() - 1;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			m_server->m_incomingBlocks.push_back(_r[i].data().toBytes());
			m_knownBlocks.insert(sha3(_r[i].data()));
		}
		if (g_logVerbosity >= 3)
			for (unsigned i = 1; i < _r.itemCount(); ++i)
			{
				auto h = sha3(_r[i].data());
				BlockInfo bi(_r[i].data());
				if (!m_server->m_chain->details(bi.parentHash) && !m_knownBlocks.count(bi.parentHash))
					clogS(NetMessageDetail) << "Unknown parent " << bi.parentHash << " of block " << h;
				else
					clogS(NetMessageDetail) << "Known parent " << bi.parentHash << " of block " << h;
			}
		if (_r.itemCount() > 1)	// we received some - check if there's any more
		{
			RLPStream s;
			prep(s).appendList(3);
			s << GetChainPacket;
			s << sha3(_r[1].data());
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		break;
	case GetChainPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		// ********************************************************************
		// NEEDS FULL REWRITE!
		h256s parents;
		parents.reserve(_r.itemCount() - 2);
		for (unsigned i = 1; i < _r.itemCount() - 1; ++i)
			parents.push_back(_r[i].toHash<h256>());
		clogS(NetMessageSummary) << "GetChain (" << (_r.itemCount() - 2) << " hashes, " << (_r[_r.itemCount() - 1].toInt<bigint>()) << ")";
		if (_r.itemCount() == 2)
			break;
		// return 2048 block max.
		uint baseCount = (uint)min<bigint>(_r[_r.itemCount() - 1].toInt<bigint>(), c_maxBlocks);
		clogS(NetMessageSummary) << "GetChain (" << baseCount << " max, from " << parents.front() << " to " << parents.back() << ")";
		for (auto parent: parents)
		{
			auto h = m_server->m_chain->currentHash();
			h256 latest = m_server->m_chain->currentHash();
			uint latestNumber = 0;
			uint parentNumber = 0;
			RLPStream s;

			if (m_server->m_chain->details(parent))
			{
				latestNumber = m_server->m_chain->details(latest).number;
				parentNumber = m_server->m_chain->details(parent).number;
				uint count = min<uint>(latestNumber - parentNumber, baseCount);
				clogS(NetAllDetail) << "Requires " << dec << (latestNumber - parentNumber) << " blocks from " << latestNumber << " to " << parentNumber;
				clogS(NetAllDetail) << latest << " - " << parent;

				prep(s);
				s.appendList(1 + count) << BlocksPacket;
				uint endNumber = m_server->m_chain->details(parent).number;
				uint startNumber = endNumber + count;
				clogS(NetAllDetail) << "Sending " << dec << count << " blocks from " << startNumber << " to " << endNumber;

				uint n = latestNumber;
				for (; n > startNumber; n--, h = m_server->m_chain->details(h).parent) {}
				for (uint i = 0; h != parent && n > endNumber && i < count; ++i, --n, h = m_server->m_chain->details(h).parent)
				{
					clogS(NetAllDetail) << "   " << dec << i << " " << h;
					s.appendRaw(m_server->m_chain->block(h));
				}
				clogS(NetAllDetail) << "Parent: " << h;
			}
			else if (parent != parents.back())
				continue;

			if (h != parent)
			{
				// not in the blockchain;
				if (parent == parents.back())
				{
					// out of parents...
					clogS(NetAllDetail) << "GetChain failed; not in chain";
					// No good - must have been on a different branch.
					s.clear();
					prep(s).appendList(2) << NotInChainPacket << parents.back();
				}
				else
					// still some parents left - try them.
					continue;
			}
			// send the packet (either Blocks or NotInChain) & exit.
			sealAndSend(s);
			break;
			// ********************************************************************
		}
		break;
	}
	case NotInChainPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		h256 noGood = _r[1].toHash<h256>();
		clogS(NetMessageSummary) << "NotInChain (" << noGood << ")";
		if (noGood == m_server->m_chain->genesisHash())
		{
			clogS(NetWarn) << "Discordance over genesis block! Disconnect.";
			disconnect(WrongGenesis);
		}
		else
		{
			unsigned count = std::min<unsigned>(c_maxHashes, m_server->m_chain->details(noGood).number);
			RLPStream s;
			prep(s).appendList(2 + count);
			s << GetChainPacket;
			auto h = m_server->m_chain->details(noGood).parent;
			for (unsigned i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
				s << h;
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		break;
	}
	case GetTransactionsPacket:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		m_requireTransactions = true;
		break;
	}
	default:
		break;
	}
	return true;
}

void PeerSession::ping()
{
	RLPStream s;
	sealAndSend(prep(s).appendList(1) << PingPacket);
	m_ping = std::chrono::steady_clock::now();
}

RLPStream& PeerSession::prep(RLPStream& _s)
{
	return _s.appendRaw(bytes(8, 0));
}

void PeerServer::seal(bytes& _b)
{
	_b[0] = 0x22;
	_b[1] = 0x40;
	_b[2] = 0x08;
	_b[3] = 0x91;
	uint32_t len = _b.size() - 8;
	_b[4] = (len >> 24) & 0xff;
	_b[5] = (len >> 16) & 0xff;
	_b[6] = (len >> 8) & 0xff;
	_b[7] = len & 0xff;
}

void PeerSession::sealAndSend(RLPStream& _s)
{
	bytes b;
	_s.swapOut(b);
	m_server->seal(b);
	sendDestroy(b);
}

void PeerSession::sendDestroy(bytes& _msg)
{
	clogS(NetLeft) << RLP(bytesConstRef(&_msg).cropped(8));
	std::shared_ptr<bytes> buffer = std::make_shared<bytes>();
	swap(*buffer, _msg);
	assert((*buffer)[0] == 0x22);
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			dropped();
//		cbug << length << " bytes written (EC: " << ec << ")";
	});
}

void PeerSession::send(bytesConstRef _msg)
{
	clogS(NetLeft) << RLP(_msg.cropped(8));
	std::shared_ptr<bytes> buffer = std::make_shared<bytes>(_msg.toBytes());
	assert((*buffer)[0] == 0x22);
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			dropped();
//		cbug << length << " bytes written (EC: " << ec << ")";
	});
}

void PeerSession::dropped()
{
	if (m_socket.is_open())
		try {
			clogS(NetNote) << "Closing " << m_socket.remote_endpoint();
		}catch (...){}
	m_socket.close();
	for (auto i = m_server->m_peers.begin(); i != m_server->m_peers.end(); ++i)
		if (i->second.lock().get() == this)
		{
			m_server->m_peers.erase(i);
			break;
		}
}

void PeerSession::disconnect(int _reason)
{
	if (m_socket.is_open())
	{
		if (m_disconnect == chrono::steady_clock::time_point::max())
		{
			RLPStream s;
			prep(s);
			s.appendList(1) << DisconnectPacket << _reason;
			sealAndSend(s);
			m_disconnect = chrono::steady_clock::now();
		}
		else
		{
			if (m_socket.is_open())
				try {
				clogS(NetNote) << "Closing " << m_socket.remote_endpoint();
				} catch (...){}
			else
				clogS(NetNote) << "Remote closed on" << m_socket.native_handle();
			m_socket.close();
		}
	}
}

void PeerSession::start()
{
	RLPStream s;
	prep(s);
	s.appendList(m_server->m_public.port() ? 6 : 5) << HelloPacket << (uint)1 << (uint)m_server->m_requiredNetworkId << m_server->m_clientVersion << (m_server->m_mode == NodeMode::Full ? 0x07 : m_server->m_mode == NodeMode::PeerServer ? 0x01 : 0) << m_server->m_public.port() << m_server->m_key.pub();
	sealAndSend(s);

	ping();

	doRead();
}

void PeerSession::doRead()
{
	auto self(shared_from_this());
	m_socket.async_read_some(boost::asio::buffer(m_data), [this, self](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			dropped();
		else
		{
			try
			{
				m_incoming.resize(m_incoming.size() + length);
				memcpy(m_incoming.data() + m_incoming.size() - length, m_data.data(), length);
				while (m_incoming.size() > 8)
				{
					if (m_incoming[0] != 0x22 || m_incoming[1] != 0x40 || m_incoming[2] != 0x08 || m_incoming[3] != 0x91)
					{
						clogS(NetWarn) << "Out of alignment. Skipping: " << hex << showbase << (int)m_incoming[0] << dec;
						memmove(m_incoming.data(), m_incoming.data() + 1, m_incoming.size() - 1);
						m_incoming.resize(m_incoming.size() - 1);
					}
					else
					{
						uint32_t len = fromBigEndian<uint32_t>(bytesConstRef(m_incoming.data() + 4, 4));
						if (m_incoming.size() - 8 < len)
							break;

						// enough has come in.
//						cerr << "Received " << len << ": " << asHex(bytesConstRef(m_incoming.data() + 8, len)) << endl;
						RLP r(bytesConstRef(m_incoming.data() + 8, len));
						if (!interpret(r))
							// error
							break;
						memmove(m_incoming.data(), m_incoming.data() + len + 8, m_incoming.size() - (len + 8));
						m_incoming.resize(m_incoming.size() - (len + 8));
					}
				}
				doRead();
			}
			catch (Exception const& _e)
			{
				clogS(NetWarn) << "ERROR: " << _e.description();
				dropped();
			}
			catch (std::exception const& _e)
			{
				clogS(NetWarn) << "ERROR: " << _e.what();
				dropped();
			}
		}
	});
}

PeerServer::PeerServer(std::string const& _clientVersion, BlockChain const& _ch, uint _networkId, short _port, NodeMode _m, string const& _publicAddress, bool _upnp):
	m_clientVersion(_clientVersion),
	m_mode(_m),
	m_listenPort(_port),
	m_chain(&_ch),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), _port)),
	m_socket(m_ioService),
	m_key(KeyPair::create()),
	m_requiredNetworkId(_networkId)
{
	populateAddresses();
	determinePublic(_publicAddress, _upnp);
	ensureAccepting();
	clog(NetNote) << "Id:" << asHex(m_key.address().ref().cropped(0, 4)) << "Mode: " << (_m == NodeMode::PeerServer ? "PeerServer" : "Full");
}

PeerServer::PeerServer(std::string const& _clientVersion, uint _networkId, NodeMode _m):
	m_clientVersion(_clientVersion),
	m_mode(_m),
	m_listenPort(-1),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), 0)),
	m_socket(m_ioService),
	m_key(KeyPair::create()),
	m_requiredNetworkId(_networkId)
{
	// populate addresses.
	populateAddresses();
	clog(NetNote) << "Id:" << asHex(m_key.address().ref().cropped(0, 4)) << "Mode: " << (m_mode == NodeMode::PeerServer ? "PeerServer" : "Full");
}

PeerServer::~PeerServer()
{
	for (auto const& i: m_peers)
		if (auto p = i.second.lock())
			p->disconnect(ClientQuit);
	delete m_upnp;
}

void PeerServer::determinePublic(string const& _publicAddress, bool _upnp)
{
	if (_upnp)
		try
		{
			m_upnp = new UPnP;
		}
		catch (NoUPnPDevice) {}	// let m_upnp continue as null - we handle it properly.

	bi::tcp::resolver r(m_ioService);
	if (m_upnp && m_upnp->isValid() && m_peerAddresses.size())
	{
		clog(NetNote) << "External addr: " << m_upnp->externalIP();
		int p = m_upnp->addRedirect(m_peerAddresses[0].to_string().c_str(), m_listenPort);
		if (p)
			clog(NetNote) << "Punched through NAT and mapped local port" << m_listenPort << "onto external port" << p << ".";
		else
		{
			// couldn't map
			clog(NetWarn) << "Couldn't punch through NAT (or no NAT in place). Assuming " << m_listenPort << " is local & external port.";
			p = m_listenPort;
		}

		auto eip = m_upnp->externalIP();
		if (eip == string("0.0.0.0") && _publicAddress.empty())
			m_public = bi::tcp::endpoint(bi::address(), p);
		else
		{
			m_public = bi::tcp::endpoint(bi::address::from_string(_publicAddress.empty() ? eip : _publicAddress), p);
			m_addresses.push_back(m_public.address().to_v4());
		}
	}
	else
	{
		// No UPnP - fallback on given public address or, if empty, the assumed peer address.
		m_public = bi::tcp::endpoint(_publicAddress.size() ? bi::address::from_string(_publicAddress)
									: m_peerAddresses.size() ? m_peerAddresses[0]
									: bi::address(), m_listenPort);
		m_addresses.push_back(m_public.address().to_v4());
	}
}

void PeerServer::populateAddresses()
{
#ifdef _WIN32
	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0)
		throw NoNetworking();

	char ac[80];
	if (gethostname(ac, sizeof(ac)) == SOCKET_ERROR)
	{
		clog(NetWarn) << "Error " << WSAGetLastError() << " when getting local host name.";
		WSACleanup();
		throw NoNetworking();
	}

	struct hostent* phe = gethostbyname(ac);
	if (phe == 0)
	{
		clog(NetWarn) << "Bad host lookup.";
		WSACleanup();
		throw NoNetworking();
	}

	for (int i = 0; phe->h_addr_list[i] != 0; ++i)
	{
		struct in_addr addr;
		memcpy(&addr, phe->h_addr_list[i], sizeof(struct in_addr));
		char *addrStr = inet_ntoa(addr);
		bi::address ad(bi::address::from_string(addrStr));
		m_addresses.push_back(ad.to_v4());
		bool isLocal = std::find(c_rejectAddresses.begin(), c_rejectAddresses.end(), ad) != c_rejectAddresses.end();
		if (isLocal)
			m_peerAddresses.push_back(ad.to_v4());
		clog(NetNote) << "Address: " << ac << " = " << m_addresses.back() << (isLocal ? " [LOCAL]" : " [PEER]");
	}

	WSACleanup();
#else
	ifaddrs* ifaddr;
	if (getifaddrs(&ifaddr) == -1)
		throw NoNetworking();

	bi::tcp::resolver r(m_ioService);

	for (ifaddrs* ifa = ifaddr; ifa; ifa = ifa->ifa_next)
	{
		if (!ifa->ifa_addr)
			continue;
		if (ifa->ifa_addr->sa_family == AF_INET)
		{
			char host[NI_MAXHOST];
			if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST))
				continue;
			auto it = r.resolve({host, "30303"});
			bi::tcp::endpoint ep = it->endpoint();
			bi::address ad = ep.address();
			m_addresses.push_back(ad.to_v4());
			bool isLocal = std::find(c_rejectAddresses.begin(), c_rejectAddresses.end(), ad) != c_rejectAddresses.end();
			if (!isLocal)
				m_peerAddresses.push_back(ad.to_v4());
			clog(NetNote) << "Address: " << host << " = " << m_addresses.back() << (isLocal ? " [LOCAL]" : " [PEER]");
		}
	}

	freeifaddrs(ifaddr);
#endif
}

std::map<Public, bi::tcp::endpoint> PeerServer::potentialPeers()
{
	std::map<Public, bi::tcp::endpoint> ret;
	if (!m_public.address().is_unspecified())
		ret.insert(make_pair(m_key.pub(), m_public));
	for (auto i: m_peers)
		if (auto j = i.second.lock())
		{
			auto ep = j->endpoint();
			if (ep.port() && j->m_id)
				ret.insert(make_pair(i.first, ep));
		}
	return ret;
}

void PeerServer::ensureAccepting()
{
	if (m_accepting == false)
	{
		clog(NetNote) << "Listening on local port " << m_listenPort << " (public: " << m_public << ")";
		m_accepting = true;
		m_acceptor.async_accept(m_socket, [=](boost::system::error_code ec)
		{
			if (!ec)
				try
				{
					try {
						clog(NetNote) << "Accepted connection from " << m_socket.remote_endpoint();
					} catch (...){}
					auto p = std::make_shared<PeerSession>(this, std::move(m_socket), m_requiredNetworkId);
					p->start();
				}
				catch (std::exception const& _e)
				{
					clog(NetWarn) << "ERROR: " << _e.what();
				}

			m_accepting = false;
			if (m_mode == NodeMode::PeerServer || m_peers.size() < m_idealPeerCount)
				ensureAccepting();
		});
	}
}

void PeerServer::connect(bi::tcp::endpoint const& _ep)
{
	clog(NetNote) << "Attempting connection to " << _ep;
	bi::tcp::socket* s = new bi::tcp::socket(m_ioService);
	s->async_connect(_ep, [=](boost::system::error_code const& ec)
	{
		if (ec)
		{
			clog(NetNote) << "Connection refused to " << _ep << " (" << ec.message() << ")";
		}
		else
		{
			auto p = make_shared<PeerSession>(this, std::move(*s), m_requiredNetworkId);
			clog(NetNote) << "Connected to " << p->endpoint();
			p->start();
		}
		delete s;
	});
}

bool PeerServer::process(BlockChain& _bc)
{
	bool ret = false;
	m_ioService.poll();

	auto n = chrono::steady_clock::now();
	bool fullProcess = (n > m_lastFullProcess + chrono::seconds(1));
	if (fullProcess)
		m_lastFullProcess = n;

	if (fullProcess)
		for (auto i = m_peers.begin(); i != m_peers.end();)
		{
			auto p = i->second.lock();
			if (p && p->m_socket.is_open() &&
					(p->m_disconnect == chrono::steady_clock::time_point::max() || chrono::steady_clock::now() - p->m_disconnect < chrono::seconds(1)))	// kill old peers that should be disconnected.
				++i;
			else
			{
				i = m_peers.erase(i);
				ret = true;
			}
		}
	return ret;
}

bool PeerServer::process(BlockChain& _bc, TransactionQueue& _tq, Overlay& _o)
{
	bool ret = false;

	if (m_latestBlockSent == h256())
	{
		// First time - just initialise.
		m_latestBlockSent = _bc.currentHash();
		clog(NetNote) << "Initialising: latest=" << m_latestBlockSent;

		for (auto const& i: _tq.transactions())
			m_transactionsSent.insert(i.first);
		m_lastPeersRequest = chrono::steady_clock::time_point::min();
		m_lastFullProcess = chrono::steady_clock::time_point::min();
		ret = true;
	}

	auto n = chrono::steady_clock::now();
	bool fullProcess = (n > m_lastFullProcess + chrono::seconds(1));

	if (process(_bc))
		ret = true;

	if (m_mode == NodeMode::Full)
	{
		for (auto it = m_incomingTransactions.begin(); it != m_incomingTransactions.end(); ++it)
			if (_tq.import(*it))
				ret = true;
			else
				m_transactionsSent.insert(sha3(*it));	// if we already had the transaction, then don't bother sending it on.
		m_incomingTransactions.clear();

		// Send any new transactions.
		if (fullProcess)
		{
			for (auto j: m_peers)
				if (auto p = j.second.lock())
				{
					bytes b;
					uint n = 0;
					for (auto const& i: _tq.transactions())
						if ((!m_transactionsSent.count(i.first) && !p->m_knownTransactions.count(i.first)) || p->m_requireTransactions)
						{
							b += i.second;
							++n;
							m_transactionsSent.insert(i.first);
						}
					if (n)
					{
						RLPStream ts;
						PeerSession::prep(ts);
						ts.appendList(n + 1) << TransactionsPacket;
						ts.appendRaw(b).swapOut(b);
						seal(b);
						p->send(&b);
					}
					p->m_knownTransactions.clear();
					p->m_requireTransactions = false;
				}

			// Send any new blocks.
			auto h = _bc.currentHash();
			if (h != m_latestBlockSent)
			{
				// TODO: find where they diverge and send complete new branch.
				RLPStream ts;
				PeerSession::prep(ts);
				ts.appendList(2) << BlocksPacket;
				bytes b;
				ts.appendRaw(_bc.block(_bc.currentHash())).swapOut(b);
				seal(b);
				for (auto j: m_peers)
					if (auto p = j.second.lock())
					{
						if (!p->m_knownBlocks.count(_bc.currentHash()))
							p->send(&b);
						p->m_knownBlocks.clear();
					}
			}
			m_latestBlockSent = h;

			for (int accepted = 1; accepted;)
			{
				accepted = 0;
				if (m_incomingBlocks.size())
					for (auto it = prev(m_incomingBlocks.end());; --it)
					{
						try
						{
							_bc.import(*it, _o);
							it = m_incomingBlocks.erase(it);
							++accepted;
							ret = true;
						}
						catch (UnknownParent)
						{
							// Don't (yet) know its parent. Leave it for later.
						}
						catch (...)
						{
							// Some other error - erase it.
							it = m_incomingBlocks.erase(it);
						}

						if (it == m_incomingBlocks.begin())
							break;
					}
			}

			// Connect to additional peers
			while (m_peers.size() < m_idealPeerCount)
			{
				if (m_incomingPeers.empty())
				{
					if (chrono::steady_clock::now() > m_lastPeersRequest + chrono::seconds(10))
					{
						RLPStream s;
						bytes b;
						(PeerSession::prep(s).appendList(1) << GetPeersPacket).swapOut(b);
						seal(b);
						for (auto const& i: m_peers)
							if (auto p = i.second.lock())
								if (p->isOpen())
									p->send(&b);
						m_lastPeersRequest = chrono::steady_clock::now();
					}


					if (!m_accepting)
						ensureAccepting();

					break;
				}
				connect(m_incomingPeers.begin()->second);
				m_incomingPeers.erase(m_incomingPeers.begin());
			}
		}
	}

	// platform for consensus of social contract.
	// restricts your freedom but does so fairly. and that's the value proposition.
	// guarantees that everyone else respect the rules of the system. (i.e. obeys laws).

	if (fullProcess)
	{
		// We'll keep at most twice as many as is ideal, halfing what counts as "too young to kill" until we get there.
		for (uint old = 15000; m_peers.size() > m_idealPeerCount * 2 && old > 100; old /= 2)
			while (m_peers.size() > m_idealPeerCount)
			{
				// look for worst peer to kick off
				// first work out how many are old enough to kick off.
				shared_ptr<PeerSession> worst;
				unsigned agedPeers = 0;
				for (auto i: m_peers)
					if (auto p = i.second.lock())
						if ((m_mode != NodeMode::PeerServer || p->m_caps != 0x01) && chrono::steady_clock::now() > p->m_connect + chrono::milliseconds(old))	// don't throw off new peers; peer-servers should never kick off other peer-servers.
						{
							++agedPeers;
							if ((!worst || p->m_rating < worst->m_rating || (p->m_rating == worst->m_rating && p->m_connect > worst->m_connect)))	// kill older ones
								worst = p;
						}
				if (!worst || agedPeers <= m_idealPeerCount)
					break;
				worst->disconnect(TooManyPeers);
			}
	}

	return ret;
}

std::vector<PeerInfo> PeerServer::peers() const
{
	const_cast<PeerServer*>(this)->pingAll();
	this_thread::sleep_for(chrono::milliseconds(200));
	std::vector<PeerInfo> ret;
	for (auto& i: m_peers)
		if (auto j = i.second.lock())
			if (j->m_socket.is_open())
				ret.push_back(j->m_info);
	return ret;
}

void PeerServer::pingAll()
{
	for (auto& i: m_peers)
		if (auto j = i.second.lock())
			j->ping();
}
