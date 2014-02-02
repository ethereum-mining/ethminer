/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 2 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file PeerNetwork.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <sys/types.h>
#include <ifaddrs.h>

#include <chrono>
#include <miniupnpc/miniupnpc.h>
#include "Common.h"
#include "BlockChain.h"
#include "BlockInfo.h"
#include "TransactionQueue.h"
#include "PeerNetwork.h"
using namespace std;
using namespace eth;

static const eth::uint c_maxHashes = 256;		///< Maximum number of hashes GetChain will ever send.
static const eth::uint c_maxBlocks = 128;		///< Maximum number of blocks Blocks will ever send. BUG: if this gets too big (e.g. 2048) stuff starts going wrong.
static const eth::uint c_maxBlocksAsk = 2048;	///< Maximum number of blocks we ask to receive in Blocks (when using GetChain).

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
	if (m_server->m_verbosity >= 8)
		cout << ">>> " << _r << endl;
	switch (_r[0].toInt<unsigned>())
	{
	case Hello:
	{
		m_protocolVersion = _r[1].toInt<uint>();
		m_networkId = _r[2].toInt<uint>();
		auto clientVersion = _r[3].toString();
		m_caps = _r.itemCount() > 4 ? _r[4].toInt<uint>() : 0x07;
		m_listenPort = _r.itemCount() > 5 ? _r[5].toInt<short>() : 0;

		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | Hello: " << clientVersion << " " << showbase << hex << m_caps << dec << " " << m_listenPort << endl;

		if (m_protocolVersion != 0 || m_networkId != m_reqNetworkId)
		{
			disconnect();
			return false;
		}
		try
			{ m_info = PeerInfo({clientVersion, m_socket.remote_endpoint().address().to_string(), (short)m_socket.remote_endpoint().port(), std::chrono::steady_clock::duration()}); }
		catch (...)
		{
			disconnect();
			return false;
		}

		// Grab their block chain off them.
		{
			unsigned count = std::min<unsigned>(c_maxHashes, m_server->m_chain->details(m_server->m_latestBlockSent).number + 1);
			RLPStream s;
			prep(s).appendList(2 + count);
			s << (uint)GetChain;
			auto h = m_server->m_latestBlockSent;
			for (unsigned i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
				s << h;
			s << c_maxBlocksAsk;
			sealAndSend(s);
			s.clear();
			prep(s).appendList(1);
			s << GetTransactions;
			sealAndSend(s);
		}
		break;
	}
	case Disconnect:
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | Disconnect" << endl;
		if (m_server->m_verbosity >= 1)
		{
			if (m_socket.is_open())
				cout << "Closing " << m_socket.remote_endpoint() << endl;
			else
				cout << "Remote closed on " << m_socket.native_handle() << endl;
		}
		m_socket.close();
		return false;
	case Ping:
	{
//		cout << std::setw(2) << m_socket.native_handle() << " | Ping" << endl;
		RLPStream s;
		sealAndSend(prep(s).appendList(1) << (uint)Pong);
		break;
	}
	case Pong:
		m_info.lastPing = std::chrono::steady_clock::now() - m_ping;
//		cout << "Latency: " << chrono::duration_cast<chrono::milliseconds>(m_lastPing).count() << " ms" << endl;
		break;
	case GetPeers:
	{
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | GetPeers" << endl;
		std::vector<bi::tcp::endpoint> peers = m_server->potentialPeers();
		RLPStream s;
		prep(s).appendList(peers.size() + 1);
		s << (uint)Peers;
		for (auto i: peers)
		{
			if (m_server->m_verbosity >= 3)
				cout << "  Sending peer " << i << endl;
			s.appendList(2) << i.address().to_v4().to_bytes() << i.port();
		}
		sealAndSend(s);
		break;
	}
	case Peers:
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | Peers (" << dec << (_r.itemCount() - 1) << " entries)" << endl;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			auto ep = bi::tcp::endpoint(bi::address_v4(_r[i][0].toArray<byte, 4>()), _r[i][1].toInt<short>());
			if (m_server->m_verbosity >= 6)
				cout << "Checking: " << ep << endl;
			// check that we're not already connected to addr:
			if (!ep.port())
				goto CONTINUE;
			for (auto i: m_server->m_addresses)
				if (ep.address() == i && ep.port() == m_server->listenPort())
					goto CONTINUE;
			for (auto i: m_server->m_peers)
				if (shared_ptr<PeerSession> p = i.lock())
				{
					if (m_server->m_verbosity >= 6)
						cout << "   ...against " << p->endpoint() << endl;
					if (p->m_socket.is_open() && p->endpoint() == ep)
						goto CONTINUE;
				}
			for (auto i: m_server->m_incomingPeers)
				if (i == ep)
					goto CONTINUE;
			m_server->m_incomingPeers.push_back(ep);
			if (m_server->m_verbosity >= 3)
				cout << "New peer: " << ep << endl;
			CONTINUE:;
		}
		break;
	case Transactions:
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | Transactions (" << dec << (_r.itemCount() - 1) << " entries)" << endl;
		m_rating += _r.itemCount() - 1;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			m_server->m_incomingTransactions.push_back(_r[i].data().toBytes());
			m_knownTransactions.insert(sha3(_r[i].data()));
		}
		break;
	case Blocks:
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | Blocks (" << dec << (_r.itemCount() - 1) << " entries)" << endl;
		m_rating += _r.itemCount() - 1;
		for (unsigned i = 1; i < _r.itemCount(); ++i)
		{
			m_server->m_incomingBlocks.push_back(_r[i].data().toBytes());
			m_knownBlocks.insert(sha3(_r[i].data()));
		}
		if (m_server->m_verbosity >= 3)
			for (unsigned i = 1; i < _r.itemCount(); ++i)
			{
				auto h = sha3(_r[i].data());
				BlockInfo bi(_r[i].data());
				if (!m_server->m_chain->details(bi.parentHash) && !m_knownBlocks.count(bi.parentHash))
					cerr << "*** Unknown parent " << bi.parentHash << " of block " << h << endl;
				else
					cerr << "--- Known parent " << bi.parentHash << " of block " << h << endl;
			}
		if (_r.itemCount() > 1)	// we received some - check if there's any more
		{
			RLPStream s;
			prep(s).appendList(3);
			s << (uint)GetChain;
			s << sha3(_r[1].data());
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		break;
	case GetChain:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		// ********************************************************************
		// NEEDS FULL REWRITE!
		h256s parents;
		parents.reserve(_r.itemCount() - 2);
		for (unsigned i = 1; i < _r.itemCount() - 1; ++i)
			parents.push_back(_r[i].toHash<h256>());
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | GetChain (" << (_r.itemCount() - 2) << " hashes, " << (_r[_r.itemCount() - 1].toInt<bigint>()) << ")" << endl;
		if (_r.itemCount() == 2)
			break;
		// return 2048 block max.
		uint baseCount = (uint)min<bigint>(_r[_r.itemCount() - 1].toInt<bigint>(), c_maxBlocks);
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | GetChain (" << baseCount << " max, from " << parents.front() << " to " << parents.back() << ")" << endl;
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
				if (m_server->m_verbosity >= 6)
					cout << "Requires " << dec << (latestNumber - parentNumber) << " blocks from " << latestNumber << " to " << parentNumber << endl
					<< latest << " - " << parent << endl;

				prep(s);
				s.appendList(1 + count) << (uint)Blocks;
				uint endNumber = m_server->m_chain->details(parent).number;
				uint startNumber = endNumber + count;
				if (m_server->m_verbosity >= 6)
					cout << "Sending " << dec << count << " blocks from " << startNumber << " to " << endNumber << endl;

				uint n = latestNumber;
				for (; n > startNumber; n--, h = m_server->m_chain->details(h).parent) {}
				for (uint i = 0; h != parent && n > endNumber && i < count; ++i, --n, h = m_server->m_chain->details(h).parent)
				{
					if (m_server->m_verbosity >= 6)
						cout << "   " << dec << i << " " << h << endl;
					s.appendRaw(m_server->m_chain->block(h));
				}
				if (m_server->m_verbosity >= 6)
					cout << "Parent: " << h << endl;
			}
			else if (parent != parents.back())
				continue;

			if (h != parent)
			{
				// not in the blockchain;
				if (parent == parents.back())
				{
					// out of parents...
					if (m_server->m_verbosity >= 6)
						cout << std::setw(2) << m_socket.native_handle() << " | GetChain failed; not in chain" << endl;
					// No good - must have been on a different branch.
					s.clear();
					prep(s).appendList(2) << (uint)NotInChain << parents.back();
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
	case NotInChain:
	{
		if (m_server->m_mode == NodeMode::PeerServer)
			break;
		h256 noGood = _r[1].toHash<h256>();
		if (m_server->m_verbosity >= 2)
			cout << std::setw(2) << m_socket.native_handle() << " | NotInChain (" << noGood << ")" << endl;
		if (noGood == m_server->m_chain->genesisHash())
		{
			if (m_server->m_verbosity)
				cout << std::setw(2) << m_socket.native_handle() << " | Discordance over genesis block! Disconnect." << endl;
			disconnect();
		}
		else
		{
			unsigned count = std::min<unsigned>(c_maxHashes, m_server->m_chain->details(noGood).number);
			RLPStream s;
			prep(s).appendList(2 + count);
			s << (uint)GetChain;
			auto h = m_server->m_chain->details(noGood).parent;
			for (unsigned i = 0; i < count; ++i, h = m_server->m_chain->details(h).parent)
				s << h;
			s << c_maxBlocksAsk;
			sealAndSend(s);
		}
		break;
	}
	case GetTransactions:
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
	sealAndSend(prep(s).appendList(1) << Ping);
	m_ping = std::chrono::steady_clock::now();
}

RLPStream& PeerSession::prep(RLPStream& _s)
{
	return _s.appendRaw(bytes(8, 0));
}

void PeerServer::seal(bytes& _b)
{
	if (m_verbosity >= 9)
		cout << "<<< " << RLP(bytesConstRef(&_b).cropped(8)) << endl;
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
	std::shared_ptr<bytes> buffer = std::make_shared<bytes>();
	swap(*buffer, _msg);
	assert((*buffer)[0] == 0x22);
//	cout << "Sending " << (buffer->size() - 8) << endl;
//	cout << "Sending " << RLP(bytesConstRef(buffer.get()).cropped(8)) << endl;
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			dropped();
//		cout << length << " bytes written (EC: " << ec << ")" << endl;
	});
}

void PeerSession::send(bytesConstRef _msg)
{
	std::shared_ptr<bytes> buffer = std::make_shared<bytes>(_msg.toBytes());
	assert((*buffer)[0] == 0x22);
//	cout << "Sending " << (_msg.size() - 8) << endl;// RLP(bytesConstRef(buffer.get()).cropped(8)) << endl;
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			dropped();
//		cout << length << " bytes written (EC: " << ec << ")" << endl;
	});
}

void PeerSession::dropped()
{
	if (m_server->m_verbosity >= 1)
	{
		if (m_socket.is_open())
			try {
				cout << "Closing " << m_socket.remote_endpoint() << endl;
			}catch (...){}
	}
	m_socket.close();
	for (auto i = m_server->m_peers.begin(); i != m_server->m_peers.end(); ++i)
		if (i->lock().get() == this)
		{
			m_server->m_peers.erase(i);
			break;
		}
}

void PeerSession::disconnect()
{
	if (m_socket.is_open())
	{
		if (m_disconnect == chrono::steady_clock::time_point::max())
		{
			RLPStream s;
			prep(s);
			s.appendList(1) << (uint)Disconnect;
			sealAndSend(s);
			m_disconnect = chrono::steady_clock::now();
		}
		else
		{
			if (m_server->m_verbosity >= 1)
			{
				if (m_socket.is_open())
					try {
						cout << "Closing " << m_socket.remote_endpoint() << endl;
					} catch (...){}
				else
					cout << "Remote closed on" << m_socket.native_handle() << endl;
			}
			m_socket.close();
		}
	}
}

void PeerSession::start()
{
	RLPStream s;
	prep(s);
	s.appendList(m_server->m_public.port() ? 6 : 5) << (uint)Hello << (uint)0 << (uint)0 << m_server->m_clientVersion << (m_server->m_mode == NodeMode::Full ? 0x07 : m_server->m_mode == NodeMode::PeerServer ? 0x01 : 0);
	if (m_server->m_public.port())
		s << m_server->m_public.port();
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
						if (m_server->m_verbosity)
							cerr << std::setw(2) << m_socket.native_handle() << " | Out of alignment. Skipping: " << hex << showbase << (int)m_incoming[0] << dec << endl;
						memmove(m_incoming.data(), m_incoming.data() + 1, m_incoming.size() - 1);
						m_incoming.resize(m_incoming.size() - 1);
					}
					else
					{
						uint32_t len = fromBigEndian<uint32_t>(bytesConstRef(m_incoming.data() + 4, 4));
	//					cout << "Received packet of " << len << " bytes" << endl;
						if (m_incoming.size() - 8 < len)
							break;

						// enough has come in.
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
			catch (std::exception const& _e)
			{
				if (m_server->m_verbosity)
					cerr << std::setw(2) << m_socket.native_handle() << " | ERROR: " << _e.what() << endl;
				dropped();
			}
		}
	});
}

#include <stdio.h>
#include <string.h>

#include <miniupnpc/miniwget.h>
#include <miniupnpc/miniupnpc.h>
#include <miniupnpc/upnpcommands.h>

namespace eth {
struct UPnP
{
	UPnP()
	{
		ok = false;

		struct UPNPDev * devlist;
		struct UPNPDev * dev;
		char * descXML;
		int descXMLsize = 0;
		int upnperror = 0;
		printf("TB : init_upnp()\n");
		memset(&urls, 0, sizeof(struct UPNPUrls));
		memset(&data, 0, sizeof(struct IGDdatas));
		devlist = upnpDiscover(2000, NULL/*multicast interface*/, NULL/*minissdpd socket path*/, 0/*sameport*/, 0/*ipv6*/, &upnperror);
		if (devlist)
		{
			dev = devlist;
			while (dev)
			{
				if (strstr (dev->st, "InternetGatewayDevice"))
					break;
				dev = dev->pNext;
			}
			if (!dev)
				dev = devlist; /* defaulting to first device */

			printf("UPnP device :\n"
				   " desc: %s\n st: %s\n",
				   dev->descURL, dev->st);
#if MINIUPNPC_API_VERSION >= 9
			descXML = (char*)miniwget(dev->descURL, &descXMLsize, 0);
#else
			descXML = (char*)miniwget(dev->descURL, &descXMLsize);
#endif
			if (descXML)
			{
				parserootdesc (descXML, descXMLsize, &data);
				free (descXML); descXML = 0;
#if MINIUPNPC_API_VERSION >= 9
				GetUPNPUrls (&urls, &data, dev->descURL, 0);
#else
				GetUPNPUrls (&urls, &data, dev->descURL);
#endif
				ok = true;
			}
			freeUPNPDevlist(devlist);
		}
		else
		{
			/* error ! */
		}
	}
	~UPnP()
	{
		auto r = m_reg;
		for (auto i: r)
			removeRedirect(i);
	}

	string externalIP()
	{
		char addr[16];
		UPNP_GetExternalIPAddress(urls.controlURL, data.first.servicetype, addr);
		return addr;
	}

	int addRedirect(char const* addr, int port)
	{
		char port_str[16];
		int r;
		printf("TB : upnp_add_redir (%d)\n", port);
		if (urls.controlURL[0] == '\0')
		{
			printf("TB : the init was not done !\n");
			return -1;
		}
		sprintf(port_str, "%d", port);
		r = UPNP_AddPortMapping(urls.controlURL, data.first.servicetype, port_str, port_str, addr, "ethereum", "TCP", NULL, NULL);
		if (r)
		{
			printf("AddPortMapping(%s, %s, %s) failed with %d. Trying non-specific external port...\n", port_str, port_str, addr, r);
			r = UPNP_AddPortMapping(urls.controlURL, data.first.servicetype, port_str, NULL, addr, "ethereum", "TCP", NULL, NULL);
		}
		if (r)
		{
			printf("AddPortMapping(%s, NULL, %s) failed with %d. Trying non-specific internal port...\n", port_str, addr, r);
			r = UPNP_AddPortMapping(urls.controlURL, data.first.servicetype, NULL, port_str, addr, "ethereum", "TCP", NULL, NULL);
		}
		if (r)
		{
			printf("AddPortMapping(NULL, %s, %s) failed with %d. Trying non-specific both ports...\n", port_str, addr, r);
			r = UPNP_AddPortMapping(urls.controlURL, data.first.servicetype, NULL, NULL, addr, "ethereum", "TCP", NULL, NULL);
		}
		if (r)
			printf("AddPortMapping(NULL, NULL, %s) failed with %d\n", addr, r);
		else
		{
			unsigned num = 0;
			UPNP_GetPortMappingNumberOfEntries(urls.controlURL, data.first.servicetype, &num);
			for (unsigned i = 0; i < num; ++i)
			{
				char extPort[16];
				char intClient[16];
				char intPort[6];
				char protocol[4];
				char desc[80];
				char enabled[4];
				char rHost[64];
				char duration[16];
				UPNP_GetGenericPortMappingEntry(urls.controlURL, data.first.servicetype, toString(i).c_str(), extPort, intClient, intPort, protocol, desc, enabled, rHost, duration);
				if (string("ethereum") == desc)
				{
					m_reg.insert(atoi(extPort));
					return atoi(extPort);
				}
			}
			cerr << "ERROR: Mapped port not found." << endl;
		}
		return 0;
	}

	void removeRedirect(int port)
	{
		char port_str[16];
//		int t;
		printf("TB : upnp_rem_redir (%d)\n", port);
		if (urls.controlURL[0] == '\0')
		{
			printf("TB : the init was not done !\n");
			return;
		}
		sprintf(port_str, "%d", port);
		UPNP_DeletePortMapping(urls.controlURL, data.first.servicetype, port_str, "TCP", NULL);
		m_reg.erase(port);
	}

	bool isValid() const
	{
		return ok;
	}

	set<int> m_reg;

	bool ok;
	struct UPNPUrls urls;
	struct IGDdatas data;
};
}

class NoNetworking: public std::exception {};

PeerServer::PeerServer(std::string const& _clientVersion, BlockChain const& _ch, uint _networkId, short _port, NodeMode _m, string const& _publicAddress):
	m_clientVersion(_clientVersion),
	m_mode(_m),
	m_listenPort(_port),
	m_chain(&_ch),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), _port)),
	m_socket(m_ioService),
	m_requiredNetworkId(_networkId)
{
	populateAddresses();
	determinePublic(_publicAddress);
	ensureAccepting();
	if (m_verbosity)
		cout << "Mode: " << (_m == NodeMode::PeerServer ? "PeerServer" : "Full") << endl;
}

PeerServer::PeerServer(std::string const& _clientVersion, uint _networkId):
	m_clientVersion(_clientVersion),
	m_listenPort(-1),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), 0)),
	m_socket(m_ioService),
	m_requiredNetworkId(_networkId)
{
	// populate addresses.
	populateAddresses();
	if (m_verbosity)
		cout << "Genesis: " << m_chain->genesisHash() << endl;
}

PeerServer::~PeerServer()
{
	for (auto const& i: m_peers)
		if (auto p = i.lock())
			p->disconnect();
	delete m_upnp;
}

void PeerServer::determinePublic(string const& _publicAddress)
{
	m_upnp = new UPnP;
	if (m_upnp->isValid() && m_peerAddresses.size())
	{
		bi::tcp::resolver r(m_ioService);
		cout << "external addr: " << m_upnp->externalIP() << endl;
		int p = m_upnp->addRedirect(m_peerAddresses[0].to_string().c_str(), m_listenPort);
		if (!p)
		{
			// couldn't map
			cerr << "*** WARNING: Couldn't punch through NAT (or no NAT in place). Using " << m_listenPort << endl;
			p = m_listenPort;
		}

		if (m_upnp->externalIP() == string("0.0.0.0") && _publicAddress.empty())
			m_public = bi::tcp::endpoint(bi::address(), p);
		else
		{
			auto it = r.resolve({_publicAddress.empty() ? m_upnp->externalIP() : _publicAddress, toString(p)});
			m_public = it->endpoint();
			m_addresses.push_back(m_public.address().to_v4());
		}
	}
/*	int er;
	UPNPDev* dlist = upnpDiscover(250, 0, 0, 0, 0, &er);
	for (UPNPDev* d = dlist; d; d = dlist->pNext)
	{
		IGDdatas data;
		parserootdesc(d->descURL, 0, &data);
		data.presentationurl()
	}
	freeUPNPDevlist(dlist);*/
}

void PeerServer::populateAddresses()
{
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
			m_addresses.push_back(ep.address().to_v4());
			if (ifa->ifa_name != string("lo"))
				m_peerAddresses.push_back(ep.address().to_v4());
			if (m_verbosity >= 1)
				cout << "Address: " << host << " = " << m_addresses.back() << (ifa->ifa_name != string("lo") ? " [PEER]" : " [LOCAL]") << endl;
		}
	}

	freeifaddrs(ifaddr);
}

std::vector<bi::tcp::endpoint> PeerServer::potentialPeers()
{
	std::vector<bi::tcp::endpoint> ret;
	if (!m_public.address().is_unspecified())
		ret.push_back(m_public);
	for (auto i: m_peers)
		if (auto j = i.lock())
		{
			auto ep = j->endpoint();
			if (ep.port())
				ret.push_back(ep);
		}
	return ret;
}

void PeerServer::ensureAccepting()
{
	if (m_accepting == false)
	{
		if (m_verbosity >= 1)
			cout << "Listening on local port " << m_listenPort << " (public: " << m_public << ")" << endl;
		m_accepting = true;
		m_acceptor.async_accept(m_socket, [=](boost::system::error_code ec)
		{
			if (!ec)
				try
				{
					if (m_verbosity >= 1)
						try {
							cout << "Accepted connection from " << m_socket.remote_endpoint() << std::endl;
						} catch (...){}
					auto p = std::make_shared<PeerSession>(this, std::move(m_socket), m_requiredNetworkId);
					m_peers.push_back(p);
					p->start();
				}
				catch (std::exception const& _e)
				{
					if (m_verbosity)
						cerr << "*** ERROR: " << _e.what() << endl;
				}

			m_accepting = false;
			if (m_mode == NodeMode::PeerServer || m_peers.size() < m_idealPeerCount)
				ensureAccepting();
		});
	}
}

void PeerServer::connect(bi::tcp::endpoint const& _ep)
{
	if (m_verbosity >= 1)
		cout << "Attempting connection to " << _ep << endl;
	bi::tcp::socket* s = new bi::tcp::socket(m_ioService);
	s->async_connect(_ep, [=](boost::system::error_code const& ec)
	{
		if (ec)
		{
			if (m_verbosity >= 1)
				cout << "Connection refused to " << _ep << " (" << ec.message() << ")" << endl;
		}
		else
		{
			auto p = make_shared<PeerSession>(this, std::move(*s), m_requiredNetworkId);
			m_peers.push_back(p);
			if (m_verbosity >= 1)
				cout << "Connected to " << p->endpoint() << endl;
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
			auto p = i->lock();
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
		if (m_verbosity)
			cout << "Initialising: latest=" << m_latestBlockSent << endl;

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
				if (auto p = j.lock())
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
						ts.appendList(n + 1) << Transactions;
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
				ts.appendList(2) << Blocks;
				bytes b;
				ts.appendRaw(_bc.block(_bc.currentHash())).swapOut(b);
				seal(b);
				for (auto j: m_peers)
					if (auto p = j.lock())
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
						(PeerSession::prep(s).appendList(1) << GetPeers).swapOut(b);
						seal(b);
						for (auto const& i: m_peers)
							if (auto p = i.lock())
								p->send(&b);
						m_lastPeersRequest = chrono::steady_clock::now();
					}


					if (!m_accepting)
						ensureAccepting();

					break;
				}
				connect(m_incomingPeers.back());
				m_incomingPeers.pop_back();
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
					if (auto p = i.lock())
						if ((m_mode != NodeMode::PeerServer || p->m_caps != 0x01) && chrono::steady_clock::now() > p->m_connect + chrono::milliseconds(old))	// don't throw off new peers; peer-servers should never kick off other peer-servers.
						{
							++agedPeers;
							if ((!worst || p->m_rating < worst->m_rating || (p->m_rating == worst->m_rating && p->m_connect > worst->m_connect)))	// kill older ones
								worst = p;
						}
				if (!worst || agedPeers <= m_idealPeerCount)
					break;
				worst->disconnect();
			}
	}

	return ret;
}

std::vector<PeerInfo> PeerServer::peers() const
{
	const_cast<PeerServer*>(this)->pingAll();
	usleep(200000);
	std::vector<PeerInfo> ret;
	for (auto& i: m_peers)
		if (auto j = i.lock())
			if (j->m_socket.is_open())
				ret.push_back(j->m_info);
	return ret;
}

void PeerServer::pingAll()
{
	for (auto& i: m_peers)
		if (auto j = i.lock())
			j->ping();
}
