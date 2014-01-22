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
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "BlockChain.h"
#include "PeerNetwork.h"
using namespace std;
using namespace eth;

PeerSession::PeerSession(PeerServer* _s, bi::tcp::socket _socket, uint _rNId): m_server(_s), m_socket(std::move(_socket)), m_reqNetworkId(_rNId)
{
}

PeerSession::~PeerSession()
{
	disconnect();
}

bool PeerSession::interpret(RLP const& _r)
{
	switch (_r[0].toInt<unsigned>())
	{
	case Hello:
		m_protocolVersion = _r[1].toInt<uint>();
		m_networkId = _r[2].toInt<uint>();
		m_clientVersion = _r[3].toString();
		if (m_protocolVersion != 0 || m_networkId != m_reqNetworkId)
		{
			disconnect();
			return false;
		}
		cout << std::setw(2) << m_socket.native_handle() << " | Client version: " << m_clientVersion << endl;
		break;
	case Disconnect:
		m_socket.close();
		return false;
	case Ping:
	{
		RLPStream s;
		sealAndSend(prep(s).appendList(1) << Pong);
		break;
	}
	case Pong:
		cout << "Latency: " << chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - m_ping).count() << " ms" << endl;
		break;
	case GetPeers:
	{
		std::vector<bi::tcp::endpoint> peers = m_server->peers();
		RLPStream s;
		prep(s).appendList(2);
		s << Peers;
		s.appendList(peers.size());
		for (auto i: peers)
			s.appendList(2) << i.address().to_v4().to_bytes() << i.port();
		sealAndSend(s);
		break;
	}
	case Peers:
		for (auto i: _r[1])
		{
			auto ep = bi::tcp::endpoint(bi::address_v4(i[0].toArray<byte, 4>()), i[1].toInt<short>());
			m_server->m_incomingPeers.push_back(ep);
			cout << "New peer: " << ep << endl;
		}
		break;
	case Transactions:
		for (auto i: _r[1])
			m_server->m_incomingTransactions.push_back(i.data().toBytes());
		break;
	case Blocks:
		for (auto i: _r[1])
			m_server->m_incomingBlocks.push_back(i.data().toBytes());
		break;
	case GetChain:
	{
		h256 parent = _r[1].toHash<h256>();
		// return 256 block max.
		uint count = (uint)min<bigint>(_r[1].toInt<bigint>(), 256);
		h256 latest = m_server->m_chain->currentHash();
		uint latestNumber = 0;
		uint parentNumber = 0;
		if (m_server->m_chain->details(parent))
		{
			latestNumber = m_server->m_chain->details(latest).number;
			parentNumber = m_server->m_chain->details(parent).number;
		}
		count = min<uint>(latestNumber - parentNumber, count);
		RLPStream s;
		prep(s);
		s.appendList(2);
		s.append(Blocks);
		s.appendList(count);
		uint startNumber = m_server->m_chain->details(parent).number + count;
		auto h = m_server->m_chain->currentHash();
		for (uint n = latestNumber; h != parent; n--, h = m_server->m_chain->details(h).parent)
			if (m_server->m_chain->details(h).number <= startNumber)
				s.appendRaw(m_server->m_chain->block(h));
		sealAndSend(s);
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

void PeerSession::sealAndSend(RLPStream& _s)
{
	bytes b;
	_s.swapOut(b);
	b[0] = 0x22;
	b[1] = 0x40;
	b[2] = 0x08;
	b[3] = 0x91;
	uint32_t len = b.size() - 8;
	b[4] = len >> 24;
	b[5] = len >> 16;
	b[6] = len >> 8;
	b[7] = len;
	send(b);
}

void PeerSession::send(bytes& _msg)
{
	std::shared_ptr<bytes> buffer = std::make_shared<bytes>();
	swap(*buffer, _msg);
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length) {});
}

void PeerSession::disconnect()
{
	RLPStream s;
	prep(s);
	s.appendList(1) << Disconnect;
	sealAndSend(s);
	sleep(1);
	m_socket.close();
}

void PeerSession::start()
{
	cout << "Starting session." << endl;
	RLPStream s;
	prep(s);
	s.appendList(4) << (uint)Hello << (uint)0 << (uint)0 << "Ethereum++/0.1.0";
	sealAndSend(s);
	doRead();
}

void PeerSession::doRead()
{
	auto self(shared_from_this());
	m_socket.async_read_some(boost::asio::buffer(m_data), [this, self](boost::system::error_code ec, std::size_t length)
	{
		if (!ec)
		{
			m_incoming.resize(m_incoming.size() + length);
			memcpy(m_incoming.data() + m_incoming.size() - length, m_data.data(), length);
			while (m_incoming.size() > 8)
			{
				uint32_t len = fromBigEndian<uint32_t>(bytesConstRef(m_incoming.data() + 4, 4));
				if (m_incoming.size() - 8 >= len)
				{
					// enough has come in.
					RLP r(bytesConstRef(m_incoming.data() + 8, len));
					if (!interpret(r))
						// error
						break;
					memmove(m_incoming.data(), m_incoming.data() + len + 8, m_incoming.size() - (len + 8));
					m_incoming.resize(m_incoming.size() - (len + 8));
				}
				else
					break;
			}
		}
		doRead();
	});
}

PeerServer::PeerServer(BlockChain const& _ch, uint _networkId, short _port):
	m_chain(&_ch),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), _port)),
	m_socket(m_ioService),
	m_requiredNetworkId(_networkId)
{
	doAccept();
}

PeerServer::PeerServer(uint _networkId):
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), 0)),
	m_socket(m_ioService),
	m_requiredNetworkId(_networkId)
{
}

std::vector<bi::tcp::endpoint> PeerServer::peers()
{
	std::vector<bi::tcp::endpoint> ret;
	bool haveLocal = false;
	for (auto i: m_peers)
		if (auto j = i.lock())
		{
			if (!haveLocal)
				ret.push_back(j->m_socket.local_endpoint());
			haveLocal = true;
			ret.push_back(j->m_socket.remote_endpoint());
		}
	return ret;
}

void PeerServer::doAccept()
{
	cout << "Listening on " << m_acceptor.local_endpoint() << endl;
	m_acceptor.async_accept(m_socket, [&](boost::system::error_code ec)
	{
		if (!ec)
		{
			std::cout << "Accepted connection from " << m_socket.remote_endpoint() << std::endl;
			auto p = std::make_shared<PeerSession>(this, std::move(m_socket), m_requiredNetworkId);
			m_peers.push_back(p);
			p->start();
		}
		doAccept();
	});
}

bool PeerServer::connect(string const& _addr, uint _port)
{
	bi::tcp::resolver resolver(m_ioService);
	cout << "Attempting connection to " << _addr << ":" << dec << _port << endl;
	try
	{
		bi::tcp::socket s(m_ioService);
		boost::asio::connect(s, resolver.resolve({ _addr, toString(_port) }));
		auto p = make_shared<PeerSession>(this, std::move(s), m_requiredNetworkId);
		m_peers.push_back(p);
		cout << "Connected." << endl;
		p->start();
		return true;
	}
	catch (exception& _e)
	{
		cout << "Connection refused (" << _e.what() << ")" << endl;
		return false;
	}
}

void PeerServer::process(BlockChain& _bc, TransactionQueue const& _tq)
{
	m_ioService.poll();
	for (auto i = m_peers.begin(); i != m_peers.end();)
		if (auto j = i->lock())
		{}
		else
			i = m_peers.erase(i);
	/*
		while (incomingData())
		{
			// import new block
			bytes const& data = net.incomingData();
			if (!tq.attemptImport(data) && !_bc.attemptImport(data))
				handleMessage(data);
			popIncoming();
		}
	*/
}

void PeerServer::pingAll()
{
	for (auto& i: m_peers)
		if (auto j = i.lock())
			j->ping();
}
