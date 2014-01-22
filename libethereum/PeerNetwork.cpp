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
#include "PeerNetwork.h"
using namespace std;
using namespace eth;

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
	}
	return true;
}

PeerSession::PeerSession(bi::tcp::socket _socket, uint _rNId): m_socket(std::move(_socket)), m_reqNetworkId(_rNId)
{
}

PeerSession::~PeerSession()
{
	disconnect();
}

void PeerSession::prep(RLPStream& _s)
{
	_s.appendRaw(bytes(8, 0));
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
	ba::async_write(m_socket, ba::buffer(*buffer), [=](boost::system::error_code ec, std::size_t length)
	{
	});
}

void PeerSession::disconnect()
{
	RLPStream s;
	prep(s);
	s.appendList(1) << Disconnect;
	sealAndSend(s);
	sleep(1);
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
			std::cout << "Got data" << std::endl;
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
		//doWrite(length);
		doRead();
	});
}

PeerServer::PeerServer(uint _networkId, short _port):
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

void PeerServer::doAccept()
{
	cout << "Listening on " << m_acceptor.local_endpoint() << endl;
	m_acceptor.async_accept(m_socket, [&](boost::system::error_code ec)
	{
		if (!ec)
		{
			std::cout << "Accepted connection from " << m_socket.remote_endpoint() << std::endl;
			auto p = std::make_shared<PeerSession>(std::move(m_socket), m_requiredNetworkId);
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
		auto p = make_shared<PeerSession>(std::move(s), m_requiredNetworkId);
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

void PeerServer::sync(BlockChain& _bc, TransactionQueue const& _tq)
{
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
