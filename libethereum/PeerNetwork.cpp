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

bool PeerServer::connect(string const& _addr, uint _port)
{
	bi::tcp::resolver resolver(m_ioService);
	cout << "Attempting connection to " << _addr << " @" << dec << _port << endl;
	try
	{
		bi::tcp::socket s(m_ioService);
		boost::asio::connect(s, resolver.resolve({ _addr, toString(_port) }));
		auto p = make_shared<PeerSession>(std::move(s), m_requiredNetworkId);
		m_peers.push_back(p);
		p->start();
		return true;
	}
	catch (exception& _e)
	{
		cout << "Connection refused (" << _e.what() << ")" << endl;
		return false;
	}
}

#if 0
void PeerConnection::start()
{
	cout << "Connected." << endl;
	RLPStream s;
	prep(s);
	s.appendList(4) << (uint)Hello << (uint)0 << (uint)0 << "Ethereum++/0.1.0";
	sealAndSend(s);
	handleRead();
}

void PeerConnection::handleRead()
{
	m_socket.async_read_some(boost::asio::buffer(m_buffer), [&](boost::system::error_code ec, std::size_t length)
	{
		if (ec)
			return;	// bomb out on error.
		std::cout << "Got data" << std::endl;
		m_incoming.resize(m_incoming.size() + length);
		memcpy(m_incoming.data() + m_incoming.size() - length, m_buffer.data(), length);
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
		handleRead();
	});
}

PeerNetwork::PeerNetwork(uint _networkId):
	m_ioService(),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), 0)),
	m_networkId(_networkId)
{
	m_ioService.run();
}

PeerNetwork::PeerNetwork(uint _networkId, uint _listenPort):
	m_ioService(),
	m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), _listenPort)),
	m_networkId(_networkId)
{
	try
	{
		start();
	}
	catch (exception& _e)
	{
		cerr << "Network error: " << _e.what() << endl;
		exit(1);
	}
}

PeerNetwork::~PeerNetwork()
{
}

bool PeerNetwork::connect(std::string const& _addr, uint _port)
{
	cout << "Connecting to " << _addr << " @" << dec << _port << endl;
	PeerConnection::pointer newConnection = PeerConnection::create(m_acceptor.get_io_service(), m_networkId);
	return newConnection->connect(_addr, _port, m_ioService);
}

void PeerNetwork::process()
{
	m_ioService.run();
}

void PeerNetwork::sync(BlockChain& _bc, TransactionQueue const& _tq)
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

void PeerNetwork::start()
{
	m_listener = new thread([&](){ justListen(); });
}

void PeerNetwork::justListen()
{
	bi::tcp::endpoint ep(bi::tcp::v4(), 30303);
	while (true)
	{
		PeerConnection::pointer newConnection = PeerConnection::create(m_acceptor.get_io_service(), m_networkId);
		cout << "Accepting incoming connections..." << endl;
		try
		{
			m_acceptor.accept(newConnection->socket(), ep);
			newConnection->start();
		}
		catch (exception& _e)
		{
			return;
		}
	}
}
#endif
