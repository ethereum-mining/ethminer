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
/** @file PeerNetwork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <thread>
#include "RLP.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace eth
{

class BlockChain;
class TransactionQueue;

enum PacketType
{
	Hello = 0,
	Disconnect,
	Ping,
	Pong,
	GetPeers = 0x10,
	Peers,
	Transactions,
	Blocks,
	GetChain
};

class PeerSession: public std::enable_shared_from_this<PeerSession>
{
public:
	PeerSession(bi::tcp::socket _socket, uint _rNId): m_socket(std::move(_socket)), m_reqNetworkId(_rNId)
	{
	}

	void start()
	{
		doRead();
	}

	bool interpret(RLP const& _r)
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
			break;
		}
		return true;
	}

	void disconnect()
	{
		RLPStream s;
		prep(s);
		s.appendList(1) << Disconnect;
		sealAndSend(s);
		sleep(1);
	}

private:
	void doRead()
	{
		auto self(shared_from_this());
		m_socket.async_read_some(boost::asio::buffer(m_data), [this, self](boost::system::error_code ec, std::size_t length)
		{
			if (!ec)
				doWrite(length);
		});
	}

	void doWrite(std::size_t length)
	{
		auto self(shared_from_this());
		boost::asio::async_write(m_socket, boost::asio::buffer(m_data, length), [this, self](boost::system::error_code ec, std::size_t /*length*/)
		{
			if (!ec)
				doRead();
		});
	}

	void prep(RLPStream& _s)
	{
		_s.appendRaw(bytes(8, 0));
	}

	void sealAndSend(RLPStream& _s)
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

	void send(bytes& _msg)
	{
		bytes* buffer = new bytes;
		swap(*buffer, _msg);
		ba::async_write(m_socket, ba::buffer(*buffer), [&](boost::system::error_code ec, std::size_t length)
		{
			if (!ec)
				// Callback for how the write went. For now, just kill the buffer.
				delete buffer;
		});
	}

	bi::tcp::socket m_socket;
	std::array<byte, 1024> m_data;

	bytes m_incoming;
	std::string m_clientVersion;
	uint m_protocolVersion;
	uint m_networkId;
	uint m_reqNetworkId;
};

class PeerServer
{
public:
	PeerServer(uint _networkId, short _port):
		m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), _port)),
		m_socket(m_ioService),
		m_requiredNetworkId(_networkId)
	{
		doAccept();
	}

	PeerServer(uint _networkId):
		m_acceptor(m_ioService, bi::tcp::endpoint(bi::tcp::v4(), 0)),
		m_socket(m_ioService),
		m_requiredNetworkId(_networkId)
	{
	}

	/// Conduct I/O, polling, syncing, whatever.
	/// Ideally all time-consuming I/O is done in a background thread, but you get this call every 100ms or so anyway.
	void run() { m_ioService.run(); }
	void process() { m_ioService.poll(); }

	bool connect(std::string const& _addr = "127.0.0.1", uint _port = 30303);

	/// Sync with the BlockChain. It might contain one of our mined blocks, we might have new candidates from the network.
	void sync(BlockChain& _bc, TransactionQueue const&) {}

	/// Get an incoming transaction from the queue. @returns bytes() if nothing waiting.
	bytes const& incomingTransaction() { return NullBytes; }

	/// Remove incoming transaction from the queue. Make sure you've finished with the data from any previous incomingTransaction() calls.
	void popIncomingTransaction() {}

private:
	void doAccept()
	{
		m_acceptor.async_accept(m_socket, [&](boost::system::error_code ec)
		{
			if (!ec)
			{
				auto p = std::make_shared<PeerSession>(std::move(m_socket), m_requiredNetworkId);
				m_peers.push_back(p);
				p->start();
			}
			doAccept();
		});
	}

	ba::io_service m_ioService;
	bi::tcp::acceptor m_acceptor;
	bi::tcp::socket m_socket;

	uint m_requiredNetworkId;
	std::vector<std::weak_ptr<PeerSession>> m_peers;
};


}
