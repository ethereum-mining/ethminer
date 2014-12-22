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
/** @file UDP.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <array>
#include <libdevcore/Guards.h>
#include <libdevcrypto/Common.h>
#include <libdevcrypto/SHA3.h>
#include <libdevcore/RLP.h>
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

namespace dev
{
namespace p2p
{

struct UDPDatagram
{
	UDPDatagram() = default;
	UDPDatagram(bi::udp::endpoint _ep, bytes _data): to(_ep), data(std::move(_data)) {}
	bi::udp::endpoint to;

	bytes data;
};
	
struct RLPDatagram: UDPDatagram
{
	void seal(Secret const& _k)
	{
		RLPStream packet;
		streamRLP(packet);
		bytes b(packet.out());
		Signature sig = dev::sign(_k, dev::sha3(b));
		data.resize(data.size() + Signature::size);
		sig.ref().copyTo(&data);
		memcpy(data.data()+sizeof(Signature),b.data(),b.size());
	}
	
protected:
	virtual void streamRLP(RLPStream& _s) const {};
};

struct UDPSocketFace
{
	virtual bool send(UDPDatagram const& _msg) = 0;
	virtual void disconnect() = 0;
};

struct UDPSocketEvents
{
	virtual void onDisconnected(UDPSocketFace*) {};
	virtual void onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packetData) = 0;
};
	
/**
 * @brief UDP Interface
 * Handler must implement UDPSocketEvents.
 */
template <typename Handler, unsigned MaxDatagramSize>
class UDPSocket: UDPSocketFace, public std::enable_shared_from_this<UDPSocket<Handler, MaxDatagramSize>>
{
public:
	static constexpr unsigned maxDatagramSize = MaxDatagramSize;
	static_assert(maxDatagramSize < 65507, "UDP datagrams cannot be larger than 65507 bytes");
	
	/// Construct open socket to endpoint.
	UDPSocket(ba::io_service& _io, UDPSocketEvents& _host, unsigned _port): m_host(_host), m_endpoint(bi::udp::v4(), _port), m_socket(_io) { m_started.store(false); m_closed.store(true); };
	virtual ~UDPSocket() { disconnect(); }

	/// Socket will begin listening for and delivering packets
	void connect()
	{
		bool expect = false;
		if (!m_started.compare_exchange_strong(expect, true))
			return;
		
		m_socket.open(bi::udp::v4());
		m_socket.bind(m_endpoint);

		// clear write queue so reconnect doesn't send stale messages
		Guard l(x_sendQ);
		sendQ.clear();
		
		m_closed = false;
		doRead();
	}

	bool send(UDPDatagram const& _datagram)
	{
		if (m_closed)
			return false;
		
		Guard l(x_sendQ);
		sendQ.push_back(_datagram);
		if (sendQ.size() == 1)
			doWrite();
		
		return true;
	}
	
	bool isOpen() { return !m_closed; }

	void disconnect() { disconnectWithError(boost::asio::error::connection_reset); }
	
protected:
	void doRead()
	{
		auto self(UDPSocket<Handler, MaxDatagramSize>::shared_from_this());
		m_socket.async_receive_from(boost::asio::buffer(recvData), recvEndpoint, [this, self](boost::system::error_code _ec, size_t _len)
		{
			if (_ec)
				return disconnectWithError(_ec);

			assert(_len);
			m_host.onReceived(this, recvEndpoint, bytesConstRef(recvData.data(), _len));
			if (!m_closed)
				doRead();
		});
	}
	
	void doWrite()
	{
		const UDPDatagram& datagram = sendQ[0];
		auto self(UDPSocket<Handler, MaxDatagramSize>::shared_from_this());
		m_socket.async_send_to(boost::asio::buffer(datagram.data), datagram.to, [this, self](boost::system::error_code _ec, std::size_t)
		{
			if (_ec)
				return disconnectWithError(_ec);
			else
			{
				Guard l(x_sendQ);
				sendQ.pop_front();
				if (sendQ.empty())
					return;
			}
			doWrite();
		});
	}
	
	void disconnectWithError(boost::system::error_code _ec)
	{
		// If !started and already stopped, shutdown has already occured. (EOF or Operation canceled)
		if (!m_started && m_closed && !m_socket.is_open() /* todo: veirfy this logic*/)
			return;

		assert(_ec);
		{
			// disconnect-operation following prior non-zero errors are ignored
			Guard l(x_socketError);
			if (socketError != boost::system::error_code())
				return;
			socketError = _ec;
		}
		// TODO: (if non-zero error) schedule high-priority writes

		// prevent concurrent disconnect
		bool expected = true;
		if (!m_started.compare_exchange_strong(expected, false))
			return;
		
		// set m_closed to true to prevent undeliverable egress messages
		bool wasClosed = m_closed;
		m_closed = true;
		
		// close sockets
		boost::system::error_code ec;
		m_socket.shutdown(bi::udp::socket::shutdown_both, ec);
		m_socket.close();

		// socket never started if it never left stopped-state (pre-handshake)
		if (wasClosed)
			return;

		m_host.onDisconnected(this);
	}

	std::atomic<bool> m_closed;		///< Set when connection is stopping or stopped. Handshake cannot occur unless m_closed is true.
	std::atomic<bool> m_started;		///< Atomically ensure connection is started once. Start cannot occur unless m_started is false. Managed by start and disconnectWithError.
	
	UDPSocketEvents& m_host;					///< Interface which owns this socket.
	bi::udp::endpoint m_endpoint;			///< Endpoint which we listen to.
	
	Mutex x_sendQ;
	std::deque<UDPDatagram> sendQ;
	std::array<byte, maxDatagramSize> recvData;		///< Buffer for ingress datagrams.
	bi::udp::endpoint recvEndpoint;			///< Endpoint data was received from.
	bi::udp::socket m_socket;
	
	Mutex x_socketError;				///< Mutex for error which can occur from host or IO thread.
	boost::system::error_code socketError;	///< Set when shut down due to error.
};
	
}
}