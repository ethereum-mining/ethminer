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

#include <atomic>
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

/**
 * UDP Datagram
 * @todo make data protected/functional
 */
class UDPDatagram
{
public:
	UDPDatagram(bi::udp::endpoint const& _ep): locus(_ep) {}
	UDPDatagram(bi::udp::endpoint const& _ep, bytes _data): data(_data), locus(_ep) {}
	bi::udp::endpoint const& endpoint() const { return locus; }
	
	bytes data;
protected:
	bi::udp::endpoint locus;
};

/**
 * @brief RLPX Datagram which can be signed.
 */
struct RLPXDatagramFace: public UDPDatagram
{
	static uint64_t futureFromEpoch(std::chrono::milliseconds _ms) { return std::chrono::duration_cast<std::chrono::seconds>((std::chrono::system_clock::now() + _ms).time_since_epoch()).count(); }
	static uint64_t futureFromEpoch(std::chrono::seconds _sec) { return std::chrono::duration_cast<std::chrono::seconds>((std::chrono::system_clock::now() + _sec).time_since_epoch()).count(); }
	static Public authenticate(bytesConstRef _sig, bytesConstRef _rlp);
	
	virtual uint8_t packetType() = 0;
	RLPXDatagramFace(bi::udp::endpoint const& _ep): UDPDatagram(_ep) {}
	virtual h256 sign(Secret const& _from);

	virtual void streamRLP(RLPStream&) const = 0;
	virtual void interpretRLP(bytesConstRef _bytes) = 0;
};

template <class T>
struct RLPXDatagram: public RLPXDatagramFace
{
	RLPXDatagram(bi::udp::endpoint const& _ep): RLPXDatagramFace(_ep) {}
	static T fromBytesConstRef(bi::udp::endpoint const& _ep, bytesConstRef _bytes) { T t(_ep); t.interpretRLP(_bytes); return std::move(t); }
	uint8_t packetType() { return T::type; }
};

/**
 * @brief Interface which UDPSocket will implement.
 */
struct UDPSocketFace
{
	virtual bool send(UDPDatagram const& _msg) = 0;
	virtual void disconnect() = 0;
};

/**
 * @brief Interface which a UDPSocket's owner must implement.
 */
struct UDPSocketEvents
{
	virtual void onDisconnected(UDPSocketFace*) {};
	virtual void onReceived(UDPSocketFace*, bi::udp::endpoint const& _from, bytesConstRef _packetData) = 0;
};
	
/**
 * @brief UDP Interface
 * Handler must implement UDPSocketEvents.
 *
 * @todo multiple endpoints (we cannot advertise 0.0.0.0)
 * @todo decouple deque from UDPDatagram and add ref() to datagram for fire&forget
 */
template <typename Handler, unsigned MaxDatagramSize>
class UDPSocket: UDPSocketFace, public std::enable_shared_from_this<UDPSocket<Handler, MaxDatagramSize>>
{
public:
	enum { maxDatagramSize = MaxDatagramSize };
	static_assert(maxDatagramSize < 65507, "UDP datagrams cannot be larger than 65507 bytes");
	
	UDPSocket(ba::io_service& _io, UDPSocketEvents& _host, unsigned _port): m_host(_host), m_endpoint(bi::udp::v4(), _port), m_socket(_io) { m_started.store(false); m_closed.store(true); };
	virtual ~UDPSocket() { disconnect(); }

	/// Socket will begin listening for and delivering packets
	void connect();

	/// Send datagram.
	bool send(UDPDatagram const& _datagram);
	
	/// Returns if socket is open.
	bool isOpen() { return !m_closed; }

	/// Disconnect socket.
	void disconnect() { disconnectWithError(boost::asio::error::connection_reset); }
	
protected:
	void doRead();
	
	void doWrite();
	
	void disconnectWithError(boost::system::error_code _ec);

	std::atomic<bool> m_started;					///< Atomically ensure connection is started once. Start cannot occur unless m_started is false. Managed by start and disconnectWithError.
	std::atomic<bool> m_closed;					///< Connection availability.
	
	UDPSocketEvents& m_host;						///< Interface which owns this socket.
	bi::udp::endpoint m_endpoint;					///< Endpoint which we listen to.
	
	Mutex x_sendQ;
	std::deque<UDPDatagram> m_sendQ;				///< Queue for egress data.
	std::array<byte, maxDatagramSize> m_recvData;	///< Buffer for ingress data.
	bi::udp::endpoint m_recvEndpoint;				///< Endpoint data was received from.
	bi::udp::socket m_socket;						///< Boost asio udp socket.
	
	Mutex x_socketError;							///< Mutex for error which can be set from host or IO thread.
	boost::system::error_code m_socketError;		///< Set when shut down due to error.
};

template <typename Handler, unsigned MaxDatagramSize>
void UDPSocket<Handler, MaxDatagramSize>::connect()
{
	bool expect = false;
	if (!m_started.compare_exchange_strong(expect, true))
		return;
	
	m_socket.open(bi::udp::v4());
	m_socket.bind(m_endpoint);
	
	// clear write queue so reconnect doesn't send stale messages
	Guard l(x_sendQ);
	m_sendQ.clear();
	
	m_closed = false;
	doRead();
}
	
template <typename Handler, unsigned MaxDatagramSize>
bool UDPSocket<Handler, MaxDatagramSize>::send(UDPDatagram const& _datagram)
{
	if (m_closed)
		return false;
	
	Guard l(x_sendQ);
	m_sendQ.push_back(_datagram);
	if (m_sendQ.size() == 1)
		doWrite();
	
	return true;
}

template <typename Handler, unsigned MaxDatagramSize>
void UDPSocket<Handler, MaxDatagramSize>::doRead()
{
	if (m_closed)
		return;
	
	auto self(UDPSocket<Handler, MaxDatagramSize>::shared_from_this());
	m_socket.async_receive_from(boost::asio::buffer(m_recvData), m_recvEndpoint, [this, self](boost::system::error_code _ec, size_t _len)
	{
		if (_ec)
			return disconnectWithError(_ec);

		assert(_len);
		m_host.onReceived(this, m_recvEndpoint, bytesConstRef(m_recvData.data(), _len));
		doRead();
	});
}
	
template <typename Handler, unsigned MaxDatagramSize>
void UDPSocket<Handler, MaxDatagramSize>::doWrite()
{
	if (m_closed)
		return;
	
	const UDPDatagram& datagram = m_sendQ[0];
	auto self(UDPSocket<Handler, MaxDatagramSize>::shared_from_this());
	m_socket.async_send_to(boost::asio::buffer(datagram.data), datagram.endpoint(), [this, self](boost::system::error_code _ec, std::size_t)
	{
		if (_ec)
			return disconnectWithError(_ec);
		else
		{
			Guard l(x_sendQ);
			m_sendQ.pop_front();
			if (m_sendQ.empty())
				return;
		}
		doWrite();
	});
}

template <typename Handler, unsigned MaxDatagramSize>
void UDPSocket<Handler, MaxDatagramSize>::disconnectWithError(boost::system::error_code _ec)
{
	// If !started and already stopped, shutdown has already occured. (EOF or Operation canceled)
	if (!m_started && m_closed && !m_socket.is_open() /* todo: veirfy this logic*/)
		return;

	assert(_ec);
	{
		// disconnect-operation following prior non-zero errors are ignored
		Guard l(x_socketError);
		if (m_socketError != boost::system::error_code())
			return;
		m_socketError = _ec;
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
	
}
}