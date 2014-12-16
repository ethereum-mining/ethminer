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
/** @file Network.h
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

namespace dev
{
namespace p2p
{

struct NetworkPreferences
{
	NetworkPreferences(unsigned short p = 30303, std::string i = std::string(), bool u = true, bool l = false): listenPort(p), publicIP(i), upnp(u), localNetworking(l) {}

	unsigned short listenPort = 30303;
	std::string publicIP;
	bool upnp = true;
	bool localNetworking = false;
};

struct Packet
{
	bytes payload() const { return s.out(); }
	
	bool required = false;
	RLPStream s;
};
	
class SocketFace
{
	virtual void send(Packet const& _msg) = 0;
};
class SocketEventFace;
	
/**
 * @brief Generic Socket Interface
 * Owners of sockets must outlive the socket.
 * Boost ASIO uses lowercase template for udp/tcp, which is adopted here.
 */
template <class T>
class Socket: SocketFace, public std::enable_shared_from_this<Socket<T>>
{
public:
	using socketType = typename T::socket;
	using endpointType = typename T::endpoint;
	Socket(SocketEventFace* _seface);
	Socket(SocketEventFace* _seface, endpointType _endpoint);

protected:
	void send(Packet const& _msg)
	{
		if (!m_started)
			return;
		
		Guard l(x_sendQ);
		sendQ.push_back(_msg.payload());
		if (sendQ.size() == 1 && !m_stopped)
			doWrite();
	}
	
	void doWrite()
	{
		const bytes& bytes = sendQ[0];
		auto self(Socket<T>::shared_from_this());
//		boost::asio::async_write(m_socket, boost::asio::buffer(bytes), [this, self](boost::system::error_code _ec, std::size_t /*length*/)
//		{
//			if (_ec)
//				return stopWithError(_ec);
//			else
//			{
//				Guard l(x_sendQ);
//				sendQ.pop_front();
//				if (sendQ.empty())
//					return;
//			}
//			doWrite();
//		});
	}
	
	void stopWithError(boost::system::error_code _ec);
	
	std::atomic<bool> m_stopped;		///< Set when connection is stopping or stopped. Handshake cannot occur unless m_stopped is true.
	std::atomic<bool> m_started;		///< Atomically ensure connection is started once. Start cannot occur unless m_started is false. Managed by start() and shutdown(bool).
	
	SocketEventFace* m_eventDelegate = nullptr;
	
	Mutex x_sendQ;
	std::deque<bytes> sendQ;
	bytes recvBuffer;
	size_t recvdBytes = 0;
	socketType m_socket;
	
	Mutex x_socketError;				///< Mutex for error which can occur from host or IO thread.
	boost::system::error_code socketError;	///< Set when shut down due to error.
};

class SocketEventFace
{
public:
	virtual ba::io_service& ioService() = 0;
	virtual void onStopped(SocketFace*) = 0;
	virtual void onReceive(SocketFace*, Packet&) = 0;
};

struct UDPSocket: public Socket<bi::udp>
{
	UDPSocket(ba::io_service& _io, unsigned _port): Socket<bi::udp>(nullptr, bi::udp::endpoint(bi::udp::v4(), _port)) {}
	~UDPSocket() { boost::system::error_code ec; m_socket.shutdown(bi::udp::socket::shutdown_both, ec); m_socket.close(); }
	
//	bi::udp::socket m_socket;
};

/**
 * @brief Network Class
 * Static network operations and interface(s).
 */
class Network
{
public:
	/// @returns public and private interface addresses
	static std::vector<bi::address> getInterfaceAddresses();
	
	/// Try to bind and listen on _listenPort, else attempt net-allocated port.
	static int tcp4Listen(bi::tcp::acceptor& _acceptor, unsigned short _listenPort);

	/// Return public endpoint of upnp interface. If successful o_upnpifaddr will be a private interface address and endpoint will contain public address and port.
	static bi::tcp::endpoint traverseNAT(std::vector<bi::address> const& _ifAddresses, unsigned short _listenPort, bi::address& o_upnpifaddr);
};

}
}
