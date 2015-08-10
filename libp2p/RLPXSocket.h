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
/** @file RLPXSocket.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#pragma once

#include "Common.h"

namespace dev
{
namespace p2p
{

/**
 * @brief Shared pointer wrapper for ASIO TCP socket.
 *
 * Thread Safety
 * Distinct Objects: Safe.
 * Shared objects: Unsafe.
 * * an instance method must not be called concurrently
 */
class RLPXSocket: public std::enable_shared_from_this<RLPXSocket>
{
public:
	RLPXSocket(ba::io_service& _ioService): m_socket(_ioService) {}
	~RLPXSocket() { close(); }
	
	bool isConnected() const { return m_socket.is_open(); }
	void close() { try { boost::system::error_code ec; m_socket.shutdown(bi::tcp::socket::shutdown_both, ec); if (m_socket.is_open()) m_socket.close(); } catch (...){} }
	bi::tcp::endpoint remoteEndpoint() { boost::system::error_code ec; return m_socket.remote_endpoint(ec); }
	bi::tcp::socket& ref() { return m_socket; }
	
protected:
	bi::tcp::socket m_socket;
};

}
}
