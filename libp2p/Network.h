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
#include <array>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

namespace dev
{
namespace p2p
{

static const unsigned short c_defaultListenPort = 30303;

struct NetworkPreferences
{
	// Default Network Preferences
	NetworkPreferences(unsigned short lp = 30303): listenPort(lp) {}
	
	// Network Preferences with specific Listen IP
	NetworkPreferences(std::string const& l, unsigned short lp = 30303, bool u = true): publicIPAddress(), listenIPAddress(l), listenPort(lp), traverseNAT(u) {}
	
	// Network Preferences with intended Public IP
	NetworkPreferences(std::string const& publicIP, std::string const& l = std::string(), unsigned short lp = 30303, bool u = true): publicIPAddress(publicIP), listenIPAddress(l), listenPort(lp), traverseNAT(u) { if (!publicIPAddress.empty() && !isPublicAddress(publicIPAddress)) BOOST_THROW_EXCEPTION(InvalidPublicIPAddress()); }

	/// Addressing
	
	std::string publicIPAddress;
	std::string listenIPAddress;
	unsigned short listenPort = c_defaultListenPort;
	
	
	/// Preferences
	
	bool traverseNAT = true;
	bool discovery = true;		// Discovery is activated with network.
	bool pin = false;			// Only accept or connect to trusted peers.
};

/**
 * @brief Network Class
 * Static network operations and interface(s).
 */
class Network
{
public:
	/// @returns public and private interface addresses
	static std::set<bi::address> getInterfaceAddresses();
	
	/// Try to bind and listen on _listenPort, else attempt net-allocated port.
	static int tcp4Listen(bi::tcp::acceptor& _acceptor, NetworkPreferences const& _netPrefs);

	/// Return public endpoint of upnp interface. If successful o_upnpifaddr will be a private interface address and endpoint will contain public address and port.
	static bi::tcp::endpoint traverseNAT(std::set<bi::address> const& _ifAddresses, unsigned short _listenPort, bi::address& o_upnpInterfaceAddr);
	
	/// Resolve "host:port" string as TCP endpoint. Returns unspecified endpoint on failure.
	static bi::tcp::endpoint resolveHost(std::string const& _host);
};
	
}
}
