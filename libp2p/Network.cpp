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
/** @file Network.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @author Eric Lombrozo <elombrozo@gmail.com> (Windows version of getInterfaceAddresses())
 * @date 2014
 */

#include <sys/types.h>
#ifndef _WIN32
#include <ifaddrs.h>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <libdevcore/Common.h>
#include <libdevcore/Assertions.h>
#include <libdevcore/CommonIO.h>
#include <libethcore/Exceptions.h>
#include "Common.h"
#include "UPnP.h"
#include "Network.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

std::set<bi::address> Network::getInterfaceAddresses()
{
	std::set<bi::address> addresses;

#ifdef _WIN32
	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0)
		BOOST_THROW_EXCEPTION(NoNetworking());

	char ac[80];
	if (gethostname(ac, sizeof(ac)) == SOCKET_ERROR)
	{
		clog(NetWarn) << "Error " << WSAGetLastError() << " when getting local host name.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}

	struct hostent* phe = gethostbyname(ac);
	if (phe == 0)
	{
		clog(NetWarn) << "Bad host lookup.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}

	for (int i = 0; phe->h_addr_list[i] != 0; ++i)
	{
		struct in_addr addr;
		memcpy(&addr, phe->h_addr_list[i], sizeof(struct in_addr));
		char *addrStr = inet_ntoa(addr);
		bi::address address(bi::address::from_string(addrStr));
		if (!isLocalHostAddress(address))
			addresses.insert(address.to_v4());
	}

	WSACleanup();
#else
	ifaddrs* ifaddr;
	if (getifaddrs(&ifaddr) == -1)
		BOOST_THROW_EXCEPTION(NoNetworking());

	for (auto ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
	{
		if (!ifa->ifa_addr || string(ifa->ifa_name) == "lo0" || !(ifa->ifa_flags & IFF_UP))
			continue;

		if (ifa->ifa_addr->sa_family == AF_INET)
		{
			in_addr addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
			boost::asio::ip::address_v4 address(boost::asio::detail::socket_ops::network_to_host_long(addr.s_addr));
			if (!isLocalHostAddress(address))
				addresses.insert(address);
		}
		else if (ifa->ifa_addr->sa_family == AF_INET6)
		{
			sockaddr_in6* sockaddr = ((struct sockaddr_in6 *)ifa->ifa_addr);
			in6_addr addr = sockaddr->sin6_addr;
			boost::asio::ip::address_v6::bytes_type bytes;
			memcpy(&bytes[0], addr.s6_addr, 16);
			boost::asio::ip::address_v6 address(bytes, sockaddr->sin6_scope_id);
			if (!isLocalHostAddress(address))
				addresses.insert(address);
		}
	}

	if (ifaddr!=NULL)
		freeifaddrs(ifaddr);

#endif

	return addresses;
}

int Network::tcp4Listen(bi::tcp::acceptor& _acceptor, NetworkPreferences const& _netPrefs)
{
	int retport = -1;
	if (_netPrefs.listenIPAddress.empty())
		for (unsigned i = 0; i < 2; ++i)
		{
			// try to connect w/listenPort, else attempt net-allocated port
			bi::tcp::endpoint endpoint(bi::tcp::v4(), i ? 0 : _netPrefs.listenPort);
			try
			{
				_acceptor.open(endpoint.protocol());
				_acceptor.set_option(ba::socket_base::reuse_address(true));
				_acceptor.bind(endpoint);
				_acceptor.listen();
				retport = _acceptor.local_endpoint().port();
				break;
			}
			catch (...)
			{
				if (i)
				{
					// both attempts failed
					cwarn << "Couldn't start accepting connections on host. Something very wrong with network?\n" << boost::current_exception_diagnostic_information();
				}
				
				// first attempt failed
				_acceptor.close();
				continue;
			}
		}
	else
	{
		bi::tcp::endpoint endpoint(bi::address::from_string(_netPrefs.listenIPAddress), _netPrefs.listenPort);
		try
		{
			_acceptor.open(endpoint.protocol());
			_acceptor.set_option(ba::socket_base::reuse_address(true));
			_acceptor.bind(endpoint);
			_acceptor.listen();
			retport = _acceptor.local_endpoint().port();
			assert(retport == _netPrefs.listenPort);
		}
		catch (...)
		{
			clog(NetWarn) << "Couldn't start accepting connections on host. Failed to accept socket.\n" << boost::current_exception_diagnostic_information();
		}
		return retport;
	}
	return retport;
}

bi::tcp::endpoint Network::traverseNAT(std::set<bi::address> const& _ifAddresses, unsigned short _listenPort, bi::address& o_upnpInterfaceAddr)
{
	asserts(_listenPort != 0);

	UPnP* upnp = nullptr;
	try
	{
		upnp = new UPnP;
	}
	// let m_upnp continue as null - we handle it properly.
	catch (...) {}

	bi::tcp::endpoint upnpEP;
	if (upnp && upnp->isValid())
	{
		bi::address pAddr;
		int extPort = 0;
		for (auto const& addr: _ifAddresses)
			if (addr.is_v4() && isPrivateAddress(addr) && (extPort = upnp->addRedirect(addr.to_string().c_str(), _listenPort)))
			{
				pAddr = addr;
				break;
			}

		auto eIP = upnp->externalIP();
		bi::address eIPAddr(bi::address::from_string(eIP));
		if (extPort && eIP != string("0.0.0.0") && !isPrivateAddress(eIPAddr))
		{
			clog(NetNote) << "Punched through NAT and mapped local port" << _listenPort << "onto external port" << extPort << ".";
			clog(NetNote) << "External addr:" << eIP;
			o_upnpInterfaceAddr = pAddr;
			upnpEP = bi::tcp::endpoint(eIPAddr, (unsigned short)extPort);
		}
		else
			clog(NetWarn) << "Couldn't punch through NAT (or no NAT in place).";

		if (upnp)
			delete upnp;
	}

	return upnpEP;
}

bi::tcp::endpoint Network::resolveHost(string const& _addr)
{
	static boost::asio::io_service s_resolverIoService;
	vector<string> split;
	boost::split(split, _addr, boost::is_any_of(":"));
	int givenPort;
	try
	{
		givenPort = stoi(split[1]);
	}
	catch (...)
	{
		clog(NetWarn) << "Error resolving host address..." << LogTag::Url << _addr << ". Could not find a port after ':'";
		return bi::tcp::endpoint();
	}
	unsigned port = split.size() > 1 ? givenPort : dev::p2p::c_defaultIPPort;

	boost::system::error_code ec;
	bi::address address = bi::address::from_string(split[0], ec);
	bi::tcp::endpoint ep(bi::address(), port);
	if (!ec)
		ep.address(address);
	else
	{
		boost::system::error_code ec;
		// resolve returns an iterator (host can resolve to multiple addresses)
		bi::tcp::resolver r(s_resolverIoService);
		auto it = r.resolve({bi::tcp::v4(), split[0], toString(port)}, ec);
		if (ec)
		{
			clog(NetWarn) << "Error resolving host address..." << LogTag::Url << _addr << ":" << LogTag::Error << ec.message();
			return bi::tcp::endpoint();
		}
		else
			ep = *it;
	}
	return ep;
}

