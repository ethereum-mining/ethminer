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
/** @file UPnP.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <stdio.h>
#include <string.h>
#include <miniupnpc/miniwget.h>
#include <miniupnpc/miniupnpc.h>
#include <miniupnpc/upnpcommands.h>
#include "Common.h"
#include "Log.h"
#include "Exceptions.h"
#include "UPnP.h"
using namespace std;
using namespace eth;

UPnP::UPnP()
{
	m_urls.reset(new UPNPUrls);
	m_data.reset(new IGDdatas);

	m_ok = false;

	struct UPNPDev* devlist;
	struct UPNPDev* dev;
	char* descXML;
	int descXMLsize = 0;
	int upnperror = 0;
	memset(m_urls.get(), 0, sizeof(struct UPNPUrls));
	memset(m_data.get(), 0, sizeof(struct IGDdatas));
	devlist = upnpDiscover(2000, NULL/*multicast interface*/, NULL/*minissdpd socket path*/, 0/*sameport*/, 0/*ipv6*/, &upnperror);
	if (devlist)
	{
		dev = devlist;
		while (dev)
		{
			if (strstr (dev->st, "InternetGatewayDevice"))
				break;
			dev = dev->pNext;
		}
		if (!dev)
			dev = devlist; /* defaulting to first device */

		cnote << "UPnP device:" << dev->descURL << "[st:" << dev->st << "]";
#if MINIUPNPC_API_VERSION >= 9
		descXML = (char*)miniwget(dev->descURL, &descXMLsize, 0);
#else
		descXML = (char*)miniwget(dev->descURL, &descXMLsize);
#endif
		if (descXML)
		{
			parserootdesc (descXML, descXMLsize, m_data.get());
			free (descXML); descXML = 0;
#if MINIUPNPC_API_VERSION >= 9
			GetUPNPUrls (m_urls.get(), m_data.get(), dev->descURL, 0);
#else
			GetUPNPUrls (m_urls.get(), m_data.get(), dev->descURL);
#endif
			m_ok = true;
		}
		freeUPNPDevlist(devlist);
	}
	else
	{
		cnote << "UPnP device not found.";
		throw NoUPnPDevice();
	}
}

UPnP::~UPnP()
{
	auto r = m_reg;
	for (auto i: r)
		removeRedirect(i);
}

string UPnP::externalIP()
{
	char addr[16];
	if (!UPNP_GetExternalIPAddress(m_urls->controlURL, m_data->first.servicetype, addr))
		return addr;
	else
		return "0.0.0.0";
}

int UPnP::addRedirect(char const* addr, int port)
{
	if (m_urls->controlURL[0] == '\0')
	{
		cwarn << "UPnP::addRedirect() called without proper initialisation?";
		return -1;
	}

	// Try direct mapping first (port external, port internal).
	char port_str[16];
	sprintf(port_str, "%d", port);
	if (!UPNP_AddPortMapping(m_urls->controlURL, m_data->first.servicetype, port_str, port_str, addr, "ethereum", "TCP", NULL, NULL))
		return port;

	// Failed - now try (random external, port internal) and cycle up to 10 times.
	for (uint i = 0; i < 10; ++i)
	{
		port = rand() % 65535 - 1024 + 1024;
		sprintf(port_str, "%d", port);
		if (!UPNP_AddPortMapping(m_urls->controlURL, m_data->first.servicetype, NULL, port_str, addr, "ethereum", "TCP", NULL, NULL))
			return port;
	}

	// Failed. Try asking the router to give us a free external port.
	if (UPNP_AddPortMapping(m_urls->controlURL, m_data->first.servicetype, port_str, NULL, addr, "ethereum", "TCP", NULL, NULL))
		// Failed. Exit.
		return 0;

	// We got mapped, but we don't know which ports we got mapped to. Now to find...
	unsigned num = 0;
	UPNP_GetPortMappingNumberOfEntries(m_urls->controlURL, m_data->first.servicetype, &num);
	for (unsigned i = 0; i < num; ++i)
	{
		char extPort[16];
		char intClient[16];
		char intPort[6];
		char protocol[4];
		char desc[80];
		char enabled[4];
		char rHost[64];
		char duration[16];
		UPNP_GetGenericPortMappingEntry(m_urls->controlURL, m_data->first.servicetype, toString(i).c_str(), extPort, intClient, intPort, protocol, desc, enabled, rHost, duration);
		if (string("ethereum") == desc)
		{
			m_reg.insert(atoi(extPort));
			return atoi(extPort);
		}
	}
	cerr << "ERROR: Mapped port not found." << endl;
	return 0;
}

void UPnP::removeRedirect(int port)
{
	char port_str[16];
//		int t;
	printf("TB : upnp_rem_redir (%d)\n", port);
	if (m_urls->controlURL[0] == '\0')
	{
		printf("TB : the init was not done !\n");
		return;
	}
	sprintf(port_str, "%d", port);
	UPNP_DeletePortMapping(m_urls->controlURL, m_data->first.servicetype, port_str, "TCP", NULL);
	m_reg.erase(port);
}
