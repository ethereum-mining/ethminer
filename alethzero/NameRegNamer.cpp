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
/** @file NameRegNamer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "NameRegNamer.h"
#include <libdevcore/Log.h>
#include <libethereum/Client.h>
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(NameRegNamer);

NameRegNamer::NameRegNamer(MainFace* _m):
	AccountNamerPlugin(_m, "NameRegNamer")
{
}

NameRegNamer::~NameRegNamer()
{
}

string NameRegNamer::toName(Address const& _a) const
{
	for (auto const& r: m_registrars)
	{
		string n = abiOut<string>(main()->ethereum()->call(r, abiIn("name(address)", _a)).output);
		if (!n.empty())
			return n;
	}
	return string();
}

Address NameRegNamer::toAddress(std::string const& _n) const
{
	for (auto const& r: m_registrars)
		if (Address a = abiOut<Address>(main()->ethereum()->call(r, abiIn("addr(string)", _n)).output))
			return a;
	return Address();
}

Addresses NameRegNamer::knownAddresses() const
{
	return m_knownCache;
}

void NameRegNamer::killRegistrar(Address const& _r)
{
	if (m_filters.count(_r))
	{
		main()->uninstallWatch(m_filters.at(_r));
		m_filters.erase(_r);
	}
	for (auto i = m_registrars.begin(); i != m_registrars.end();)
		if (*i == _r)
			i = m_registrars.erase(i);
		else
			++i;
}

void NameRegNamer::updateCache()
{
//	m_forwardCache.clear();
//	m_reverseCache.clear();
	m_knownCache.clear();
#if ETH_FATDB || !ETH_TRUE
	for (auto const& r: m_registrars)
		for (u256 const& a: keysOf(ethereum()->storageAt(r)))
			if (a < u256(1) << 160)
				m_knownCache.push_back(Address((u160)a - 1));
#endif
}

void NameRegNamer::readSettings(QSettings const& _s)
{
	(void)_s;
	while (!m_registrars.empty())
		killRegistrar(m_registrars.back());

	Address a("96d76ae3397b52d9f61215270df65d72358709e3");
	m_filters[a] = main()->installWatch(LogFilter().address(a), [=](LocalisedLogEntries const&){ updateCache(); });

	noteKnownChanged();
}

void NameRegNamer::writeSettings(QSettings&)
{
}
