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
/** @file MemoryDB.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/Common.h>
#include "MemoryDB.h"
using namespace std;
using namespace dev;

namespace dev
{

const char* DBChannel::name() { return "TDB"; }
const char* DBWarn::name() { return "TDB"; }

std::unordered_map<h256, std::string> MemoryDB::get() const
{
	std::unordered_map<h256, std::string> ret;
	for (auto const& i: m_main)
		if (!m_enforceRefs || i.second.second > 0)
			ret.insert(make_pair(i.first, i.second.first));
	return ret;
}

std::string MemoryDB::lookup(h256 const& _h) const
{
	auto it = m_main.find(_h);
	if (it != m_main.end())
	{
		if (!m_enforceRefs || it->second.second > 0)
			return it->second.first;
//		else if (m_enforceRefs && m_refCount.count(it->first) && !m_refCount.at(it->first))
//			cnote << "Lookup required for value with no refs. Let's hope it's in the DB." << _h;
	}
	return std::string();
}

bool MemoryDB::exists(h256 const& _h) const
{
	auto it = m_main.find(_h);
	if (it != m_main.end() && (!m_enforceRefs || it->second.second > 0))
		return true;
	return false;
}

void MemoryDB::insert(h256 const& _h, bytesConstRef _v)
{
	auto it = m_main.find(_h);
	if (it != m_main.end())
	{
		it->second.first = _v.toString();
		it->second.second++;
	}
	else
		m_main[_h] = make_pair(_v.toString(), 1);
#if ETH_PARANOIA
	dbdebug << "INST" << _h << "=>" << m_main[_h].second;
#endif
}

bool MemoryDB::kill(h256 const& _h)
{
	if (m_main.count(_h))
	{
		if (m_main[_h].second > 0)
			m_main[_h].second--;
#if ETH_PARANOIA
		else
		{
			// If we get to this point, then there was probably a node in the level DB which we need to remove and which we have previously
			// used as part of the memory-based MemoryDB. Nothing to be worried about *as long as the node exists in the DB*.
			dbdebug << "NOKILL-WAS" << _h;
			return false;
		}
		dbdebug << "KILL" << _h << "=>" << m_main[_h].second;
		return true;
	}
	else
	{
		dbdebug << "NOKILL" << _h;
		return false;
	}
#else
	}
	return true;
#endif
}

void MemoryDB::purge()
{
	for (auto it = m_main.begin(); it != m_main.end(); )
		if (it->second.second)
			++it;
		else
			it = m_main.erase(it);
}

h256Hash MemoryDB::keys() const
{
	h256Hash ret;
	for (auto const& i: m_main)
		if (i.second.second)
			ret.insert(i.first);
	return ret;
}

}
