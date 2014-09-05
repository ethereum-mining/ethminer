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
using namespace dev::eth;

namespace dev
{
namespace eth
{

std::map<h256, std::string> MemoryDB::get() const
{
	if (!m_enforceRefs)
		return m_over;
	std::map<h256, std::string> ret;
	for (auto const& i: m_refCount)
		if (i.second)
			ret.insert(*m_over.find(i.first));
	return ret;
}

std::string MemoryDB::lookup(h256 _h) const
{
	auto it = m_over.find(_h);
	if (it != m_over.end())
	{
		if (!m_enforceRefs || (m_refCount.count(it->first) && m_refCount.at(it->first)))
			return it->second;
//		else if (m_enforceRefs && m_refCount.count(it->first) && !m_refCount.at(it->first))
//			cnote << "Lookup required for value with no refs. Let's hope it's in the DB." << _h.abridged();
	}
	return std::string();
}

bool MemoryDB::exists(h256 _h) const
{
	auto it = m_over.find(_h);
	if (it != m_over.end() && (!m_enforceRefs || (m_refCount.count(it->first) && m_refCount.at(it->first))))
		return true;
	return false;
}

void MemoryDB::insert(h256 _h, bytesConstRef _v)
{
	m_over[_h] = _v.toString();
	m_refCount[_h]++;
#if ETH_PARANOIA
	dbdebug << "INST" << _h.abridged() << "=>" << m_refCount[_h];
#endif
}

bool MemoryDB::kill(h256 _h)
{
	if (m_refCount.count(_h))
	{
		if (m_refCount[_h] > 0)
			--m_refCount[_h];
#if ETH_PARANOIA
		else
		{
			// If we get to this point, then there was probably a node in the level DB which we need to remove and which we have previously
			// used as part of the memory-based MemoryDB. Nothing to be worried about *as long as the node exists in the DB*.
			dbdebug << "NOKILL-WAS" << _h.abridged();
			return false;
		}
		dbdebug << "KILL" << _h.abridged() << "=>" << m_refCount[_h];
		return true;
	}
	else
	{
		dbdebug << "NOKILL" << _h.abridged();
		return false;
	}
#else
	}
	return true;
#endif
}

void MemoryDB::purge()
{
	for (auto const& i: m_refCount)
		if (!i.second)
			m_over.erase(i.first);
}

set<h256> MemoryDB::keys() const
{
	set<h256> ret;
	for (auto const& i: m_refCount)
		if (i.second)
			ret.insert(i.first);
	return ret;
}

}
}
