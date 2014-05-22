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
/** @file TrieDB.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "TrieDB.h"
using namespace std;
using namespace eth;

#define tdebug ndebug

namespace eth
{

const h256 c_shaNull = sha3(rlp(""));

std::string BasicMap::lookup(h256 _h) const
{
	auto it = m_over.find(_h);
	if (it != m_over.end())
	{
		if (!m_enforceRefs || (m_refCount.count(it->first) && m_refCount.at(it->first)))
			return it->second;
		else if (m_enforceRefs && m_refCount.count(it->first) && !m_refCount.at(it->first))
			cwarn << "XXX Lookup required for value with no refs:" << _h.abridged();
	}
	return std::string();
}

void BasicMap::insert(h256 _h, bytesConstRef _v)
{
	m_over[_h] = _v.toString();
	m_refCount[_h]++;
	tdebug << "INST" << _h.abridged() << "=>" << m_refCount[_h];
}

void BasicMap::kill(h256 _h)
{
	if (m_refCount.count(_h))
	{
		if (m_refCount[_h] > 0)
			--m_refCount[_h];
		else
			cwarn << "Decreasing DB node ref count below zero. Probably have a corrupt Trie.";
	}
	tdebug << "KILL" << _h.abridged() << "=>" << m_refCount[_h];
}

void BasicMap::purge()
{
	for (auto const& i: m_refCount)
		if (!i.second)
			m_over.erase(i.first);
}

Overlay::~Overlay()
{
	if (m_db.use_count() == 1 && m_db.get())
		cnote << "Closing state DB";
}

void Overlay::setDB(ldb::DB* _db, bool _clearOverlay)
{
	m_db = std::shared_ptr<ldb::DB>(_db);
	if (_clearOverlay)
		m_over.clear();
}

void Overlay::commit()
{
	if (m_db)
	{
//		cnote << "Committing nodes to disk DB:";
		for (auto const& i: m_over)
		{
//			cnote << i.first << "#" << m_refCount[i.first];
			if (m_refCount[i.first])
				m_db->Put(m_writeOptions, ldb::Slice((char const*)i.first.data(), i.first.size), ldb::Slice(i.second.data(), i.second.size()));
		}
		m_over.clear();
		m_refCount.clear();
	}
}

void Overlay::rollback()
{
	m_over.clear();
	m_refCount.clear();
}

std::string Overlay::lookup(h256 _h) const
{
	std::string ret = BasicMap::lookup(_h);
	if (ret.empty() && m_db)
		m_db->Get(m_readOptions, ldb::Slice((char const*)_h.data(), 32), &ret);
	return ret;
}

}
