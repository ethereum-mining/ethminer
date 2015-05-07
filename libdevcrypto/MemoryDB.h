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
/** @file MemoryDB.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
#include <libdevcore/Common.h>
#include <libdevcore/FixedHash.h>
#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include "SHA3.h"

namespace dev
{

struct DBChannel: public LogChannel  { static const char* name(); static const int verbosity = 18; };
struct DBWarn: public LogChannel  { static const char* name(); static const int verbosity = 1; };

#define dbdebug clog(DBChannel)
#define dbwarn clog(DBWarn)

class MemoryDB
{
	friend class EnforceRefs;

public:
	MemoryDB() {}

	void clear() { m_main.clear(); }	// WARNING !!!! didn't originally clear m_refCount!!!
	std::map<h256, std::string> get() const;

	std::string lookup(h256 const& _h) const;
	bool exists(h256 const& _h) const;
	void insert(h256 const& _h, bytesConstRef _v);
	bool kill(h256 const& _h);
	void purge();

	bytes lookupAux(h256 const& _h) const { try { return m_aux.at(_h).first; } catch (...) { return bytes(); } }
	void removeAux(h256 const& _h) { m_aux[_h].second = false; }
	void insertAux(h256 const& _h, bytesConstRef _v) { m_aux[_h] = make_pair(_v.toBytes(), true); }

	std::set<h256> keys() const;

protected:
	std::map<h256, std::pair<std::string, unsigned>> m_main;
	std::map<h256, std::pair<bytes, bool>> m_aux;

	mutable bool m_enforceRefs = false;
};

class EnforceRefs
{
public:
	EnforceRefs(MemoryDB const& _o, bool _r): m_o(_o), m_r(_o.m_enforceRefs) { _o.m_enforceRefs = _r; }
	~EnforceRefs() { m_o.m_enforceRefs = m_r; }

private:
	MemoryDB const& m_o;
	bool m_r;
};

inline std::ostream& operator<<(std::ostream& _out, MemoryDB const& _m)
{
	for (auto const& i: _m.get())
	{
		_out << i.first << ": ";
		_out << RLP(i.second);
		_out << " " << toHex(i.second);
		_out << std::endl;
	}
	return _out;
}

}
