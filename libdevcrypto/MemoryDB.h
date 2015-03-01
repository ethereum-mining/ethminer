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

struct DBChannel: public LogChannel  { static const char* name() { return "TDB"; } static const int verbosity = 18; };

#define dbdebug clog(DBChannel)

class MemoryDB
{
	friend class EnforceRefs;

public:
	MemoryDB() {}

	void clear() { m_over.clear(); }
	std::map<h256, std::string> get() const;

	std::string lookup(h256 _h) const;
	bool exists(h256 _h) const;
	void insert(h256 _h, bytesConstRef _v);
	bool kill(h256 _h);
	void purge();

	bytes lookupAux(h256 _h) const { return asBytes(lookup(h256(sha3(_h).ref().cropped(16), h256::AlignRight))); }
	void insertAux(h256 _h, bytesConstRef _v) { return insert(h256(sha3(_h).ref().cropped(16), h256::AlignRight), _v); }

	std::set<h256> keys() const;

protected:
	static h256 aux(h256 _k) { return h256(sha3(_k).ref().cropped(0, 24), h256::AlignLeft); }

	std::map<h256, std::string> m_over;
	std::map<h256, unsigned> m_refCount;
	std::set<h256> m_auxActive;
	std::map<h256, bytes> m_aux;

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
	for (auto i: _m.get())
	{
		_out << i.first << ": ";
		_out << RLP(i.second);
		_out << " " << toHex(i.second);
		_out << std::endl;
	}
	return _out;
}

}
