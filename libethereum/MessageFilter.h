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
/** @file MessageFilter.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libethential/Common.h>
#include <libethential/RLP.h>
#include <libethcore/CommonEth.h>
#include "PastMessage.h"

namespace eth
{

struct Manifest;
class State;

class MessageFilter
{
public:
	MessageFilter(int _earliest = 0, int _latest = -1, unsigned _max = 10, unsigned _skip = 0): m_earliest(_earliest), m_latest(_latest), m_max(_max), m_skip(_skip) {}

	void fillStream(RLPStream& _s) const;
	h256 sha3() const;

	int earliest() const { return m_earliest; }
	int latest() const { return m_latest; }
	unsigned max() const { return m_max; }
	unsigned skip() const { return m_skip; }
	bool matches(h256 _bloom) const;
	bool matches(State const& _s, unsigned _i) const;
	PastMessages matches(Manifest const& _m, unsigned _i) const;

	MessageFilter from(Address _a) { m_from.insert(_a); return *this; }
	MessageFilter to(Address _a) { m_to.insert(_a); return *this; }
	MessageFilter altered(Address _a, u256 _l) { m_stateAltered.insert(std::make_pair(_a, _l)); return *this; }
	MessageFilter altered(Address _a) { m_altered.insert(_a); return *this; }
	MessageFilter withMax(unsigned _m) { m_max = _m; return *this; }
	MessageFilter withSkip(unsigned _m) { m_skip = _m; return *this; }
	MessageFilter withEarliest(int _e) { m_earliest = _e; return *this; }
	MessageFilter withLatest(int _e) { m_latest = _e; return *this; }

private:
	bool matches(Manifest const& _m, std::vector<unsigned> _p, Address _o, PastMessages _limbo, PastMessages& o_ret) const;

	std::set<Address> m_from;
	std::set<Address> m_to;
	std::set<std::pair<Address, u256>> m_stateAltered;
	std::set<Address> m_altered;
	int m_earliest = 0;
	int m_latest = -1;
	unsigned m_max;
	unsigned m_skip;
};

}
