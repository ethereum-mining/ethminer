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
/** @file MessageFilter.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "MessageFilter.h"

#include <libethcore/SHA3.h>
#include "State.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

void MessageFilter::fillStream(RLPStream& _s) const
{
	_s.appendList(8) << m_from << m_to << m_stateAltered << m_altered << m_earliest << m_latest << m_max << m_skip;
}

h256 MessageFilter::sha3() const
{
	RLPStream s;
	fillStream(s);
	return dev::eth::sha3(s.out());
}

bool MessageFilter::matches(h256 _bloom) const
{
	auto have = [=](Address const& a) { return _bloom.contains(a.bloom()); };
	if (m_from.size())
	{
		for (auto i: m_from)
			if (have(i))
				goto OK1;
		return false;
	}
	OK1:
	if (m_to.size())
	{
		for (auto i: m_to)
			if (have(i))
				goto OK2;
		return false;
	}
	OK2:
	if (m_stateAltered.size() || m_altered.size())
	{
		for (auto i: m_altered)
			if (have(i))
				goto OK3;
		for (auto i: m_stateAltered)
			if (have(i.first) && _bloom.contains(h256(i.second).bloom()))
				goto OK3;
		return false;
	}
	OK3:
	return true;
}

bool MessageFilter::matches(State const& _s, unsigned _i) const
{
	h256 b = _s.changesFromPending(_i).bloom();
	if (!matches(b))
		return false;

	Transaction t = _s.pending()[_i];
	if (!m_to.empty() && !m_to.count(t.receiveAddress))
		return false;
	if (!m_from.empty() && !m_from.count(t.sender()))
		return false;
	if (m_stateAltered.empty() && m_altered.empty())
		return true;
	StateDiff d = _s.pendingDiff(_i);
	if (!m_altered.empty())
	{
		for (auto const& s: m_altered)
			if (d.accounts.count(s))
				return true;
		return false;
	}
	if (!m_stateAltered.empty())
	{
		for (auto const& s: m_stateAltered)
			if (d.accounts.count(s.first) && d.accounts.at(s.first).storage.count(s.second))
				return true;
		return false;
	}
	return true;
}

PastMessages MessageFilter::matches(Manifest const& _m, unsigned _i) const
{
	PastMessages ret;
	matches(_m, vector<unsigned>(1, _i), _m.from, PastMessages(), ret);
	return ret;
}

bool MessageFilter::matches(Manifest const& _m, vector<unsigned> _p, Address _o, PastMessages _limbo, PastMessages& o_ret) const
{
	bool ret;

	if ((m_from.empty() || m_from.count(_m.from)) && (m_to.empty() || m_to.count(_m.to)))
		_limbo.push_back(PastMessage(_m, _p, _o));

	// Handle limbos, by checking against all addresses in alteration.
	bool alters = m_altered.empty() && m_stateAltered.empty();
	alters = alters || m_altered.count(_m.from) || m_altered.count(_m.to);

	if (!alters)
		for (auto const& i: _m.altered)
			if (m_altered.count(_m.to) || m_stateAltered.count(make_pair(_m.to, i)))
			{
				alters = true;
				break;
			}
	// If we do alter stuff,
	if (alters)
	{
		o_ret += _limbo;
		_limbo.clear();
		ret = true;
	}

	_p.push_back(0);
	for (auto const& m: _m.internal)
	{
		if (matches(m, _p, _o, _limbo, o_ret))
		{
			_limbo.clear();
			ret = true;
		}
		_p.back()++;
	}

	return ret;
}
