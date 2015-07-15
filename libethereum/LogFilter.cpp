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
/** @file LogFilter.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "LogFilter.h"

#include <libdevcore/SHA3.h>
#include "State.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

std::ostream& dev::eth::operator<<(std::ostream& _out, LogFilter const& _s)
{
	// TODO
	_out << "(@" << _s.m_addresses << "#" << _s.m_topics << ">" << _s.m_earliest << "-" << _s.m_latest << "< )";
	return _out;
}

void LogFilter::streamRLP(RLPStream& _s) const
{
	_s.appendList(4) << m_addresses << m_topics << m_earliest << m_latest;
}

h256 LogFilter::sha3() const
{
	RLPStream s;
	streamRLP(s);
	return dev::sha3(s.out());
}

bool LogFilter::isRangeFilter() const
{
	if (m_addresses.size())
		return false;

	for (auto const& t: m_topics)
		if (t.size())
			return false;

	return true;
}

bool LogFilter::matches(LogBloom _bloom) const
{
	if (m_addresses.size())
	{
		for (auto const& i: m_addresses)
			if (_bloom.containsBloom<3>(dev::sha3(i)))
				goto OK1;
		return false;
	}
	OK1:
	for (auto const& t: m_topics)
		if (t.size())
		{
			for (auto const& i: t)
				if (_bloom.containsBloom<3>(dev::sha3(i)))
					goto OK2;
			return false;
			OK2:;
		}
	return true;
}

bool LogFilter::matches(State const& _s, unsigned _i) const
{
	return matches(_s.receipt(_i)).size() > 0;
}

vector<LogBloom> LogFilter::bloomPossibilities() const
{
	// return combination of each of the addresses/topics
	vector<LogBloom> ret;

	// | every address with every topic
	for (auto const& i: m_addresses)
	{
		// 1st case, there are addresses and topics
		//
		// m_addresses = [a0, a1];
		// m_topics = [[t0], [t1a, t1b], [], []];
		//
		// blooms = [
		// a0 | t0, a0 | t1a | t1b,
		// a1 | t0, a1 | t1a | t1b
		// ]
		//
		for (auto const& t: m_topics)
			if (t.size())
			{
				LogBloom b = LogBloom().shiftBloom<3>(dev::sha3(i));
				for (auto const &j: t)
					b = b.shiftBloom<3>(dev::sha3(j));
				ret.push_back(b);
			}
	}

	// 2nd case, there are no topics
	//
	// m_addresses = [a0, a1];
	// m_topics = [[t0], [t1a, t1b], [], []];
	//
	// blooms = [a0, a1];
	//
	if (!ret.size())
		for (auto const& i: m_addresses)
			ret.push_back(LogBloom().shiftBloom<3>(dev::sha3(i)));

	// 3rd case, there are no addresses, at least create blooms from topics
	//
	// m_addresses = [];
	// m_topics = [[t0], [t1a, t1b], [], []];
	//
	// blooms = [t0, t1a | t1b];
	//
	if (!m_addresses.size())
		for (auto const& t: m_topics)
			if (t.size())
			{
				LogBloom b;
				for (auto const &j: t)
					b = b.shiftBloom<3>(dev::sha3(j));
				ret.push_back(b);
			}

	return ret;
}

LogEntries LogFilter::matches(TransactionReceipt const& _m) const
{
	// there are no addresses or topics to filter
	if (isRangeFilter())
		return _m.log();

	LogEntries ret;
	if (matches(_m.bloom()))
		for (LogEntry const& e: _m.log())
		{
			if (!m_addresses.empty() && !m_addresses.count(e.address))
				goto continue2;
			for (unsigned i = 0; i < 4; ++i)
				if (!m_topics[i].empty() && (e.topics.size() < i || !m_topics[i].count(e.topics[i])))
					goto continue2;
			ret.push_back(e);
			continue2:;
		}
	return ret;
}
