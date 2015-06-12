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
/** @file BloomFilter.h
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date June 2015
*/

#pragma once

#include "Common.h"

namespace dev
{
namespace shh
{

class BloomFilter
{
public:
	virtual ~BloomFilter() {}
	BloomFilter(): m_filter(0) {}
	BloomFilter(unsigned _i): m_filter(_i) {}
	BloomFilter(AbridgedTopic const& _t): m_filter(AbridgedTopic::Arith(_t).convert_to<unsigned>()) {}

	unsigned getUnsigned() const { return m_filter; }
	AbridgedTopic getAbridgedTopic() const { return AbridgedTopic(m_filter); }

	bool matches(AbridgedTopic const& _t) const;
	virtual void add(Topic const& _t) { add(abridge(_t)); }
	virtual void add(Topics const& _topics) { for (Topic t : _topics) add(abridge(t)); }
	virtual void add(AbridgedTopic const& _t) { m_filter |= AbridgedTopic::Arith(_t).convert_to<unsigned>(); }
	virtual void remove(AbridgedTopic const&) {} // not implemented in this class, use derived class instead.

protected:
	unsigned m_filter;
};

class SharedBloomFilter: public BloomFilter
{
public:
	virtual ~SharedBloomFilter() {}
	SharedBloomFilter() { init(); }
	SharedBloomFilter(unsigned _i): BloomFilter(_i) { init(); }
	SharedBloomFilter(AbridgedTopic const& _t): BloomFilter(_t) { init(); }

	void add(AbridgedTopic const& _t) override;
	void remove(AbridgedTopic const& _t) override;

protected:
	void init() { for (unsigned i = 0; i < ArrSize; ++i) m_refCounter[i] = 0; }

private:
	enum { ArrSize = 8 * AbridgedTopic::size };
	unsigned short m_refCounter[ArrSize];
};

}
}






