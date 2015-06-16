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

class TopicBloomFilter: public AbridgedTopic
{
public:
	TopicBloomFilter() { init(); }
	TopicBloomFilter(AbridgedTopic const& _h): AbridgedTopic(_h) { init(); }

	void addBloom(AbridgedTopic const& _h) { addRaw(_h.template bloomPart<BitsPerBloom, 4>()); }
	void removeBloom(AbridgedTopic const& _h) { removeRaw(_h.template bloomPart<BitsPerBloom, 4>()); }
	bool containsBloom(AbridgedTopic const& _h) const { return contains(_h.template bloomPart<BitsPerBloom, 4>()); }

	void addRaw(AbridgedTopic const& _h);
	void removeRaw(AbridgedTopic const& _h);
	bool containsRaw(AbridgedTopic const& _h) const { return contains(_h); }

	enum { BitsPerBloom = 3 };
	
private:
	void init() { for (unsigned i = 0; i < CounterSize; ++i) m_refCounter[i] = 0; }
	static bool isBitSet(AbridgedTopic const& _h, unsigned _index);

	enum { CounterSize = 8 * TopicBloomFilter::size };
	std::array<uint16_t, CounterSize> m_refCounter;
};

}
}






