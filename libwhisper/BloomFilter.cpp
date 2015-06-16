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
/** @file BloomFilter.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date June 2015
*/

#include "BloomFilter.h"

using namespace std;
using namespace dev;
using namespace dev::shh;

static unsigned const c_mask[] = { 1, 2, 4, 8, 16, 32, 64, 128 };

void TopicBloomFilter::add(AbridgedTopic const& _h)
{
	*this |= _h;
	unsigned count = 0;
	for (unsigned i = 0; i < CounterSize; ++i)
		if (isBitSet(_h, i))
		{
			if (m_refCounter[i] != numeric_limits<uint16_t>::max())
				m_refCounter[i]++;
			//else: overflow

			// in order to encounter overflow, you have to set at least 65536 filters simultaneously.
			// even then, the problem will only arise after at least 65536 filters will be be removed.
			// we assume, it will never happen.
		}
}

void TopicBloomFilter::remove(AbridgedTopic const& _h)
{
	unsigned count = 0;
	for (unsigned i = 0; i < CounterSize; ++i)
		if (isBitSet(_h, i))
		{
			if (m_refCounter[i])
				m_refCounter[i]--;

			if (!m_refCounter[i])
				(*this)[i/8] &= ~c_mask[i%8];
		}
}

bool TopicBloomFilter::isBitSet(AbridgedTopic const& _h, unsigned _index)
{	
	unsigned iByte = _index / 8;
	unsigned iBit = _index % 8;
	return (_h[iByte] & c_mask[iBit]) != 0;
}



