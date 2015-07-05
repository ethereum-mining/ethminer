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

template <unsigned N>
class TopicBloomFilterBase: public FixedHash<N>
{
public:
	TopicBloomFilterBase() { init(); }
	TopicBloomFilterBase(FixedHash<N> const& _h): FixedHash<N>(_h) { init(); }

	void addBloom(dev::shh::AbridgedTopic const& _h) { addRaw(_h.template bloomPart<BitsPerBloom, N>()); }
	void removeBloom(dev::shh::AbridgedTopic const& _h) { removeRaw(_h.template bloomPart<BitsPerBloom, N>()); }
	bool containsBloom(dev::shh::AbridgedTopic const& _h) const { return this->contains(_h.template bloomPart<BitsPerBloom, N>()); }

	void addRaw(FixedHash<N> const& _h);
	void removeRaw(FixedHash<N> const& _h);
	bool containsRaw(FixedHash<N> const& _h) const { return this->contains(_h); }
	
private:
	void init() { for (unsigned i = 0; i < CounterSize; ++i) m_refCounter[i] = 0; }
	static bool isBitSet(FixedHash<N> const& _h, unsigned _index);

	static const unsigned CounterSize = N * 8;
	std::array<uint16_t, CounterSize> m_refCounter;
};

static unsigned const c_powerOfTwoBitMmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

template <unsigned N>
void TopicBloomFilterBase<N>::addRaw(FixedHash<N> const& _h)
{
	*this |= _h;
	for (unsigned i = 0; i < CounterSize; ++i)
		if (isBitSet(_h, i))
		{
			if (m_refCounter[i] != std::numeric_limits<uint16_t>::max())
				m_refCounter[i]++;
			else
				BOOST_THROW_EXCEPTION(Overflow());
		}
}

template <unsigned N>
void TopicBloomFilterBase<N>::removeRaw(FixedHash<N> const& _h)
{
	for (unsigned i = 0; i < CounterSize; ++i)
		if (isBitSet(_h, i))
		{
			if (m_refCounter[i])
				m_refCounter[i]--;

			if (!m_refCounter[i])
				(*this)[i / 8] &= ~c_powerOfTwoBitMmask[i % 8];
		}
}

template <unsigned N>
bool TopicBloomFilterBase<N>::isBitSet(FixedHash<N> const& _h, unsigned _index)
{	
	unsigned iByte = _index / 8;
	unsigned iBit = _index % 8;
	return (_h[iByte] & c_powerOfTwoBitMmask[iBit]) != 0;
}

using TopicBloomFilter = TopicBloomFilterBase<TopicBloomFilterSize>;

}
}






