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
/** @file BasicGasPricer.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include <boost/math/distributions/normal.hpp>
#include "BasicGasPricer.h"
#include "BlockChain.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

void BasicGasPricer::update(BlockChain const& _bc)
{
	unsigned c = 0;
	h256 p = _bc.currentHash();
	m_gasPerBlock = _bc.info(p).gasLimit();

	map<u256, u256> dist;
	u256 total = 0;

	// make gasPrice versus gasUsed distribution for the last 1000 blocks
	while (c < 1000 && p)
	{
		BlockInfo bi = _bc.info(p);
		if (bi.transactionsRoot() != EmptyTrie)
		{
			auto bb = _bc.block(p);
			RLP r(bb);
			BlockReceipts brs(_bc.receipts(bi.hash()));
			size_t i = 0;
			for (auto const& tr: r[1])
			{
				Transaction tx(tr.data(), CheckTransaction::None);
				u256 gu = brs.receipts[i].gasUsed();
				dist[tx.gasPrice()] += gu;
				total += gu;
				i++;
			}
		}
		p = bi.parentHash();
		++c;
	}

	// fill m_octiles with weighted gasPrices
	if (total > 0)
	{
		m_octiles[0] = dist.begin()->first;

		// calc mean
		u256 mean = 0;
		for (auto const& i: dist)
			mean += i.first * i.second;
		mean /= total;

		// calc standard deviation
		u256 sdSquared = 0;
		for (auto const& i: dist)
			sdSquared += i.second * (i.first - mean) * (i.first - mean);
		sdSquared /= total;

		if (sdSquared)
		{
			long double sd = sqrt(sdSquared.convert_to<long double>());
			long double normalizedSd = sd / mean.convert_to<long double>();

			// calc octiles normalized to gaussian distribution
			boost::math::normal gauss(1.0, (normalizedSd > 0.01) ? normalizedSd : 0.01);
			for (size_t i = 1; i < 8; i++)
				m_octiles[i] = u256(mean.convert_to<long double>() * boost::math::quantile(gauss, i / 8.0));
			m_octiles[8] = dist.rbegin()->first;
		}
		else
		{
			for (size_t i = 0; i < 9; i++)
				m_octiles[i] = (i + 1) * mean / 5;
		}
	}
}
