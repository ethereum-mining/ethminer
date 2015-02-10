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
/** @file CommonEth.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CommonEth.h"
#include <random>
#include <libdevcrypto/SHA3.h>
#include "Exceptions.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace eth
{

const unsigned c_protocolVersion = 53;
const unsigned c_databaseVersion = 5;

template <size_t n> u256 exp10()
{
	return exp10<n - 1>() * u256(10);
}

template <> u256 exp10<0>()
{
	return u256(1);
}

vector<pair<u256, string>> const& units()
{
	static const vector<pair<u256, string>> s_units =
	{
		{exp10<54>(), "Uether"},
		{exp10<51>(), "Vether"},
		{exp10<48>(), "Dether"},
		{exp10<45>(), "Nether"},
		{exp10<42>(), "Yether"},
		{exp10<39>(), "Zether"},
		{exp10<36>(), "Eether"},
		{exp10<33>(), "Pether"},
		{exp10<30>(), "Tether"},
		{exp10<27>(), "Gether"},
		{exp10<24>(), "Mether"},
		{exp10<21>(), "grand"},
		{exp10<18>(), "ether"},
		{exp10<15>(), "finney"},
		{exp10<12>(), "szabo"},
		{exp10<9>(), "Gwei"},
		{exp10<6>(), "Mwei"},
		{exp10<3>(), "Kwei"},
		{exp10<0>(), "wei"}
	};

	return s_units;
}

std::string formatBalance(bigint const& _b)
{
	ostringstream ret;
	u256 b;
	if (_b < 0)
	{
		ret << "-";
		b = (u256)-_b;
	}
	else
		b = (u256)_b;

	if (b > units()[0].first * 10000)
	{
		ret << (b / units()[0].first) << " " << units()[0].second;
		return ret.str();
	}
	ret << setprecision(5);
	for (auto const& i: units())
		if (i.first != 1 && b >= i.first * 100)
		{
			ret << (double(b / (i.first / 1000)) / 1000.0) << " " << i.second;
			return ret.str();
		}
	ret << b << " wei";
	return ret.str();
}

}}
