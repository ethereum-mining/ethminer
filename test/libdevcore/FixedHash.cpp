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
/** @file FixedHash.cpp
 * @author Lefterus <lefteris@ethdev.com>
 * @date 2015
 */

#include <libdevcore/FixedHash.h>
#include "../TestHelper.h"

using namespace std;
using namespace dev;

namespace dev
{
namespace test
{

BOOST_AUTO_TEST_SUITE(FixedHashTests)

BOOST_AUTO_TEST_CASE(FixedHashComparisons)
{
	FixedHash<4> h1(sha3("abcd"));
	FixedHash<4> h2(sha3("abcd"));
	FixedHash<4> h3(sha3("aadd"));
	FixedHash<4> h4(0xBAADF00D);
	FixedHash<4> h5(0xAAAAAAAA);
	FixedHash<4> h6(0xBAADF00D);

	BOOST_CHECK(h1 == h2);
	BOOST_CHECK(h2 != h3);

	BOOST_CHECK(h4 > h5);
	BOOST_CHECK(h5 < h4);
	BOOST_CHECK(h6 <= h4);
	BOOST_CHECK(h6 >= h4);
}

BOOST_AUTO_TEST_CASE(FixedHashXOR)
{
	FixedHash<2> h1("0xAAAA");
	FixedHash<2> h2("0xBBBB");

	BOOST_CHECK((h1 ^ h2) == FixedHash<2>("0x1111"));
	h1 ^= h2;
	BOOST_CHECK(h1 == FixedHash<2>("0x1111"));
}

BOOST_AUTO_TEST_CASE(FixedHashOR)
{
	FixedHash<4> h1("0xD3ADB33F");
	FixedHash<4> h2("0xBAADF00D");
	FixedHash<4> res("0xFBADF33F");

	BOOST_CHECK((h1 | h2) == res);
	h1 |= h2;
	BOOST_CHECK(h1 == res);
}

BOOST_AUTO_TEST_CASE(FixedHashAND)
{
	FixedHash<4> h1("0xD3ADB33F");
	FixedHash<4> h2("0xBAADF00D");
	FixedHash<4> h3("0x92aDB00D");

	BOOST_CHECK((h1 & h2) == h3);
	h1 &= h2;
	BOOST_CHECK(h1 = h3);
}

BOOST_AUTO_TEST_CASE(FixedHashInvert)
{
	FixedHash<4> h1("0xD3ADB33F");
	FixedHash<4> h2("0x2C524CC0");

	BOOST_CHECK(~h1  == h2);
}

BOOST_AUTO_TEST_CASE(FixedHashContains)
{
	FixedHash<4> h1("0xD3ADB331");
	FixedHash<4> h2("0x0000B331");
	FixedHash<4> h3("0x0000000C");

	BOOST_CHECK(h1.contains(h2));
	BOOST_CHECK(!h1.contains(h3));
}

void incrementSingleIteration(unsigned seed)
{
	unsigned next = seed + 1;

	FixedHash<4> h1(seed);
	FixedHash<4> h2 = h1;
	FixedHash<4> h3(next);

	FixedHash<32> hh1(seed);
	FixedHash<32> hh2 = hh1;
	FixedHash<32> hh3(next);

	BOOST_CHECK_EQUAL(++h2, h3);
	BOOST_CHECK_EQUAL(++hh2, hh3);

	BOOST_CHECK(h2 > h1);
	BOOST_CHECK(hh2 > hh1);

	unsigned reverse1 = ((FixedHash<4>::Arith)h2).convert_to<unsigned>();
	unsigned reverse2 = ((FixedHash<32>::Arith)hh2).convert_to<unsigned>();

	BOOST_CHECK_EQUAL(next, reverse1);
	BOOST_CHECK_EQUAL(next, reverse2);
}

BOOST_AUTO_TEST_CASE(FixedHashIncrement)
{
	incrementSingleIteration(0);
	incrementSingleIteration(1);
	incrementSingleIteration(0xBAD);
	incrementSingleIteration(0xBEEF);
	incrementSingleIteration(0xFFFF);
	incrementSingleIteration(0xFEDCBA);
	incrementSingleIteration(0x7FFFFFFF);
}

BOOST_AUTO_TEST_SUITE_END()

}
}
