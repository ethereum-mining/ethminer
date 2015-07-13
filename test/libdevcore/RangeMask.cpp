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
/** @file RangeMask.cpp
 * @author Christian <c@ethdev.com>
 * @date 2015
 */

#include <libdevcore/RangeMask.h>
#include "../TestHelper.h"

using namespace std;
using namespace dev;

namespace dev
{
namespace test
{

BOOST_AUTO_TEST_SUITE(RangeMaskTest)

BOOST_AUTO_TEST_CASE(constructor)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	for (RM r: {RM(), RM(1, 10), RM(Range(2, 10))})
	{
		BOOST_CHECK(r.empty());
		BOOST_CHECK(!r.contains(0));
		BOOST_CHECK(!r.contains(1));
		BOOST_CHECK_EQUAL(0, r.size());
	}
	BOOST_CHECK(RM().full());
	BOOST_CHECK(!RM(1, 10).full());
	BOOST_CHECK(!RM(Range(2, 10)).full());
}

BOOST_AUTO_TEST_CASE(simple_unions)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	RM m(Range(0, 2000));
	m.unionWith(Range(1, 2));
	BOOST_CHECK_EQUAL(m.size(), 1);
	m.unionWith(Range(50, 250));
	BOOST_CHECK_EQUAL(m.size(), 201);
	m.unionWith(Range(10, 16));
	BOOST_CHECK_EQUAL(m.size(), 207);
	BOOST_CHECK(m.contains(1));
	BOOST_CHECK(m.contains(11));
	BOOST_CHECK(m.contains(51));
	BOOST_CHECK(m.contains(200));
	BOOST_CHECK(!m.contains(2));
	BOOST_CHECK(!m.contains(7));
	BOOST_CHECK(!m.contains(17));
	BOOST_CHECK(!m.contains(258));
}

BOOST_AUTO_TEST_CASE(empty_union)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	RM m(Range(0, 2000));
	m.unionWith(Range(3, 6));
	BOOST_CHECK_EQUAL(m.size(), 3);
	m.unionWith(Range(50, 50));
	BOOST_CHECK_EQUAL(m.size(), 3);
	m.unionWith(Range(0, 0));
	BOOST_CHECK_EQUAL(m.size(), 3);
	m.unionWith(Range(1, 1));
	BOOST_CHECK_EQUAL(m.size(), 3);
	m.unionWith(Range(2, 2));
	BOOST_CHECK_EQUAL(m.size(), 3);
	m.unionWith(Range(3, 3));
	BOOST_CHECK_EQUAL(m.size(), 3);
}

BOOST_AUTO_TEST_CASE(overlapping_unions)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	RM m(Range(0, 2000));
	m.unionWith(Range(10, 20));
	BOOST_CHECK_EQUAL(10, m.size());
	m.unionWith(Range(30, 40));
	BOOST_CHECK_EQUAL(20, m.size());
	m.unionWith(Range(15, 30));
	BOOST_CHECK_EQUAL(40 - 10, m.size());
	m.unionWith(Range(50, 60));
	m.unionWith(Range(45, 55));
	// [40, 45) still missing here
	BOOST_CHECK_EQUAL(60 - 10 - 5, m.size());
	m.unionWith(Range(15, 56));
	BOOST_CHECK_EQUAL(60 - 10, m.size());
	m.unionWith(Range(15, 65));
	BOOST_CHECK_EQUAL(65 - 10, m.size());
	m.unionWith(Range(5, 70));
	BOOST_CHECK_EQUAL(70 - 5, m.size());
}

BOOST_AUTO_TEST_CASE(complement)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	RM m(Range(0, 2000));
	m.unionWith(7).unionWith(9);
	m = ~m;
	m.unionWith(7).unionWith(9);
	m = ~m;
	BOOST_CHECK(m.empty());

	m += Range(0, 10);
	m += Range(1000, 2000);
	m.invert();
	BOOST_CHECK_EQUAL(m.size(), 1000 - 10);
}

BOOST_AUTO_TEST_CASE(iterator)
{
	using RM = RangeMask<unsigned>;
	using Range = pair<unsigned, unsigned>;
	RM m(Range(0, 2000));
	m.unionWith(Range(7, 9));
	m.unionWith(11);
	m.unionWith(Range(200, 205));

	vector<unsigned> elements;
	copy(m.begin(), m.end(), back_inserter(elements));
	BOOST_CHECK(elements == (vector<unsigned>{7, 8, 11, 200, 201, 202, 203, 204}));
}

BOOST_AUTO_TEST_SUITE_END()

}
}
