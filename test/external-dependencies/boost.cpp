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
/** @file boost.cpp
 * @author Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 * Tests for external dependencies: Boost
 */

#include <boost/test/unit_test.hpp>
#include <libdevcore/Common.h>

BOOST_AUTO_TEST_SUITE(ExtDepBoost)

// test that reproduces issue https://github.com/ethereum/cpp-ethereum/issues/1977
BOOST_AUTO_TEST_CASE(u256_overflow_test)
{
	dev::u256 a = 14;
	dev::bigint b = dev::bigint("115792089237316195423570985008687907853269984665640564039457584007913129639948");
	// to fix cast `a` to dev::bigint
	BOOST_CHECK(a < b);
}

BOOST_AUTO_TEST_SUITE_END()
