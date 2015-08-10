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
/** @file icap.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include <boost/test/unit_test.hpp>
#include <libethcore/ICAP.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

BOOST_AUTO_TEST_SUITE(IcapTests)

BOOST_AUTO_TEST_CASE(addressEncoding)
{
	Address address("0x52dc504a422f0e2a9e7632a34a50f1a82f8224c7");
	ICAP icap(address);
	BOOST_CHECK_EQUAL(icap.encoded(), "XE499OG1EH8ZZI0KXC6N83EKGT1BM97P2O7");
}

BOOST_AUTO_TEST_CASE(addressEncodingRandomString)
{
	Address address("0x11c5496aee77c1ba1f0854206a26dda82a81d6d8");
	ICAP icap(address);
	BOOST_CHECK_EQUAL(icap.encoded(), "XE1222Q908LN1QBBU6XUQSO1OHWJIOS46OO");
}

BOOST_AUTO_TEST_CASE(addressEncodingWithZeroPrefix)
{
	Address address("0x00c5496aee77c1ba1f0854206a26dda82a81d6d8");
	ICAP icap(address);
	BOOST_CHECK_EQUAL(icap.encoded(), "XE7338O073KYGTWWZN0F2WZ0R8PX5ZPPZS");
}

BOOST_AUTO_TEST_CASE(addressDecoding)
{
	Address address("0x52dc504a422f0e2a9e7632a34a50f1a82f8224c7");
	ICAP icap = ICAP::decoded("XE499OG1EH8ZZI0KXC6N83EKGT1BM97P2O7");
	BOOST_CHECK_EQUAL(icap.direct(), address);
}

BOOST_AUTO_TEST_CASE(addressDecodingRandomString)
{
	Address address("0x11c5496aee77c1ba1f0854206a26dda82a81d6d8");
	ICAP icap = ICAP::decoded("XE1222Q908LN1QBBU6XUQSO1OHWJIOS46OO");
	BOOST_CHECK_EQUAL(icap.direct(), address);
}

BOOST_AUTO_TEST_CASE(addressDecodingWithZeroPrefix)
{
	Address address("0x00c5496aee77c1ba1f0854206a26dda82a81d6d8");
	ICAP icap = ICAP::decoded("XE7338O073KYGTWWZN0F2WZ0R8PX5ZPPZS");
	BOOST_CHECK_EQUAL(icap.direct(), address);
}

BOOST_AUTO_TEST_CASE(addressDecodingAndEncoding)
{
	std::string encoded = "XE499OG1EH8ZZI0KXC6N83EKGT1BM97P2O7";
	ICAP icap = ICAP::decoded(encoded);
	BOOST_CHECK_EQUAL(icap.encoded(), encoded);
}

BOOST_AUTO_TEST_SUITE_END()