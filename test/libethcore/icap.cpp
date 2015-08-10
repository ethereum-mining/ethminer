//
// Created by Marek Kotewicz on 10/08/15.
//

#include <boost/test/unit_test.hpp>
#include <libethcore/ICAP.h>

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

BOOST_AUTO_TEST_SUITE_END()