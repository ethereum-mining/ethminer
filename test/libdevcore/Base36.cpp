//
// Created by Marek Kotewicz on 10/08/15.
//

#include <boost/test/unit_test.hpp>
#include <libdevcore/Base64.h>
#include <libethcore/ICAP.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

BOOST_AUTO_TEST_SUITE(Base36Tests)

BOOST_AUTO_TEST_CASE(basicEncoding)
{
	FixedHash<2> value("0x0048");
	string encoded = toBase36<2>(value);
	BOOST_CHECK_EQUAL(encoded, "20");
}

BOOST_AUTO_TEST_CASE(basicEncoding2)
{
	FixedHash<2> value("0x0072");
	string encoded = toBase36<2>(value);
	BOOST_CHECK_EQUAL(encoded, "36");
}

BOOST_AUTO_TEST_CASE(basicEncoding3)
{
	FixedHash<2> value("0xffff");
	string encoded = toBase36<2>(value);
	BOOST_CHECK_EQUAL(encoded, "1EKF");
}


BOOST_AUTO_TEST_SUITE_END()
