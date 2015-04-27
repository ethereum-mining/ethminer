//
// Created by Marek Kotewicz on 27/04/15.
//

#include <boost/test/unit_test.hpp>
#include <libjsengine/JSV8Engine.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

BOOST_AUTO_TEST_SUITE(jsv8engine)

BOOST_AUTO_TEST_CASE(evalInteger)
{
	JSV8Engine scope;
	string result = scope.evaluate("1 + 1");
	BOOST_CHECK_EQUAL(result, "2");
}

BOOST_AUTO_TEST_CASE(evalString)
{
	JSV8Engine scope;
	string result = scope.evaluate("'hello ' + 'world'");
	BOOST_CHECK_EQUAL(result, "hello world");
}

BOOST_AUTO_TEST_CASE(evalEmpty)
{
	JSV8Engine scope;
	string result = scope.evaluate("");
	BOOST_CHECK_EQUAL(result, "undefined");
}

BOOST_AUTO_TEST_SUITE_END()
