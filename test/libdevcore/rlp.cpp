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
/** @file rlp.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * RLP test functions.
 */

#include <fstream>
#include <sstream>

#include <boost/test/unit_test.hpp>

#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <algorithm>
#include "test/JsonSpiritHeaders.h"
#include "test/TestHelper.h"

using namespace std;
using namespace dev;
namespace js = json_spirit;

namespace dev
{
	namespace test
	{	
		void buildRLP(js::mValue& _v, RLPStream& _rlp);
		void checkRLPAgainstJson(js::mValue& v, RLP& u);
		enum class RlpType
		{
			Valid,
			Invalid,
			Test
		};

		void doRlpTests(json_spirit::mValue& v, bool _fillin)
		{
			for (auto& i: v.get_obj())
			{
				js::mObject& o = i.second.get_obj();
				if (test::Options::get().singleTest && test::Options::get().singleTestName != i.first)
				{
					o.clear();
					continue;
				}

				cout << "  " << i.first << endl;
				TBOOST_REQUIRE((o.count("out") > 0));
				TBOOST_REQUIRE((!o["out"].is_null()));

				if (_fillin)
				{
					try
					{
						bytes payloadToDecode = fromHex(o["out"].get_str());
						RLP payload(payloadToDecode);
						ostringstream() << payload;
						o["in"] = "VALID";
					}
					catch (Exception const& _e)
					{
						cnote << "Exception: " << diagnostic_information(_e);
						o["in"] = "INVALID";
					}
					catch (std::exception const& _e)
					{
						cnote << "rlp exception: " << _e.what();
						o["in"] = "INVALID";
					}
				}
				else
				{
					//Check Encode
					TBOOST_REQUIRE((o.count("in") > 0));
					RlpType rlpType = RlpType::Test;
					if (o["in"].type() == js::str_type)
					{
						if (o["in"].get_str() == "INVALID")
							rlpType = RlpType::Invalid;
						else	if (o["in"].get_str() == "VALID")
							rlpType = RlpType::Valid;
					}

					if (rlpType == RlpType::Test)
					{
						RLPStream s;
						dev::test::buildRLP(o["in"], s);
						string computedText = toHex(s.out());

						string expectedText(o["out"].get_str());
						transform(expectedText.begin(), expectedText.end(), expectedText.begin(), ::tolower );

						stringstream msg;
						msg << "Encoding Failed: expected: " << expectedText << std::endl;
						msg << " But Computed: " << computedText;
						TBOOST_CHECK_MESSAGE(
							(expectedText == computedText),
							msg.str()
							);
					}

					//Check Decode
					// Uses the same test cases as encoding but in reverse.
					// We read into the string of hex values, convert to bytes,
					// and then compare the output structure to the json of the
					// input object.
					bool was_exception = false;
					js::mValue& inputData = o["in"];
					try
					{
						bytes payloadToDecode = fromHex(o["out"].get_str());
						RLP payload(payloadToDecode);

						//attempt to read all the contents of RLP
						ostringstream() << payload;

						if (rlpType == RlpType::Test)
							dev::test::checkRLPAgainstJson(inputData, payload);
					}
					catch (Exception const& _e)
					{
						cnote << "Exception: " << diagnostic_information(_e);
						was_exception = true;
					}
					catch (exception const& _e)
					{
						cnote << "rlp exception: " << _e.what();
						was_exception = true;
					}
					catch (...)
					{
						was_exception = true;
					}

					//Expect exception as input is INVALID
					if (rlpType == RlpType::Invalid && was_exception)
						continue;

					//Check that there was an exception as input is INVALID
					if (rlpType == RlpType::Invalid && !was_exception)
						TBOOST_ERROR("Expected RLP Exception as rlp should be invalid!");

					//input is VALID check that there was no exceptions
					if (was_exception)
						TBOOST_ERROR("Unexpected RLP Exception!");
				}
			}
		}

		void buildRLP(js::mValue& _v, RLPStream& _rlp)
		{
			if (_v.type() == js::array_type)
			{
				RLPStream s;
				for (auto& i: _v.get_array())
					buildRLP(i, s);
				_rlp.appendList(s.out());
			}
			else if (_v.type() == js::int_type)
				_rlp.append(_v.get_uint64());
			else if (_v.type() == js::str_type)
			{
				auto s = _v.get_str();
				if (s.size() && s[0] == '#')
					_rlp.append(bigint(s.substr(1)));
				else
					_rlp.append(s);
			}
		}

		void checkRLPAgainstJson(js::mValue& v, RLP& u)
		{
			if ( v.type() == js::str_type )
			{
				const string& expectedText = v.get_str();
				if ( !expectedText.empty() && expectedText.front() == '#' )
				{
					// Deal with bigint instead of a raw string
					string bigIntStr = expectedText.substr(1,expectedText.length()-1);
					stringstream bintStream(bigIntStr);
					bigint val;
					bintStream >> val;
					TBOOST_CHECK(( !u.isList() ));
					TBOOST_CHECK(( !u.isNull() ));
					TBOOST_CHECK(( u == val ));
				}
				else
				{
					TBOOST_CHECK(( !u.isList() ));
					TBOOST_CHECK(( !u.isNull() ));
					TBOOST_CHECK(( u.isData() ));
					TBOOST_CHECK(( u.size() == expectedText.length() ));
					TBOOST_CHECK(( u == expectedText ));
				}
			}
			else if ( v.type() == js::int_type )
			{
				const int expectedValue = v.get_int();
				TBOOST_CHECK(( u.isInt() ));
				TBOOST_CHECK(( !u.isList() ));
				TBOOST_CHECK(( !u.isNull() ));
				TBOOST_CHECK(( u == expectedValue ));
			}
			else if ( v.type() == js::array_type )
			{
				TBOOST_CHECK(( u.isList() ));
				TBOOST_CHECK(( !u.isInt() ));
				TBOOST_CHECK(( !u.isData() ));
				js::mArray& arr = v.get_array();
				TBOOST_CHECK(( u.itemCount() == arr.size() ));
				unsigned i;
				for( i = 0; i < arr.size(); i++ )
				{
					RLP item = u[i];
					checkRLPAgainstJson(arr[i], item);
				}
			}
			else
			{
				TBOOST_ERROR("Invalid Javascript object!");
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE(RlpTests)

BOOST_AUTO_TEST_CASE(EmptyArrayList)
{
	try
	{
		bytes payloadToDecode = fromHex("80");
		RLP payload(payloadToDecode);
		ostringstream() << payload;

		payloadToDecode = fromHex("с0");
		RLP payload2(payloadToDecode);
		ostringstream() << payload2;
	}
	catch (Exception const& _e)
	{
		TBOOST_ERROR("Failed test with Exception: " << _e.what());
	}
	catch (exception const& _e)
	{
		TBOOST_ERROR("Failed test with Exception: " << _e.what());
	}
}

BOOST_AUTO_TEST_CASE(invalidRLPtest)
{
	dev::test::executeTests("invalidRLPTest", "/RLPTests", dev::test::getFolder(__FILE__) + "/RLPTestsFiller", dev::test::doRlpTests);
}

BOOST_AUTO_TEST_CASE(rlptest)
{
	dev::test::executeTests("rlptest", "/RLPTests", dev::test::getFolder(__FILE__) + "/RLPTestsFiller", dev::test::doRlpTests);
}

BOOST_AUTO_TEST_CASE(rlpRandom)
{
	test::Options::get();

	string testPath = dev::test::getTestPath();
	testPath += "/RLPTests/RandomRLPTests";

	vector<boost::filesystem::path> testFiles;
	boost::filesystem::directory_iterator iterator(testPath);
	for(; iterator != boost::filesystem::directory_iterator(); ++iterator)
		if (boost::filesystem::is_regular_file(iterator->path()) && iterator->path().extension() == ".json")
			testFiles.push_back(iterator->path());

	for (auto& path: testFiles)
	{
		try
		{
			cnote << "Testing ..." << path.filename();
			json_spirit::mValue v;
			string s = asString(dev::contents(path.string()));
			TBOOST_REQUIRE_MESSAGE((s.length() > 0), "Content of " + path.string() + " is empty. Have you cloned the 'tests' repo branch develop and set ETHEREUM_TEST_PATH to its path?");
			json_spirit::read_string(s, v);
			test::Listener::notifySuiteStarted(path.filename().string());
			dev::test::doRlpTests(v, false);
		}

		catch (Exception const& _e)
		{
			TBOOST_ERROR("Failed test with Exception: " << diagnostic_information(_e));
		}
		catch (std::exception const& _e)
		{
			TBOOST_ERROR("Failed test with Exception: " << _e.what());
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
