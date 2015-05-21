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
/** @file createRandomStateTest.cpp
 * @author Christoph Jentzsch <jentzsch.simulationsoftware@gmail.com>
 * @date 2015
 * Creating a random state test.
 */

#include <string>
#include <iostream>
#include <chrono>

#include <boost/random.hpp>
#include <boost/filesystem/path.hpp>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <json_spirit/json_spirit.h>
#include <json_spirit/json_spirit_reader_template.h>
#include <json_spirit/json_spirit_writer_template.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/CommonData.h>
#include <libevmcore/Instruction.h>
#include <libevm/VMFactory.h>
#include <test/libevm/vm.h>
#include <test/TestHelper.h>
#include <test/fuzzTesting/fuzzHelper.h>

using namespace std;
using namespace json_spirit;
using namespace dev;

void doStateTests(json_spirit::mValue& _v);

int main(int argc, char *argv[])
{
	g_logVerbosity = 0;
	string randomCode = dev::test::RandomCode::generate(25);

	string const s = R"(
	{
		"randomStatetest" : {
			"env" : {
				"currentCoinbase" : "945304eb96065b2a98b57a48a06ae28d285a71b5",
				"currentDifficulty" : "5623894562375",
				"currentGasLimit" : "115792089237316195423570985008687907853269984665640564039457584007913129639935",
				"currentNumber" : "0",
				"currentTimestamp" : "1",
				"previousHash" : "5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6"
			},
			"pre" : {
				"095e7baea6a6c7c4c2dfeb977efac326af552d87" : {
					"balance" : "0",
					"code" : "0x6001600101600055",
					"nonce" : "0",
					"storage" : {
					}
				},
				"945304eb96065b2a98b57a48a06ae28d285a71b5" : {
					"balance" : "46",
					"code" : "0x6000355415600957005b60203560003555",
					"nonce" : "0",
					"storage" : {
					}
				},
				"a94f5374fce5edbc8e2a8697c15331677e6ebf0b" : {
					"balance" : "1000000000000000000",
					"code" : "0x",
					"nonce" : "0",
					"storage" : {
					}
				}
			},
			"transaction" : {
				"data" : "0x42",
				"gasLimit" : "400000",
				"gasPrice" : "1",
				"nonce" : "0",
				"secretKey" : "45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8",
				"to" : "095e7baea6a6c7c4c2dfeb977efac326af552d87",
				"value" : "100000"
			}
		}
	}
)";
	mValue v;
	read_string(s, v);

	// insert new random code
	v.get_obj().find("randomStatetest")->second.get_obj().find("pre")->second.get_obj().begin()->second.get_obj()["code"] = "0x" + randomCode;

	// insert new data in tx
	v.get_obj().find("randomStatetest")->second.get_obj().find("transaction")->second.get_obj()["data"] = "0x" + randomCode;

	// insert new value in tx
	v.get_obj().find("randomStatetest")->second.get_obj().find("transaction")->second.get_obj()["value"] = dev::test::RandomCode::randomUniInt();

	// insert new gasLimit in tx
	v.get_obj().find("randomStatetest")->second.get_obj().find("transaction")->second.get_obj()["gasLimit"] = dev::test::RandomCode::randomUniInt();

	// fill test
	doStateTests(v);

	// stream to output for further handling by the bash script
	cout << json_spirit::write_string(v, true);

	return 0;
}

void doStateTests(json_spirit::mValue& _v)
{
	eth::VMFactory::setKind(eth::VMKind::Interpreter);

	for (auto& i: _v.get_obj())
	{
		//cerr << i.first << endl;
		mObject& o = i.second.get_obj();

		assert(o.count("env") > 0);
		assert(o.count("pre") > 0);
		assert(o.count("transaction") > 0);

		test::ImportTest importer(o, true);

		eth::State theState = importer.m_statePre;
		bytes output;

		try
		{
			output = theState.execute(test::lastHashes(importer.m_environment.currentBlock.number), importer.m_transaction).output;
		}
		catch (Exception const& _e)
		{
			cnote << "state execution did throw an exception: " << diagnostic_information(_e);
			theState.commit();
		}
		catch (std::exception const& _e)
		{
			cnote << "state execution did throw an exception: " << _e.what();
		}
#if ETH_FATDB
		importer.exportTest(output, theState);
#else
		cout << "You can not fill tests when FATDB is switched off";
#endif
	}
}

