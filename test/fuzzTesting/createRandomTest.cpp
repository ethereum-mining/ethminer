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
/** @file createRandomTest.cpp
 * @author Dimitry Khokhlov <winsvega@mail.ru>
 * @date 2015
 */

#include <string>
#include <iostream>

#include <test/TestHelper.h>
#include <test/fuzzTesting/fuzzHelper.h>

//String Variables
extern std::string const c_testExampleStateTest;
extern std::string const c_testExampleTransactionTest;

//Main Test functinos
void fillRandomTest(std::function<void(json_spirit::mValue&, bool)> doTests, std::string const& testString);

//Helper Functions
std::vector<std::string> getTypes();
void parseTestWithTypes(std::string& test);

int main(int argc, char *argv[])
{
	std::string testSuite;
	for (auto i = 0; i < argc; ++i)
	{
		auto arg = std::string{argv[i]};
		dev::test::Options& options = const_cast<dev::test::Options&>(dev::test::Options::get());
		if (arg == "--fulloutput")
			options.fulloutput = true;
		if (arg == "-t" && i + 1 < argc)
		{
			testSuite = argv[i + 1];
			if (testSuite != "BlockChainTests" && testSuite != "TransactionTests" && testSuite != "StateTests")
				testSuite = "";
		}
	}

	if (testSuite == "")
		std::cout << "Error! Test suite not supported! (Usage -t TestSuite)";
	else
	if (testSuite == "BlockChainTests")
		fillRandomTest(dev::test::doTransactionTests, c_testExampleTransactionTest);
	else
	if (testSuite == "TransactionTests")
		fillRandomTest(dev::test::doTransactionTests, c_testExampleTransactionTest);
	else
	if (testSuite == "StateTests")
		fillRandomTest(dev::test::doStateTests, c_testExampleStateTest);

	return 0;
}

void fillRandomTest(std::function<void(json_spirit::mValue&, bool)> doTests, std::string const& testString)
{
	//redirect all output to the stream
	std::ostringstream strCout;
	std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();
	std::cout.rdbuf( strCout.rdbuf() );
	std::cerr.rdbuf( strCout.rdbuf() );

	json_spirit::mValue v;
	try
	{
		std::string newTest = testString;
		parseTestWithTypes(newTest);
		json_spirit::read_string(newTest, v);
		doTests(v, true);
	}
	catch(...)
	{
		std::cerr << "Test fill exception!";
	}

	//restroe output
	std::cout.rdbuf(oldCoutStreamBuf);
	std::cerr.rdbuf(oldCoutStreamBuf);
	std::cout << json_spirit::write_string(v, true);
}

/// Parse Test string replacing keywords to fuzzed values
void parseTestWithTypes(std::string& _test)
{
	dev::test::RandomCodeOptions options;
	options.setWeight(dev::eth::Instruction::STOP, 10);		//default 50
	options.setWeight(dev::eth::Instruction::SSTORE, 70);
	options.setWeight(dev::eth::Instruction::CALL, 75);
	options.addAddress(dev::Address("0xffffffffffffffffffffffffffffffffffffffff"));
	options.addAddress(dev::Address("0x1000000000000000000000000000000000000000"));
	options.addAddress(dev::Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"));
	options.addAddress(dev::Address("0x945304eb96065b2a98b57a48a06ae28d285a71b5"));
	options.addAddress(dev::Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b"));
	options.addAddress(dev::Address("0x0000000000000000000000000000000000000001"));
	options.addAddress(dev::Address("0x0000000000000000000000000000000000000002"));
	options.addAddress(dev::Address("0x0000000000000000000000000000000000000003"));
	options.addAddress(dev::Address("0x0000000000000000000000000000000000000004"));
	options.smartCodeProbability = 35;

	std::vector<std::string> types = getTypes();
	for (unsigned i = 0; i < types.size(); i++)
	{
		std::size_t pos = _test.find(types.at(i));
		while (pos != std::string::npos)
		{
			if (types.at(i) == "[CODE]")
				_test.replace(pos, 6, "0x"+dev::test::RandomCode::generate(10, options));
			else
			if (types.at(i) == "[HEX]")
				_test.replace(pos, 5, dev::test::RandomCode::randomUniIntHex());
			else
			if (types.at(i) == "[HASH20]")
				_test.replace(pos, 8, dev::test::RandomCode::rndByteSequence(20));
			else
			if (types.at(i) == "[0xHASH32]")
				_test.replace(pos, 10, "0x" + dev::test::RandomCode::rndByteSequence(32));
			else
			if (types.at(i) == "[HASH32]")
				_test.replace(pos, 8, dev::test::RandomCode::rndByteSequence(32));
			else
			if (types.at(i) == "[V]")
			{
				int random = dev::test::RandomCode::randomUniInt() % 100;
				if (random < 30)
					_test.replace(pos, 3, "28");
				else
				if (random < 60)
					_test.replace(pos, 3, "29");
				else
					_test.replace(pos, 3, "0x" + dev::test::RandomCode::rndByteSequence(1));
			}

			pos = _test.find(types.at(i));
		}
	}
}

std::vector<std::string> getTypes()
{
	return {"[CODE]", "[HEX]", "[HASH20]", "[HASH32]", "[0xHASH32]", "[V]"};
}

std::string const c_testExampleTransactionTest = R"(
{
"TransactionTest" : {
		"transaction" :
		{
			"data" : "[CODE]",
			"gasLimit" : "[HEX]",
			"gasPrice" : "[HEX]",
			"nonce" : "[HEX]",
			"to" : "[HASH20]",
			"value" : "[HEX]",
			"v" : "[V]",
			"r" : "[0xHASH32]",
			"s" : "[0xHASH32]"
		}
	}
}
)";

std::string const c_testExampleStateTest = R"(
{
	"randomStatetest" : {
		"env" : {
		"currentCoinbase" : "[HASH20]",
		"currentDifficulty" : "[HEX]",
		"currentGasLimit" : "[HEX]",
		"currentNumber" : "[HEX]",
		"currentTimestamp" : "[HEX]",
		"previousHash" : "[HASH32]"
		},
	"pre" : {
		"095e7baea6a6c7c4c2dfeb977efac326af552d87" : {
			"balance" : "[HEX]",
			"code" : "[CODE]",
			"nonce" : "[V]",
			"storage" : {
			}
		},
		"945304eb96065b2a98b57a48a06ae28d285a71b5" : {
			"balance" : "[HEX]",
			"code" : "[CODE]",
			"nonce" : "[V]",
			"storage" : {
			}
		},
		"a94f5374fce5edbc8e2a8697c15331677e6ebf0b" : {
			"balance" : "[HEX]",
			"code" : "0x",
			"nonce" : "0",
			"storage" : {
			}
		}
	},
	"transaction" : {
		"data" : "[CODE]",
		"gasLimit" : "[HEX]",
		"gasPrice" : "[V]",
		"nonce" : "0",
		"secretKey" : "45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8",
		"to" : "095e7baea6a6c7c4c2dfeb977efac326af552d87",
		"value" : "[HEX]"
		}
	}
}
)";
