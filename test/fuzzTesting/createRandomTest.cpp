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


extern std::string const c_testExampleTransactionTest;
std::vector<std::string> getTypes();
void parseTestWithTypes(std::string& test);
void randomTransactionTest();
void randomBlockChainTest();

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
			if (testSuite != "BlockChainTests" && testSuite != "TransactionTests")
				testSuite = "";
		}
	}

	if (testSuite == "")
		std::cout << "Error! Test suite not supported! (Usage -t TestSuite)";
	else
	if (testSuite == "BlockChainTests")
		randomBlockChainTest();
	else
	if (testSuite == "TransactionTests")
		randomTransactionTest();

	return 0;
}

void randomTransactionTest()
{
	std::string newTest = c_testExampleTransactionTest;
	parseTestWithTypes(newTest);
	json_spirit::mValue v;
	json_spirit::read_string(newTest, v);
	dev::test::doTransactionTests(v, true);
	std::cout << json_spirit::write_string(v, true);
}

void randomBlockChainTest()
{

}

/// Parse Test string replacing keywords to fuzzed values
void parseTestWithTypes(std::string& test)
{
	std::vector<std::string> types = getTypes();
	for (unsigned i = 0; i < types.size(); i++)
	{
		std::size_t pos = test.find(types.at(i));
		while (pos != std::string::npos)
		{
			if (types.at(i) == "[CODE]")
				test.replace(pos, 6, "0x"+dev::test::RandomCode::generate(10));
			else
			if (types.at(i) == "[HEX]")
				test.replace(pos, 5, dev::test::RandomCode::randomUniIntHex());
			else
			if (types.at(i) == "[HASH20]")
				test.replace(pos, 8, dev::test::RandomCode::rndByteSequence(20));
			else
			if (types.at(i) == "[0xHASH32]")
				test.replace(pos, 10, "0x" + dev::test::RandomCode::rndByteSequence(32));
			else
			if (types.at(i) == "[V]")
			{
				int random = dev::test::RandomCode::randomUniInt() % 100;
				if (random < 30)
					test.replace(pos, 3, "28");
				else
				if (random < 60)
					test.replace(pos, 3, "29");
				else
					test.replace(pos, 3, "0x" + dev::test::RandomCode::rndByteSequence(1));
			}

			pos = test.find(types.at(i));
		}
	}
}

std::vector<std::string> getTypes()
{
	return {"[CODE]", "[HEX]", "[HASH20]", "[0xHASH32]", "[V]"};
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



