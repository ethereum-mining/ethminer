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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * RLP tool.
 */
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include "../test/JsonSpiritHeaders.h"
#include <libdevcore/CommonIO.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/SHA3.h>
#include <libethereum/Client.h>
using namespace std;
using namespace dev;
namespace js = json_spirit;

void help()
{
	cout
		<< "Usage abi enc <signature> (<arg1>, (<arg2>, ... ))" << endl
		<< "      abi enc -a <abi.json> <unique_method_name> (<arg1>, (<arg2>, ... ))" << endl
		<< "      abi dec -a <abi.json> [ <signature> | <unique_method_name> ]" << endl
		<< "Options:" << endl
		<< "    -a,--abi-file <filename>  Specify the JSON ABI file." << endl
		<< "Output options (dec mode):" << endl
		<< "    -i,--index <n>  When decoding, output only the nth (counting from 0) return value." << endl
		<< "    -d,--decimal  All data should be displayed as decimal." << endl
		<< "    -x,--hex  Display all data as hex." << endl
		<< "    -b,--binary  Display all data as binary." << endl
		<< "    -p,--prefix  Prefix by a base identifier." << endl
		<< "    -z,--no-zeroes  Remove any leading zeroes from the data." << endl
		<< "    -n,--no-nulls  Remove any trailing nulls from the data." << endl
		<< "General options:" << endl
		<< "    -h,--help  Print this help message and exit." << endl
		<< "    -V,--version  Show the version and exit." << endl
		;
	exit(0);
}

void version()
{
	cout << "abi version " << dev::Version << endl;
	exit(0);
}

enum class Mode {
	Encode,
	Decode
};

enum class Encoding {
	Auto,
	Decimal,
	Hex,
	Binary,
};

int main(int argc, char** argv)
{
	Encoding encoding = Encoding::Auto;
	Mode mode = Mode::Encode;
	string abiFile;
	string methodName;
	bool outputPrefix = false;
	bool clearZeroes = false;
	bool clearNulls = false;
	int outputIndex = -1;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "enc" && i == 1)
			mode = Mode::Encode;
		else if (arg == "dec" && i == 1)
			mode = Mode::Decode;
		else if ((arg == "-a" || arg == "--abi") && argc > i)
			abiFile = argv[++i];
		else if ((arg == "-i" || arg == "--index") && argc > i)
			outputIndex = atoi(argv[++i]);
		else if (arg == "-p" || arg == "--prefix")
			outputPrefix = true;
		else if (arg == "-z" || arg == "--no-zeroes")
			clearZeroes = true;
		else if (arg == "-n" || arg == "--no-nulls")
			clearNulls = true;
		else if (arg == "-x" || arg == "--hex")
			encoding = Encoding::Hex;
		else if (arg == "-d" || arg == "--decimal" || arg == "--dec")
			encoding = Encoding::Decimal;
		else if (arg == "-b" || arg == "--binary" || arg == "--bin")
			encoding = Encoding::Binary;
		else if (arg == "-V" || arg == "--version")
			version();
		else
			methodName = arg;
	}

	string abi;
	if (abiFile == "--")
		for (int i = cin.get(); i != -1; i = cin.get())
			abi.push_back((char)i);
	else if (!abiFile.empty())
		abi = contentsString(abiFile);

	if (mode == Mode::Encode)
	{
		(void)encoding;
		(void)outputPrefix;
		(void)clearZeroes;
		(void)clearNulls;
		(void)outputIndex;
	}
	else if (mode == Mode::Decode)
	{
	}

	return 0;
}
