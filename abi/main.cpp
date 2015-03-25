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
		<< "Input options (enc mode):" << endl
		<< "    -p,--prefix  Require all arguments to be prefixed 0x (hex), . (decimal), # (binary)." << endl
		<< "Output options (dec mode):" << endl
		<< "    -i,--index <n>  Output only the nth (counting from 0) return value." << endl
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

struct InvalidUserString: public Exception {};

pair<bytes, bool> fromUser(std::string const& _arg, bool _requirePrefix)
{
	if (_requirePrefix)
	{
		if (_arg.substr(0, 2) == "0x")
			return make_pair(fromHex(_arg), false);
		if (_arg.substr(0, 1) == ".")
			return make_pair(toCompactBigEndian(bigint(_arg.substr(1))), false);
		if (_arg.substr(0, 1) == "#")
			return make_pair(asBytes(_arg.substr(1)), true);
		throw InvalidUserString();
	}
	else
	{
		if (_arg.substr(0, 2) == "0x")
			return make_pair(fromHex(_arg), false);
		if (_arg.find_first_not_of("0123456789"))
			return make_pair(toCompactBigEndian(bigint(_arg)), false);
		return make_pair(asBytes(_arg), true);
	}
}

bytes aligned(bytes const& _b, bool _left, unsigned _length)
{
	bytes ret = _b;
	while (ret.size() < _length)
		if (_left)
			ret.push_back(0);
		else
			ret.insert(ret.begin(), 0);
	while (ret.size() > _length)
		if (_left)
			ret.pop_back();
		else
			ret.erase(ret.begin());
	return ret;
}

int main(int argc, char** argv)
{
	Encoding encoding = Encoding::Auto;
	Mode mode = Mode::Encode;
	string abiFile;
	string method;
	bool prefix = false;
	bool clearZeroes = false;
	bool clearNulls = false;
	int outputIndex = -1;
	vector<pair<bytes, bool>> args;

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
			prefix = true;
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
		else if (method.empty())
			method = arg;
		else
			args.push_back(fromUser(arg, prefix));
	}

	string abi;
	if (abiFile == "--")
		for (int i = cin.get(); i != -1; i = cin.get())
			abi.push_back((char)i);
	else if (!abiFile.empty())
		abi = contentsString(abiFile);

	if (mode == Mode::Encode)
	{
		if (abi.empty())
		{
			bytes ret;
			if (!method.empty())
				ret = FixedHash<32>(sha3(method)).asBytes();
			if (method.empty())
				for (pair<bytes, bool> const& arg: args)
					ret += aligned(arg.first, arg.second, 32);
		}
		else
		{
			// TODO: read abi.
		}
	}
	else if (mode == Mode::Decode)
	{
		(void)encoding;
		(void)clearZeroes;
		(void)clearNulls;
		(void)outputIndex;
	}

	return 0;
}
