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
		<< "Usage abi enc <method_name> (<arg1>, (<arg2>, ... ))" << endl
		<< "      abi enc -a <abi.json> <method_name> (<arg1>, (<arg2>, ... ))" << endl
		<< "      abi dec -a <abi.json> [ <signature> | <unique_method_name> ]" << endl
		<< "Options:" << endl
		<< "    -a,--abi-file <filename>  Specify the JSON ABI file." << endl
		<< "    -h,--help  Print this help message and exit." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "Input options:" << endl
		<< "    -p,--prefix  Require all input formats to be prefixed e.g. 0x for hex, . for decimal, @ for binary." << endl
		<< "    -P,--no-prefix  Require no input format to be prefixed." << endl
		<< "    -t,--typing  Require all arguments to be typed e.g. b32: (bytes32), u64: (uint64), b[]: (byte[]), i: (int256)." << endl
		<< "    -T,--no-typing  Require no arguments to be typed." << endl
		<< "Output options:" << endl
		<< "    -i,--index <n>  Output only the nth (counting from 0) return value." << endl
		<< "    -d,--decimal  All data should be displayed as decimal." << endl
		<< "    -x,--hex  Display all data as hex." << endl
		<< "    -b,--binary  Display all data as binary." << endl
		<< "    -p,--prefix  Prefix by a base identifier." << endl
		<< "    -z,--no-zeroes  Remove any leading zeroes from the data." << endl
		<< "    -n,--no-nulls  Remove any trailing nulls from the data." << endl
		;
	exit(0);
}

void version()
{
	cout << "abi version " << dev::Version << endl;
	exit(0);
}

enum class Mode
{
	Encode,
	Decode
};

enum class Encoding
{
	Auto,
	Decimal,
	Hex,
	Binary,
};

enum class Tristate
{
	False = false,
	True = true,
	Mu
};

enum class Format
{
	Binary,
	Hex,
	Decimal
};

struct InvalidUserString: public Exception {};
struct InvalidFormat: public Exception {};

enum class Base
{
	Unknown,
	Bytes,
	Address,
	Int,
	Uint,
	Fixed
};

struct ABIType
{
	Base base = Base::Unknown;
	unsigned size = 32;
	unsigned ssize = 0;
	vector<int> dims;
	ABIType() = default;
	ABIType(std::string const& _s)
	{
		if (_s.size() < 1)
			return;
		switch (_s[0])
		{
		case 'b': base = Base::Bytes; break;
		case 'a': base = Base::Address; break;
		case 'i': base = Base::Int; break;
		case 'u': base = Base::Uint; break;
		case 'f': base = Base::Fixed; break;
		default: throw InvalidFormat();
		}
		if (_s.size() < 2)
		{
			if (base == Base::Fixed)
				size = ssize = 16;
			else
				size = 32;
			return;
		}
		if (_s.find_first_of('x') == string::npos)
			size = stoi(_s.substr(1));
		else
		{
			size = stoi(_s.substr(1, _s.find_first_of('x') - 1));
			ssize = stoi(_s.substr(_s.find_first_of('x') + 1));
		}
	}

	string canon() const
	{
		string ret;
		switch (base)
		{
		case Base::Bytes: ret = "bytes" + toString(size); break;
		case Base::Address: ret = "address"; break;
		case Base::Int: ret = "int" + toString(size); break;
		case Base::Uint: ret = "uint" + toString(size); break;
		case Base::Fixed: ret = "fixed" + toString(size) + "x" + toString(ssize); break;
		default: throw InvalidFormat();
		}
		for (int i: dims)
			ret += "[" + ((i > -1) ? toString(i) : "") + "]";
		return ret;
	}

	void noteHexInput(unsigned _nibbles) { if (base == Base::Unknown) { if (_nibbles == 40) base = Base::Address; else { base = Base::Bytes; size = _nibbles / 2; } } }
	void noteBinaryInput() { if (base == Base::Unknown) { base = Base::Bytes; size = 32; } }
	void noteDecimalInput() { if (base == Base::Unknown) { base = Base::Uint; size = 32; } }
};

tuple<bytes, ABIType, Format> fromUser(std::string const& _arg, Tristate _prefix, Tristate _typing)
{
	ABIType type;
	string val;
	if (_typing == Tristate::True || (_typing == Tristate::Mu && _arg.find(':') != string::npos))
	{
		if (_arg.find(':') == string::npos)
			throw InvalidUserString();
		type = ABIType(_arg.substr(0, _arg.find(':')));
		val = _arg.substr(_arg.find(':') + 1);
	}
	else
		val = _arg;

	if (_prefix != Tristate::False)
	{
		if (val.substr(0, 2) == "0x")
		{
			type.noteHexInput(val.size() - 2);
			return make_tuple(fromHex(val), type, Format::Hex);
		}
		if (val.substr(0, 1) == ".")
		{
			type.noteDecimalInput();
			return make_tuple(toCompactBigEndian(bigint(val.substr(1))), type, Format::Decimal);
		}
		if (val.substr(0, 1) == "@")
		{
			type.noteBinaryInput();
			return make_tuple(asBytes(val.substr(1)), type, Format::Binary);
		}
	}
	if (_prefix != Tristate::True)
	{
		if (_arg.find_first_not_of("0123456789") == string::npos)
		{
			type.noteDecimalInput();
			return make_tuple(toCompactBigEndian(bigint(val)), type, Format::Decimal);
		}
		if (_arg.find_first_not_of("0123456789abcdefABCDEF") == string::npos)
		{
			type.noteHexInput(val.size());
			return make_tuple(fromHex(val), type, Format::Hex);
		}
		type.noteBinaryInput();
		return make_tuple(asBytes(_arg), type, Format::Binary);
	}
	throw InvalidUserString();
}

void userOutput(ostream& _out, bytes const& _data, Encoding _e)
{
	switch (_e)
	{
	case Encoding::Binary:
		_out.write((char const*)_data.data(), _data.size());
		break;
	default:
		_out << toHex(_data) << endl;
	}
}

bytes aligned(bytes const& _b, ABIType _t, Format _f, unsigned _length)
{
	(void)_t;
	bytes ret = _b;
	while (ret.size() < _length)
		if (_f == Format::Binary)
			ret.push_back(0);
		else
			ret.insert(ret.begin(), 0);
	while (ret.size() > _length)
		if (_f == Format::Binary)
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
	Tristate prefix = Tristate::Mu;
	Tristate typePrefix = Tristate::Mu;
	bool clearZeroes = false;
	bool clearNulls = false;
	bool verbose = false;
	int outputIndex = -1;
	vector<tuple<bytes, ABIType, Format>> args;

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
			prefix = Tristate::True;
		else if (arg == "-P" || arg == "--no-prefix")
			prefix = Tristate::False;
		else if (arg == "-t" || arg == "--typing")
			typePrefix = Tristate::True;
		else if (arg == "-T" || arg == "--no-typing")
			typePrefix = Tristate::False;
		else if (arg == "-z" || arg == "--no-zeroes")
			clearZeroes = true;
		else if (arg == "-n" || arg == "--no-nulls")
			clearNulls = true;
		else if (arg == "-v" || arg == "--verbose")
			verbose = true;
		else if (arg == "-x" || arg == "--hex")
			encoding = Encoding::Hex;
		else if (arg == "-d" || arg == "--decimal" || arg == "--dec")
			encoding = Encoding::Decimal;
		else if (arg == "-b" || arg == "--binary" || arg == "--bin")
			encoding = Encoding::Binary;
		else if (arg == "-v" || arg == "--verbose")
			version();
		else if (arg == "-V" || arg == "--version")
			version();
		else if (method.empty())
			method = arg;
		else
			args.push_back(fromUser(arg, prefix, typePrefix));
	}

	string abi;
	if (abiFile == "--")
		for (int i = cin.get(); i != -1; i = cin.get())
			abi.push_back((char)i);
	else if (!abiFile.empty())
		abi = contentsString(abiFile);

	if (mode == Mode::Encode)
	{
		bytes ret;
		if (abi.empty())
		{
			if (!method.empty())
			{
				string methodArgs;
				for (auto const& arg: args)
					methodArgs += (methodArgs.empty() ? "" : ",") + get<1>(arg).canon();
				ret = FixedHash<4>(sha3(method + "(" + methodArgs + ")")).asBytes();
				if (verbose)
					cerr << "Method signature: " << (method + "(" + methodArgs + ")") << endl;
			}
			for (tuple<bytes, ABIType, Format> const& arg: args)
				ret += aligned(get<0>(arg), get<1>(arg), get<2>(arg), 32);
		}
		else
		{
			// TODO: read abi.
		}
		userOutput(cout, ret, encoding);
	}
	else if (mode == Mode::Decode)
	{
		// TODO: read abi to determine output format.
		(void)encoding;
		(void)clearZeroes;
		(void)clearNulls;
		(void)outputIndex;
	}

	return 0;
}
