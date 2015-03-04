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
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/SHA3.h>
#include "base64.h"
using namespace std;
using namespace dev;

void help()
{
	cout
		<< "Usage rlp [OPTIONS] [ <file> | -- ]" << endl
		<< "Options:" << endl
		<< "    -V,--version  Show the version and exit." << endl
		;
	exit(0);
}

void version()
{
	cout << "rlp version " << dev::Version << endl;
	exit(0);
}

enum class Mode {
	ListArchive,
	ExtractArchive,
	Render,
};

enum class Encoding {
	Auto,
	Hex,
	Base64,
	Binary,
};

bool isAscii(string const& _s)
{
	for (char c: _s)
		if (c < 32)
			return false;
	return true;
}

class RLPStreamer
{
public:
	struct Prefs
	{
		string indent = "  ";
		bool hexInts = false;
		bool forceString = false;
		bool escapeAll = false;
		bool forceHex = false;
	};

	RLPStreamer(ostream& _out, Prefs _p): m_out(_out), m_prefs(_p) {}

	void output(RLP const& _d, unsigned _level = 0)
	{
		if (_d.isNull())
			m_out << "null";
		else if (_d.isInt())
			if (m_prefs.hexInts)
				m_out << toHex(toCompactBigEndian(_d.toInt<bigint>(RLP::LaisezFaire)));
			else
				m_out << _d.toInt<bigint>(RLP::LaisezFaire);
		else if (_d.isData())
			if (m_prefs.forceString || (!m_prefs.forceHex && isAscii(_d.toString())))
				m_out << escaped(_d.toString(), m_prefs.escapeAll);
			else
				m_out << toHex(_d.data());
		else if (_d.isList())
		{
			m_out << "[";
			string newline = "\n";
			for (unsigned i = 0; i < _level + 1; ++i)
				newline += m_prefs.indent;
			int j = 0;
			for (auto i: _d)
			{
				m_out << (j++ ?
					(m_prefs.indent.empty() ? ", " : ("," + newline)) :
					(m_prefs.indent.empty() ? " " : newline));
				output(i, _level + 1);
			}
			newline = newline.substr(0, newline.size() - m_prefs.indent.size());
			m_out << (m_prefs.indent.empty() ? (j ? " ]" : "]") : (j ? newline + "]" : "]"));
		}
	}

private:
	std::ostream& m_out;
	Prefs m_prefs;
};

int main(int argc, char** argv)
{
	Encoding encoding = Encoding::Auto;
	Mode mode = Mode::Render;
	string inputFile = "--";
	bool lenience = false;
	RLPStreamer::Prefs prefs;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-r" || arg == "--render")
			mode = Mode::Render;
		else if ((arg == "-i" || arg == "--indent") && argc > i)
			prefs.indent = argv[++i];
		else if (arg == "--hex-ints")
			prefs.hexInts = true;
		else if (arg == "--force-string")
			prefs.forceString = true;
		else if (arg == "--force-hex")
			prefs.forceHex = true;
		else if (arg == "--force-escape")
			prefs.escapeAll = true;
		else if (arg == "-l" || arg == "--list-archive")
			mode = Mode::ListArchive;
		else if (arg == "-e" || arg == "--extract-archive")
			mode = Mode::ExtractArchive;
		else if (arg == "-L" || arg == "--lenience")
			lenience = true;
		else if (arg == "-V" || arg == "--version")
			version();
		else if (arg == "-x" || arg == "--hex" || arg == "--base-16")
			encoding = Encoding::Hex;
		else if (arg == "--64" || arg == "--base-64")
			encoding = Encoding::Base64;
		else if (arg == "-b" || arg == "--bin" || arg == "--base-256")
			encoding = Encoding::Binary;
		else
			inputFile = arg;
	}

	bytes in;
	if (inputFile == "--")
		for (int i = cin.get(); i != -1; i = cin.get())
			in.push_back((byte)i);
	else
		in = contents(inputFile);
	if (encoding == Encoding::Auto)
	{
		encoding = Encoding::Hex;
		for (char b: in)
			if (b != '\n' && b != ' ' && b != '\t')
			{
				if (encoding == Encoding::Hex && (b < '0' || b > '9' ) && (b < 'a' || b > 'f' ) && (b < 'A' || b > 'F' ))
				{
					cerr << "'" << b << "':" << (int)b << endl;
					encoding = Encoding::Base64;
				}
				if (encoding == Encoding::Base64 && (b < '0' || b > '9' ) && (b < 'a' || b > 'z' ) && (b < 'A' || b > 'Z' ) && b != '+' && b != '/')
				{
					encoding = Encoding::Binary;
					break;
				}
			}
	}
	bytes b;
	switch (encoding)
	{
	case Encoding::Hex:
	{
		string s = asString(in);
		boost::algorithm::replace_all(s, " ", "");
		boost::algorithm::replace_all(s, "\n", "");
		boost::algorithm::replace_all(s, "\t", "");
		b = fromHex(s);
		break;
	}
	case Encoding::Base64:
	{
		string s = asString(in);
		boost::algorithm::replace_all(s, " ", "");
		boost::algorithm::replace_all(s, "\n", "");
		boost::algorithm::replace_all(s, "\t", "");
		b = base64_decode(s);
		break;
	}
	default:
		swap(b, in);
		break;
	}

	try
	{
	RLP rlp(b);
	switch (mode)
	{
	case Mode::ListArchive:
	{
		if (!rlp.isList())
		{
			cout << "Error: Invalid format; RLP data is not a list." << endl;
			exit(1);
		}
		cout << rlp.itemCount() << " items:" << endl;
		for (auto i: rlp)
		{
			if (!i.isData())
			{
				cout << "Error: Invalid format; RLP list item is not data." << endl;
				if (!lenience)
					exit(1);
			}
			cout << "    " << i.size() << " bytes: " << sha3(i.data()) << endl;
		}
		break;
	}
	case Mode::ExtractArchive:
	{
		if (!rlp.isList())
		{
			cout << "Error: Invalid format; RLP data is not a list." << endl;
			exit(1);
		}
		cout << rlp.itemCount() << " items:" << endl;
		for (auto i: rlp)
		{
			if (!i.isData())
			{
				cout << "Error: Invalid format; RLP list item is not data." << endl;
				if (!lenience)
					exit(1);
			}
			ofstream fout;
			fout.open(toString(sha3(i.data())));
			fout.write(reinterpret_cast<char const*>(i.data().data()), i.data().size());
		}
		break;
	}
	case Mode::Render:
	{
		RLPStreamer s(cout, prefs);
		s.output(rlp);
		cout << endl;
		break;
	}
	default:;
	}
	}
	catch (...)
	{
		cerr << "Error: Invalid format; bad RLP." << endl;
		exit(1);
	}

	return 0;
}
