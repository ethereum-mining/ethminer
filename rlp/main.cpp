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
#include <libdevcore/SHA3.h>
using namespace std;
using namespace dev;
namespace js = json_spirit;

void help()
{
	cout
		<< "Usage rlp [OPTIONS] [ <file> | -- ]" << endl
		<< "Options:" << endl
		<< "    -r,--render  Render the given RLP. Options:" << endl
		<< "      --indent <string>  Use string as the level indentation (default '  ')." << endl
		<< "      --hex-ints  Render integers in hex." << endl
		<< "      --string-ints  Render integers in the same way as strings." << endl
		<< "      --ascii-strings  Render data as C-style strings or hex depending on content being ASCII." << endl
		<< "      --force-string  Force all data to be rendered as C-style strings." << endl
		<< "      --force-escape  When rendering as C-style strings, force all characters to be escaped." << endl
		<< "      --force-hex  Force all data to be rendered as raw hex." << endl
		<< "    -l,--list-archive  List the items in the RLP list by hash and size." << endl
		<< "    -e,--extract-archive  Extract all items in the RLP list, named by hash." << endl
		<< "    -c,--create  Given a simplified JSON string, output the RLP." << endl
		<< "General options:" << endl
		<< "    -L,--lenience  Try not to bomb out early if possible." << endl
		<< "    -x,--hex,--base-16  Treat input RLP as hex encoded data." << endl
		<< "    -k,--keccak  Output Keccak-256 hash only." << endl
		<< "    --64,--base-64  Treat input RLP as base-64 encoded data." << endl
		<< "    -b,--bin,--base-256  Treat input RLP as raw binary data." << endl
		<< "    -h,--help  Print this help message and exit." << endl
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
	Create
};

enum class Encoding {
	Auto,
	Hex,
	Base64,
	Binary,
	Keccak,
};

bool isAscii(string const& _s)
{
	// Always hex-encode anything beginning with 0x to avoid ambiguity.
	if (_s.size() >= 2 && _s.substr(0, 2) == "0x")
		return false;

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
		string indent;
		bool hexInts = false;
		bool stringInts = true;
		bool hexPrefix = true;
		bool forceString = false;
		bool escapeAll = false;
		bool forceHex = true;
	};

	RLPStreamer(ostream& _out, Prefs _p): m_out(_out), m_prefs(_p) {}

	void output(RLP const& _d, unsigned _level = 0)
	{
		if (_d.isNull())
			m_out << "null";
		else if (_d.isInt() && !m_prefs.stringInts)
			if (m_prefs.hexInts)
				m_out << (m_prefs.hexPrefix ? "0x" : "") << toHex(toCompactBigEndian(_d.toInt<bigint>(RLP::LaissezFaire), 1), 1);
			else
				m_out << _d.toInt<bigint>(RLP::LaissezFaire);
		else if (_d.isData() || (_d.isInt() && m_prefs.stringInts))
			if (m_prefs.forceString || (!m_prefs.forceHex && isAscii(_d.toString())))
				m_out << escaped(_d.toString(), m_prefs.escapeAll);
			else
				m_out << "\"" << (m_prefs.hexPrefix ? "0x" : "") << toHex(_d.toBytes()) << "\"";
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
		else if (arg == "-c" || arg == "--create")
			mode = Mode::Create;
		else if ((arg == "-i" || arg == "--indent") && argc > i)
			prefs.indent = argv[++i];
		else if (arg == "--hex-ints")
			prefs.hexInts = true;
		else if (arg == "--string-ints")
			prefs.stringInts = true;
		else if (arg == "--ascii-strings")
			prefs.forceString = prefs.forceHex = false;
		else if (arg == "--force-string")
			prefs.forceString = true;
		else if (arg == "--force-hex")
			prefs.forceHex = true, prefs.forceString = false;
		else if (arg == "--force-escape")
			prefs.escapeAll = true;
		else if (arg == "-n" || arg == "--nice")
			prefs.forceString = true, prefs.stringInts = false, prefs.forceHex = false, prefs.indent = "  ";
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
		else if (arg == "-k" || arg == "--keccak")
			encoding = Encoding::Keccak;
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

	bytes b;

	if (mode != Mode::Create)
	{
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
			b = fromBase64(s);
			break;
		}
		default:
			swap(b, in);
			break;
		}
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
		case Mode::Create:
		{
			vector<js::mValue> v(1);
			try {
				js::read_string(asString(in), v[0]);
			}
			catch (...)
			{
				cerr << "Error: Invalid format; bad JSON." << endl;
				exit(1);
			}
			RLPStream out;
			while (!v.empty())
			{
				auto vb = v.back();
				v.pop_back();
				switch (vb.type())
				{
				case js::array_type:
				{
					js::mArray a = vb.get_array();
					out.appendList(a.size());
					for (int i = a.size() - 1; i >= 0; --i)
						v.push_back(a[i]);
					break;
				}
				case js::str_type:
				{
					string const& s = vb.get_str();
					if (s.size() >= 2 && s.substr(0, 2) == "0x")
						out << fromHex(s);
					else
					{
						// assume it's a normal JS escaped string.
						bytes ss;
						ss.reserve(s.size());
						for (unsigned i = 0; i < s.size(); ++i)
							if (s[i] == '\\' && i + 1 < s.size())
							{
								if (s[++i] == 'x' && i + 2 < s.size())
									ss.push_back(fromHex(s.substr(i, 2))[0]);
							}
							else if (s[i] != '\\')
								ss.push_back((byte)s[i]);
						out << ss;
					}
					break;
				}
				case js::int_type:
					out << vb.get_int();
					break;
				default:
					cerr << "ERROR: Unsupported type in JSON." << endl;
					if (!lenience)
						exit(1);
				}
			}
			switch (encoding)
			{
			case Encoding::Hex: case Encoding::Auto:
				cout << toHex(out.out()) << endl;
				break;
			case Encoding::Base64:
				cout << toBase64(&out.out()) << endl;
				break;
			case Encoding::Binary:
				cout.write((char const*)out.out().data(), out.out().size());
				break;
			case Encoding::Keccak:
				cout << sha3(out.out()).hex() << endl;
				break;
			}
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
