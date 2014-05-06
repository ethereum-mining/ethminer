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
/** @file Instruction.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Instruction.h"

#include <boost/algorithm/string.hpp>
#include <libethcore/Log.h>
#include "CommonEth.h"
using namespace std;
using namespace eth;


const std::map<std::string, Instruction> eth::c_instructions =
{
	{ "STOP", Instruction::STOP },
	{ "ADD", Instruction::ADD },
	{ "SUB", Instruction::SUB },
	{ "MUL", Instruction::MUL },
	{ "DIV", Instruction::DIV },
	{ "SDIV", Instruction::SDIV },
	{ "MOD", Instruction::MOD },
	{ "SMOD", Instruction::SMOD },
	{ "EXP", Instruction::EXP },
	{ "NEG", Instruction::NEG },
	{ "LT", Instruction::LT },
	{ "GT", Instruction::GT },
	{ "SLT", Instruction::SLT },
	{ "SGT", Instruction::SGT },
	{ "EQ", Instruction::EQ },
	{ "NOT", Instruction::NOT },
	{ "AND", Instruction::AND },
	{ "OR", Instruction::OR },
	{ "XOR", Instruction::XOR },
	{ "BYTE", Instruction::BYTE },
	{ "SHA3", Instruction::SHA3 },
	{ "ADDRESS", Instruction::ADDRESS },
	{ "BALANCE", Instruction::BALANCE },
	{ "ORIGIN", Instruction::ORIGIN },
	{ "CALLER", Instruction::CALLER },
	{ "CALLVALUE", Instruction::CALLVALUE },
	{ "CALLDATALOAD", Instruction::CALLDATALOAD },
	{ "CALLDATASIZE", Instruction::CALLDATASIZE },
	{ "CALLDATACOPY", Instruction::CALLDATACOPY },
	{ "CODESIZE", Instruction::CODESIZE },
	{ "CODECOPY", Instruction::CODECOPY },
	{ "GASPRICE", Instruction::GASPRICE },
	{ "PREVHASH", Instruction::PREVHASH },
	{ "COINBASE", Instruction::COINBASE },
	{ "TIMESTAMP", Instruction::TIMESTAMP },
	{ "NUMBER", Instruction::NUMBER },
	{ "DIFFICULTY", Instruction::DIFFICULTY },
	{ "GASLIMIT", Instruction::GASLIMIT },
	{ "POP", Instruction::POP },
	{ "DUP", Instruction::DUP },
	{ "SWAP", Instruction::SWAP },
	{ "MLOAD", Instruction::MLOAD },
	{ "MSTORE", Instruction::MSTORE },
	{ "MSTORE8", Instruction::MSTORE8 },
	{ "SLOAD", Instruction::SLOAD },
	{ "SSTORE", Instruction::SSTORE },
	{ "JUMP", Instruction::JUMP },
	{ "JUMPI", Instruction::JUMPI },
	{ "PC", Instruction::PC },
	{ "MEMSIZE", Instruction::MEMSIZE },
	{ "GAS", Instruction::GAS },
	{ "PUSH1", Instruction::PUSH1 },
	{ "PUSH2", Instruction::PUSH2 },
	{ "PUSH3", Instruction::PUSH3 },
	{ "PUSH4", Instruction::PUSH4 },
	{ "PUSH5", Instruction::PUSH5 },
	{ "PUSH6", Instruction::PUSH6 },
	{ "PUSH7", Instruction::PUSH7 },
	{ "PUSH8", Instruction::PUSH8 },
	{ "PUSH9", Instruction::PUSH9 },
	{ "PUSH10", Instruction::PUSH10 },
	{ "PUSH11", Instruction::PUSH11 },
	{ "PUSH12", Instruction::PUSH12 },
	{ "PUSH13", Instruction::PUSH13 },
	{ "PUSH14", Instruction::PUSH14 },
	{ "PUSH15", Instruction::PUSH15 },
	{ "PUSH16", Instruction::PUSH16 },
	{ "PUSH17", Instruction::PUSH17 },
	{ "PUSH18", Instruction::PUSH18 },
	{ "PUSH19", Instruction::PUSH19 },
	{ "PUSH20", Instruction::PUSH20 },
	{ "PUSH21", Instruction::PUSH21 },
	{ "PUSH22", Instruction::PUSH22 },
	{ "PUSH23", Instruction::PUSH23 },
	{ "PUSH24", Instruction::PUSH24 },
	{ "PUSH25", Instruction::PUSH25 },
	{ "PUSH26", Instruction::PUSH26 },
	{ "PUSH27", Instruction::PUSH27 },
	{ "PUSH28", Instruction::PUSH28 },
	{ "PUSH29", Instruction::PUSH29 },
	{ "PUSH30", Instruction::PUSH30 },
	{ "PUSH31", Instruction::PUSH31 },
	{ "PUSH32", Instruction::PUSH32 },
	{ "CREATE", Instruction::CREATE },
	{ "CALL", Instruction::CALL },
	{ "RETURN", Instruction::RETURN },
	{ "SUICIDE", Instruction::SUICIDE }
};

const std::map<Instruction, InstructionInfo> eth::c_instructionInfo =
{ //                                           Add, Args, Ret
	{ Instruction::STOP,         { "STOP",         0, 0, 0 } },
	{ Instruction::ADD,          { "ADD",          0, 2, 1 } },
	{ Instruction::SUB,          { "SUB",          0, 2, 1 } },
	{ Instruction::MUL,          { "MUL",          0, 2, 1 } },
	{ Instruction::DIV,          { "DIV",          0, 2, 1 } },
	{ Instruction::SDIV,         { "SDIV",         0, 2, 1 } },
	{ Instruction::MOD,          { "MOD",          0, 2, 1 } },
	{ Instruction::SMOD,         { "SMOD",         0, 2, 1 } },
	{ Instruction::EXP,          { "EXP",          0, 2, 1 } },
	{ Instruction::NEG,          { "NEG",          0, 1, 1 } },
	{ Instruction::LT,           { "LT",           0, 2, 1 } },
	{ Instruction::GT,           { "GT",           0, 2, 1 } },
	{ Instruction::SLT,          { "SLT",          0, 2, 1 } },
	{ Instruction::SGT,          { "SGT",          0, 2, 1 } },
	{ Instruction::EQ,           { "EQ",           0, 2, 1 } },
	{ Instruction::NOT,          { "NOT",          0, 1, 1 } },
	{ Instruction::AND,          { "AND",          0, 2, 1 } },
	{ Instruction::OR,           { "OR",           0, 2, 1 } },
	{ Instruction::XOR,          { "XOR",          0, 2, 1 } },
	{ Instruction::BYTE,         { "BYTE",         0, 2, 1 } },
	{ Instruction::SHA3,         { "SHA3",         0, 2, 1 } },
	{ Instruction::ADDRESS,      { "ADDRESS",      0, 0, 1 } },
	{ Instruction::BALANCE,      { "BALANCE",      0, 1, 1 } },
	{ Instruction::ORIGIN,       { "ORIGIN",       0, 0, 1 } },
	{ Instruction::CALLER,       { "CALLER",       0, 0, 1 } },
	{ Instruction::CALLVALUE,    { "CALLVALUE",    0, 0, 1 } },
	{ Instruction::CALLDATALOAD, { "CALLDATALOAD", 0, 1, 1 } },
	{ Instruction::CALLDATASIZE, { "CALLDATASIZE", 0, 0, 1 } },
	{ Instruction::CALLDATACOPY, { "CALLDATACOPY", 0, 3, 0 } },
	{ Instruction::CODESIZE,     { "CODESIZE",     0, 0, 1 } },
	{ Instruction::CODECOPY,     { "CODECOPY",     0, 3, 0 } },
	{ Instruction::GASPRICE,     { "GASPRICE",     0, 0, 1 } },
	{ Instruction::PREVHASH,     { "PREVHASH",     0, 0, 1 } },
	{ Instruction::COINBASE,     { "COINBASE",     0, 0, 1 } },
	{ Instruction::TIMESTAMP,    { "TIMESTAMP",    0, 0, 1 } },
	{ Instruction::NUMBER,       { "NUMBER",       0, 0, 1 } },
	{ Instruction::DIFFICULTY,   { "DIFFICULTY",   0, 0, 1 } },
	{ Instruction::GASLIMIT,     { "GASLIMIT",     0, 0, 1 } },
	{ Instruction::POP,          { "POP",          0, 1, 0 } },
	{ Instruction::DUP,          { "DUP",          0, 1, 2 } },
	{ Instruction::SWAP,         { "SWAP",         0, 2, 2 } },
	{ Instruction::MLOAD,        { "MLOAD",        0, 1, 1 } },
	{ Instruction::MSTORE,       { "MSTORE",       0, 2, 0 } },
	{ Instruction::MSTORE8,      { "MSTORE8",      0, 2, 0 } },
	{ Instruction::SLOAD,        { "SLOAD",        0, 1, 1 } },
	{ Instruction::SSTORE,       { "SSTORE",       0, 2, 0 } },
	{ Instruction::JUMP,         { "JUMP",         0, 1, 0 } },
	{ Instruction::JUMPI,        { "JUMPI",        0, 2, 0 } },
	{ Instruction::PC,           { "PC",           0, 0, 1 } },
	{ Instruction::MEMSIZE,      { "MEMSIZE",      0, 0, 1 } },
	{ Instruction::GAS,          { "GAS",          0, 0, 1 } },
	{ Instruction::PUSH1,        { "PUSH1",        1, 0, 1 } },
	{ Instruction::PUSH2,        { "PUSH2",        2, 0, 1 } },
	{ Instruction::PUSH3,        { "PUSH3",        3, 0, 1 } },
	{ Instruction::PUSH4,        { "PUSH4",        4, 0, 1 } },
	{ Instruction::PUSH5,        { "PUSH5",        5, 0, 1 } },
	{ Instruction::PUSH6,        { "PUSH6",        6, 0, 1 } },
	{ Instruction::PUSH7,        { "PUSH7",        7, 0, 1 } },
	{ Instruction::PUSH8,        { "PUSH8",        8, 0, 1 } },
	{ Instruction::PUSH9,        { "PUSH9",        9, 0, 1 } },
	{ Instruction::PUSH10,       { "PUSH10",       10, 0, 1 } },
	{ Instruction::PUSH11,       { "PUSH11",       11, 0, 1 } },
	{ Instruction::PUSH12,       { "PUSH12",       12, 0, 1 } },
	{ Instruction::PUSH13,       { "PUSH13",       13, 0, 1 } },
	{ Instruction::PUSH14,       { "PUSH14",       14, 0, 1 } },
	{ Instruction::PUSH15,       { "PUSH15",       15, 0, 1 } },
	{ Instruction::PUSH16,       { "PUSH16",       16, 0, 1 } },
	{ Instruction::PUSH17,       { "PUSH17",       17, 0, 1 } },
	{ Instruction::PUSH18,       { "PUSH18",       18, 0, 1 } },
	{ Instruction::PUSH19,       { "PUSH19",       19, 0, 1 } },
	{ Instruction::PUSH20,       { "PUSH20",       20, 0, 1 } },
	{ Instruction::PUSH21,       { "PUSH21",       21, 0, 1 } },
	{ Instruction::PUSH22,       { "PUSH22",       22, 0, 1 } },
	{ Instruction::PUSH23,       { "PUSH23",       23, 0, 1 } },
	{ Instruction::PUSH24,       { "PUSH24",       24, 0, 1 } },
	{ Instruction::PUSH25,       { "PUSH25",       25, 0, 1 } },
	{ Instruction::PUSH26,       { "PUSH26",       26, 0, 1 } },
	{ Instruction::PUSH27,       { "PUSH27",       27, 0, 1 } },
	{ Instruction::PUSH28,       { "PUSH28",       28, 0, 1 } },
	{ Instruction::PUSH29,       { "PUSH29",       29, 0, 1 } },
	{ Instruction::PUSH30,       { "PUSH30",       30, 0, 1 } },
	{ Instruction::PUSH31,       { "PUSH31",       31, 0, 1 } },
	{ Instruction::PUSH32,       { "PUSH32",       32, 0, 1 } },
	{ Instruction::CREATE,       { "CREATE",       0, 3, 1 } },
	{ Instruction::CALL,         { "CALL",         0, 7, 1 } },
	{ Instruction::RETURN,       { "RETURN",       0, 2, 0 } },
	{ Instruction::SUICIDE,      { "SUICIDE",      0, 1, 0} }
};

static string readQuoted(char const*& o_d, char const* _e)
{
	string ret;
	bool escaped = 0;
	for (++o_d; o_d != _e && (escaped || *o_d != '"'); ++o_d)
		if (!escaped && *o_d == '\\')
			escaped = true;
		else
			ret.push_back(*o_d);
	if (o_d != _e)
		++o_d;	// skip last "
	return ret;
}

static u256 readNumeric(string _v, bool _quiet)
{
	u256 x = 1;
	for (auto const& i: units())
		if (boost::algorithm::ends_with(_v, i.second))
		{
			_v = _v.substr(0, _v.size() - i.second.size());
			x = i.first;
			break;
		}
	try
	{
		return x * u256(_v);
	}
	catch (...)
	{
		if (!_quiet)
			cwarn << "Invalid numeric" << _v;
	}
	return 0;
}

bytes eth::assemble(std::string const& _code, bool _quiet)
{
	bytes ret;
	map<string, unsigned> known;
	map<unsigned, string> req;
	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '_' /*&& *d != ':' && *d != '"'*/; ++d) {}
		if (d == e)
			break;

/*		if (*d == '"')
		{
			string s = readQuoted(d, e);
			if (s.size() > 32)
			{
				if (!_quiet)
					cwarn << "String literal > 32 characters. Cropping.";
				s.resize(32);
			}
			h256 valHash;
			memcpy(valHash.data(), s.data(), s.size());
			memset(valHash.data() + s.size(), 0, 32 - s.size());
			ret.push_back((u256)valHash);
		}
		else*/
		{
			char const* s = d;
			for (; d != e && (isalnum(*d) || *d == '_'/* || *d == ':' || *d == '"'*/); ++d) {}

			string t = string(s, d - s);
			if (isdigit(t[0]))
				ret.push_back((byte)readNumeric(t, _quiet));
/*			else if (t.back() == ':')
				known[t.substr(0, t.size() - 1)] = (unsigned)ret.size();
			else
			{
				auto it = c_instructions.find(boost::algorithm::to_upper_copy(t));
				if (it != c_instructions.end())
					ret.push_back((u256)it->second);
				else
				{
					req[(unsigned)ret.size()] = t;
					ret.push_back(0);
				}
			}*/
		}
	}
	for (auto i: req)
		if (known.count(i.second))
			ret[i.first] = known[i.second];
		else
			if (!_quiet)
				cwarn << "Unknown assembler token" << i.second << "at address" << i.first;

	return ret;
}

/// @returns the number of addition bytes required for the PUSH.
static void increaseLocation(bytes& o_code, unsigned _pos, unsigned _inc)
{
	assert(o_code[_pos] == (byte)Instruction::PUSH4);
	bytesRef r(&o_code[1 + _pos], 4);
	toBigEndian(fromBigEndian<uint32_t>(bytesConstRef(&o_code[1 + _pos], 4)) + _inc, r);
}

static void pushLocation(bytes& o_code, uint32_t _locationValue)
{
	o_code.push_back((byte)Instruction::PUSH4);
	o_code.resize(o_code.size() + 4);
	bytesRef r(&o_code[o_code.size() - 4], 4);
	toBigEndian(_locationValue, r);
}

unsigned eth::pushLiteral(bytes& o_code, u256 _literalValue)
{
	unsigned br = max<unsigned>(1, bytesRequired(_literalValue));
	o_code.push_back((byte)Instruction::PUSH1 + br - 1);
	o_code.resize(o_code.size() + br);
	for (unsigned i = 0; i < br; ++i)
	{
		o_code[o_code.size() - 1 - i] = (byte)(_literalValue & 0xff);
		_literalValue >>= 8;
	}
	return br + 1;
}

static void appendCode(bytes& o_code, vector<unsigned>& o_locs, bytes _code, vector<unsigned>& _locs)
{
	o_locs.reserve(o_locs.size() + _locs.size());
	for (auto i: _locs)
	{
		increaseLocation(_code, i, (unsigned)o_code.size());
		o_locs.push_back(i + (unsigned)o_code.size());
	}
	o_code.reserve(o_code.size() + _code.size());
	for (auto i: _code)
		o_code.push_back(i);
}

static int compileLispFragment(char const*& d, char const* e, bool _quiet, bytes& o_code, vector<unsigned>& o_locs, map<string, unsigned>& _vars)
{
	std::map<std::string, Instruction> const c_arith = { { "+", Instruction::ADD }, { "-", Instruction::SUB }, { "*", Instruction::MUL }, { "/", Instruction::DIV }, { "%", Instruction::MOD }, { "&", Instruction::AND }, { "|", Instruction::OR }, { "^", Instruction::XOR } };
	std::map<std::string, pair<Instruction, bool>> const c_binary = { { "<", { Instruction::LT, false } }, { "<=", { Instruction::GT, true } }, { ">", { Instruction::GT, false } }, { ">=", { Instruction::LT, true } }, { "S<", { Instruction::SLT, false } }, { "S<=", { Instruction::SGT, true } }, { "S>", { Instruction::SGT, false } }, { "S>=", { Instruction::SLT, true } }, { "=", { Instruction::EQ, false } }, { "!=", { Instruction::EQ, true } } };
	std::map<std::string, Instruction> const c_unary = { { "!", Instruction::NOT } };
	std::set<char> const c_allowed = { '+', '-', '*', '/', '%', '<', '>', '=', '!', '&', '|', '~' };

	bool exec = false;
	int outs = 0;
	bool seq = false;

	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '(' && *d != ')' && *d != '{' && *d != '}' && *d != '"' && *d != '@' && *d != '[' && !c_allowed.count(*d) && *d != ';'; ++d) {}
		if (d == e)
			break;

		switch (*d)
		{
		case ';':
			for (; d != e && *d != '\n'; ++d) {}
			break;
		case '(':
			exec = true;
			++d;
			break;
		case '{':
			++d;
			while (d != e)
			{
				bytes codes;
				vector<unsigned> locs;
				outs = 0;
				int o;
				if ((o = compileLispFragment(d, e, _quiet, codes, locs, _vars)) > -1)
				{
					for (int i = 0; i < outs; ++i)
						o_code.push_back((byte)Instruction::POP);	// pop additional items off stack for the previous item (final item's returns get left on).
					outs = o;
					appendCode(o_code, o_locs, codes, locs);
				}
				else
					break;
			}
			seq = true;
			break;
		case '}':
			if (seq)
			{
				++d;
				return outs;
			}
			return -1;
		case ')':
			if (exec)
			{
				++d;
				return outs;
			}
			else
				// unexpected - return false as we don't know what to do with it.
				return -1;

		case '@':
		{
			if (exec)
				return -1;
			bool store = false;
			++d;
			if (*d == '@')
			{
				++d;
				store = true;
			}
			bytes codes;
			vector<unsigned> locs;
			if (compileLispFragment(d, e, _quiet, codes, locs, _vars) != 1)
				return -1;
			while (d != e && isspace(*d))
				++d;
			appendCode(o_code, o_locs, codes, locs);
			o_code.push_back((byte)(store ? Instruction::SLOAD : Instruction::MLOAD));
			return 1;
		}
		case '[':
		{
			if (exec)
				return -1;
			bool store = false;
			++d;
			if (*d == '[')
			{
				++d;
				store = true;
			}
			bytes codes;
			vector<unsigned> locs;
			if (compileLispFragment(d, e, _quiet, codes, locs, _vars) != 1)
				return -1;
			while (d != e && isspace(*d))
				++d;

			if (*d != ']')
				return -1;
			++d;
			if (store)
			{
				if (*d != ']')
					return -1;
				++d;
			}

			if (compileLispFragment(d, e, _quiet, o_code, o_locs, _vars) != 1)
				return -1;

			appendCode(o_code, o_locs, codes, locs);
			o_code.push_back((byte)(store ? Instruction::SSTORE: Instruction::MSTORE));
			return 0;
		}
		default:
		{
			bool haveLiteral = false;
			u256 literalValue = 0;
			string t;

			if (*d == '"')
			{
				string s = readQuoted(d, e);
				if (s.size() > 32)
				{
					if (!_quiet)
						cwarn << "String literal > 32 characters. Cropping.";
					s.resize(32);
				}
				h256 valHash;
				memcpy(valHash.data(), s.data(), s.size());
				memset(valHash.data() + s.size(), 0, 32 - s.size());
				literalValue = (u256)valHash;
				haveLiteral = true;
			}
			else
			{
				char const* s = d;
				for (; d != e && (isalnum(*d) || *d == '_' || c_allowed.count(*d)); ++d) {}
				t = string(s, d - s);
				if (isdigit(t[0]))
				{
					literalValue = readNumeric(t, _quiet);
					haveLiteral = true;
				}
			}

			if (haveLiteral)
			{
				bool bareLoad = true;
				if (exec)
				{
					bytes codes;
					vector<unsigned> locs;
					if (compileLispFragment(d, e, _quiet, codes, locs, _vars) != -1)
					{
						appendCode(o_code, o_locs, codes, locs);
						while (compileLispFragment(d, e, _quiet, codes, locs, _vars) != -1)
							if (!_quiet)
								cwarn << "Additional items in bare store. Ignoring.";
						bareLoad = false;
					}
				}
				pushLiteral(o_code, literalValue);
				if (exec)
					o_code.push_back(bareLoad ? (byte)Instruction::SLOAD : (byte)Instruction::SSTORE);
				outs = bareLoad ? 1 : 0;
			}
			else
			{
				boost::algorithm::to_upper(t);
				if (t == "IF")
				{
					// Compile all the code...
					bytes codes[4];
					vector<unsigned> locs[4];
					for (int i = 0; i < 3; ++i)
					{
						int o = compileLispFragment(d, e, _quiet, codes[i], locs[i], _vars);
						if (i == 1)
							outs = o;
						if ((i == 0 && o != 1) || o == -1 || (i == 2 && o != outs))
							return -1;
					}
					if (compileLispFragment(d, e, _quiet, codes[3], locs[3], _vars) != -1)
						return false;

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);

					// Push the positive location.
					unsigned posLocation = (unsigned)o_code.size();
					o_locs.push_back(posLocation);
					pushLocation(o_code, 0);

					// Jump to negative if false.
					o_code.push_back((byte)Instruction::JUMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[2], locs[2]);

					// Jump to end after negative.
					unsigned endLocation = (unsigned)o_code.size();
					o_locs.push_back(endLocation);
					pushLocation(o_code, 0);
					o_code.push_back((byte)Instruction::JUMP);

					// Third fragment - positive.
					increaseLocation(o_code, posLocation, o_code.size());
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// At end now.
					increaseLocation(o_code, endLocation, o_code.size());
				}
				else if (t == "WHEN" || t == "UNLESS")
				{
					outs = 0;
					// Compile all the code...
					bytes codes[3];
					vector<unsigned> locs[3];
					for (int i = 0; i < 2; ++i)
					{
						int o = compileLispFragment(d, e, _quiet, codes[i], locs[i], _vars);
						if (o == -1 || (i == 0 && o != 1))
							return false;
						if (i == 1)
							for (int j = 0; j < o; ++j)
								codes[i].push_back((byte)Instruction::POP);	// pop additional items off stack for the previous item (final item's returns get left on).
					}
					if (compileLispFragment(d, e, _quiet, codes[2], locs[2], _vars) != -1)
						return false;

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);
					if (t == "WHEN")
						o_code.push_back((byte)Instruction::NOT);

					// Push the positive location.
					unsigned endLocation = (unsigned)o_code.size();
					o_locs.push_back(endLocation);
					pushLocation(o_code, 0);

					// Jump to end...
					o_code.push_back((byte)Instruction::JUMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// At end now.
					increaseLocation(o_code, endLocation, o_code.size());
				}
				else if (t == "WHILE")
				{
					outs = 0;
					// Compile all the code...
					bytes codes[3];
					vector<unsigned> locs[3];
					for (int i = 0; i < 2; ++i)
					{
						int o = compileLispFragment(d, e, _quiet, codes[i], locs[i], _vars);
						if (o == -1 || (i == 0 && o != 1))
							return false;
						if (i == 1)
							for (int j = 0; j < o; ++j)
								codes[i].push_back((byte)Instruction::POP);	// pop additional items off stack for the previous item (final item's returns get left on).
					}
					if (compileLispFragment(d, e, _quiet, codes[2], locs[2], _vars) != -1)
						return false;

					unsigned startLocation = (unsigned)o_code.size();

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);
					o_code.push_back((byte)Instruction::NOT);

					// Push the positive location.
					unsigned endInsertion = (unsigned)o_code.size();
					o_locs.push_back(endInsertion);
					pushLocation(o_code, 0);

					// Jump to positive if true.
					o_code.push_back((byte)Instruction::JUMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// Jump to end after negative.
					o_locs.push_back((unsigned)o_code.size());
					pushLocation(o_code, startLocation);
					o_code.push_back((byte)Instruction::JUMP);

					// At end now.
					increaseLocation(o_code, endInsertion, o_code.size());
				}
				else if (t == "FOR")
				{
					compileLispFragment(d, e, _quiet, o_code, o_locs, _vars);
					outs = 0;
					// Compile all the code...
					bytes codes[4];
					vector<unsigned> locs[4];
					for (int i = 0; i < 3; ++i)
					{
						int o = compileLispFragment(d, e, _quiet, codes[i], locs[i], _vars);
						if (o == -1 || (i == 0 && o != 1))
							return false;
						cnote << "FOR " << i << o;
						if (i > 0)
							for (int j = 0; j < o; ++j)
								codes[i].push_back((byte)Instruction::POP);	// pop additional items off stack for the previous item (final item's returns get left on).
					}
					if (compileLispFragment(d, e, _quiet, codes[3], locs[3], _vars) != -1)
						return false;

					unsigned startLocation = (unsigned)o_code.size();

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);
					o_code.push_back((byte)Instruction::NOT);

					// Push the positive location.
					unsigned endInsertion = (unsigned)o_code.size();
					o_locs.push_back(endInsertion);
					pushLocation(o_code, 0);

					// Jump to positive if true.
					o_code.push_back((byte)Instruction::JUMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[2], locs[2]);

					// Third fragment - incrementor.
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// Jump to beginning afterwards.
					o_locs.push_back((unsigned)o_code.size());
					pushLocation(o_code, startLocation);
					o_code.push_back((byte)Instruction::JUMP);

					// At end now.
					increaseLocation(o_code, endInsertion, o_code.size());
				}
				else if (t == "SEQ")
				{
					while (d != e)
					{
						bytes codes;
						vector<unsigned> locs;
						outs = 0;
						int o;
						if ((o = compileLispFragment(d, e, _quiet, codes, locs, _vars)) > -1)
						{
							for (int i = 0; i < outs; ++i)
								o_code.push_back((byte)Instruction::POP);	// pop additional items off stack for the previous item (final item's returns get left on).
							outs = o;
							appendCode(o_code, o_locs, codes, locs);
						}
						else
							break;
					}
				}
				/*else if (t == "CALL")
				{
					if (exec)
					{
						vector<pair<bytes, vector<unsigned>>> codes(1);
						int totalArgs = 0;
						while (d != e)
						{
							int o = compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second, _vars);
							if (o < 1)
								break;
							codes.push_back(pair<bytes, vector<unsigned>>());
							totalArgs += o;
						}
						if (totalArgs < 7)
						{
							cwarn << "Expected at least 7 arguments to CALL; got" << totalArgs << ".";
							break;
						}

						for (auto it = codes.rbegin(); it != codes.rend(); ++it)
							appendCode(o_code, o_locs, it->first, it->second);
						o_code.push_back((byte)Instruction::CALL);
						outs = 1;
					}
				}*/
				else if (t == "MULTI")
				{
					while (d != e)
					{
						bytes codes;
						vector<unsigned> locs;
						outs = 0;
						int o;
						if ((o = compileLispFragment(d, e, _quiet, codes, locs, _vars)) > -1)
						{
							outs += o;
							appendCode(o_code, o_locs, codes, locs);
						}
						else
							break;
					}
				}
				else if (t == "LLL")
				{
					bytes codes;
					vector<unsigned> locs;
					map<string, unsigned> vars;
					if (compileLispFragment(d, e, _quiet, codes, locs, _vars) == -1)
						return false;
					unsigned codeLoc = o_code.size() + 5 + 1;
					o_locs.push_back(o_code.size());
					pushLocation(o_code, codeLoc + codes.size());
					o_code.push_back((byte)Instruction::JUMP);
					for (auto b: codes)
						o_code.push_back(b);

					bytes ncode[1];
					vector<unsigned> nlocs[1];
					if (compileLispFragment(d, e, _quiet, ncode[0], nlocs[0], _vars) != 1)
						return false;

					pushLiteral(o_code, codes.size());
					o_code.push_back((byte)Instruction::DUP);
					int o = compileLispFragment(d, e, _quiet, o_code, o_locs, _vars);
					if (o == 1)
					{
						o_code.push_back((byte)Instruction::LT);
						o_code.push_back((byte)Instruction::NOT);
						o_code.push_back((byte)Instruction::MUL);
						o_code.push_back((byte)Instruction::DUP);
					}
					else if (o != -1)
						return false;

					o_locs.push_back(o_code.size());
					pushLocation(o_code, codeLoc);
					appendCode(o_code, o_locs, ncode[0], nlocs[0]);
					o_code.push_back((byte)Instruction::CODECOPY);
					outs = 1;
				}
				else if (t == "&&")
				{
					vector<bytes> codes;
					vector<vector<unsigned>> locs;
					while (d != e)
					{
						codes.resize(codes.size() + 1);
						locs.resize(locs.size() + 1);
						int o = compileLispFragment(d, e, _quiet, codes.back(), locs.back(), _vars);
						if (o == -1)
							break;
						if (o != 1)
							return false;
					}

					// last one is empty.
					if (codes.size() < 2)
						return false;

					codes.pop_back();
					locs.pop_back();

					vector<unsigned> ends;

					if (codes.size() > 1)
					{
						pushLiteral(o_code, 0);

						for (unsigned i = 1; i < codes.size(); ++i)
						{
							// Check if true - predicate
							appendCode(o_code, o_locs, codes[i - 1], locs[i - 1]);
							o_code.push_back((byte)Instruction::NOT);

							// Push the false location.
							ends.push_back((unsigned)o_code.size());
							o_locs.push_back(ends.back());
							pushLocation(o_code, 0);

							// Jump to end...
							o_code.push_back((byte)Instruction::JUMPI);
						}
						o_code.push_back((byte)Instruction::POP);
					}

					// Check if true - predicate
					appendCode(o_code, o_locs, codes.back(), locs.back());

					// At end now.
					for (auto i: ends)
						increaseLocation(o_code, i, o_code.size());
					outs = 1;
				}
				else if (t == "~")
				{
					if (compileLispFragment(d, e, _quiet, o_code, o_locs, _vars) == 1)
					{
						bytes codes;
						vector<unsigned> locs;
						if (compileLispFragment(d, e, _quiet, codes, locs, _vars) != -1)
							return false;
						pushLiteral(o_code, 1);
						pushLiteral(o_code, 0);
						o_code.push_back((byte)Instruction::SUB);
						o_code.push_back((byte)Instruction::SUB);
						outs = 1;
					}
				}
				else if (t == "||")
				{
					vector<bytes> codes;
					vector<vector<unsigned>> locs;
					while (d != e)
					{
						codes.resize(codes.size() + 1);
						locs.resize(locs.size() + 1);
						{
							int o = compileLispFragment(d, e, _quiet, codes.back(), locs.back(), _vars);
							if (o == -1)
								break;
							if (o != 1)
								return false;
						}
					}

					// last one is empty.
					if (codes.size() < 2)
						return false;

					codes.pop_back();
					locs.pop_back();

					vector<unsigned> ends;

					if (codes.size() > 1)
					{
						pushLiteral(o_code, 1);

						for (unsigned i = 1; i < codes.size(); ++i)
						{
							// Check if true - predicate
							appendCode(o_code, o_locs, codes[i - 1], locs[i - 1]);

							// Push the false location.
							ends.push_back((unsigned)o_code.size());
							o_locs.push_back(ends.back());
							pushLocation(o_code, 0);

							// Jump to end...
							o_code.push_back((byte)Instruction::JUMPI);
						}
						o_code.push_back((byte)Instruction::POP);
					}

					// Check if true - predicate
					appendCode(o_code, o_locs, codes.back(), locs.back());

					// At end now.
					for (auto i: ends)
						increaseLocation(o_code, i, o_code.size());
					outs = 1;
				}
				else
				{
					auto it = c_instructions.find(t);
					if (it != c_instructions.end())
					{
						if (exec)
						{
							vector<pair<bytes, vector<unsigned>>> codes(1);
							int totalArgs = 0;
							while (d != e)
							{
								int o = compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second, _vars);
								if (o < 1)
									break;
								codes.push_back(pair<bytes, vector<unsigned>>());
								totalArgs += o;
							}
							int ea = c_instructionInfo.at(it->second).args;
							if ((ea >= 0 && totalArgs != ea) || (ea < 0 && totalArgs < -ea))
							{
								cwarn << "Expected " << (ea < 0 ? "at least" : "exactly") << abs(ea) << "arguments to operation" << t << "; got" << totalArgs << ".";
								break;
							}

							for (auto it = codes.rbegin(); it != codes.rend(); ++it)
								appendCode(o_code, o_locs, it->first, it->second);
							o_code.push_back((byte)it->second);
							outs = c_instructionInfo.at(it->second).ret;
						}
						else
						{
							o_code.push_back((byte)Instruction::PUSH1);
							o_code.push_back((byte)it->second);
							outs = 1;
						}
					}
					else
					{
						auto it = c_arith.find(t);
						if (it != c_arith.end())
						{
							vector<pair<bytes, vector<unsigned>>> codes(1);
							int totalArgs = 0;
							while (d != e)
							{
								int o = compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second, _vars);
								if (o < 1)
									break;
								codes.push_back(pair<bytes, vector<unsigned>>());
								totalArgs += o;
							}
							codes.pop_back();
							if (!totalArgs)
							{
								cwarn << "Expected at least one argument to operation" << t;
								break;
							}
							for (auto jt = codes.rbegin(); jt != codes.rend(); ++jt)
								appendCode(o_code, o_locs, jt->first, jt->second);
							o_code.push_back((byte)it->second);
							outs = 1;
						}
						else
						{
							auto it = c_binary.find(t);
							if (it != c_binary.end())
							{
								vector<pair<bytes, vector<unsigned>>> codes(1);
								int totalArgs = 0;
								while (d != e)
								{
									int o = compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second, _vars);
									if (o < 1)
										break;
									codes.push_back(pair<bytes, vector<unsigned>>());
									totalArgs += o;
								}
								codes.pop_back();
//								int i = (int)codes.size();
								if (totalArgs != 2)
								{
									cwarn << "Expected two arguments to binary operator" << t << "; got" << totalArgs << ".";
									break;
								}
								for (auto jt = codes.rbegin(); jt != codes.rend(); ++jt)
									appendCode(o_code, o_locs, jt->first, jt->second);
								o_code.push_back((byte)it->second.first);
								if (it->second.second)
									o_code.push_back((byte)Instruction::NOT);
								outs = 1;
							}
							else
							{
								auto it = c_unary.find(t);
								if (it != c_unary.end())
								{
									vector<pair<bytes, vector<unsigned>>> codes(1);
									int totalArgs = 0;
									while (d != e)
									{
										int o = compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second, _vars);
										if (o == -1)
											break;
										totalArgs += o;
										codes.push_back(pair<bytes, vector<unsigned>>());
									}
									codes.pop_back();
//									int i = (int)codes.size();
									if (totalArgs != 1)
									{
										cwarn << "Expected one argument to unary operator" << t << "; got" << totalArgs << ".";
										break;
									}
									for (auto it = codes.rbegin(); it != codes.rend(); ++it)
										appendCode(o_code, o_locs, it->first, it->second);
									o_code.push_back((byte)it->second);
									outs = 1;
								}
								else
								{
									auto it = _vars.find(t);
									if (it == _vars.end())
									{
										bool ok;
										tie(it, ok) = _vars.insert(make_pair(t, _vars.size() * 32));
									}
									pushLiteral(o_code, it->second);
									outs = 1;
									// happens when it's an actual literal, escapes with -1 :-(
								}
							}
						}
					}
				}
			}

			if (!exec)
				return outs;
		}
		}
	}
	return -1;
}

bytes eth::compileLisp(std::string const& _code, bool _quiet, bytes& _init)
{
	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	bytes body;
	vector<unsigned> locs;
	map<string, unsigned> vars;
	compileLispFragment(d, e, _quiet, _init, locs, vars);
	locs.clear();
	vars.clear();
	compileLispFragment(d, e, _quiet, body, locs, vars);
	return body;
}

string eth::disassemble(bytes const& _mem)
{
	stringstream ret;
	uint numerics = 0;
	for (auto it = _mem.begin(); it != _mem.end(); ++it)
	{
		byte n = *it;
		auto iit = c_instructionInfo.find((Instruction)n);
		if (numerics || iit == c_instructionInfo.end() || (byte)iit->first != n)	// not an instruction or expecting an argument...
		{
			if (numerics)
				numerics--;
			ret << "0x" << hex << (int)n << " ";
		}
		else
		{
			auto const& ii = iit->second;
			ret << ii.name << " ";
			numerics = ii.additional;
		}
	}
	return ret.str();
}
