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
#include "CommonEth.h"
#include "Log.h"
using namespace std;
using namespace eth;

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

u256s eth::assemble(std::string const& _code, bool _quiet)
{
	u256s ret;
	map<string, unsigned> known;
	map<unsigned, string> req;
	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '_' && *d != ':' && *d != '"'; ++d) {}
		if (d == e)
			break;

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
			ret.push_back((u256)valHash);
		}
		else
		{
			char const* s = d;
			for (; d != e && (isalnum(*d) || *d == '_' || *d == ':' || *d == '"'); ++d) {}

			string t = string(s, d - s);
			if (isdigit(t[0]))
				ret.push_back(readNumeric(t, _quiet));
			else if (t.back() == ':')
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
			}
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

static void appendCode(u256s& o_code, vector<unsigned>& o_locs, u256s _code, vector<unsigned>& _locs)
{
	o_locs.reserve(o_locs.size() + _locs.size());
	for (auto i: _locs)
	{
		_code[i] += (u256)o_code.size();
		o_locs.push_back(i + (unsigned)o_code.size());
	}
	o_code.reserve(o_code.size() + _code.size());
	for (auto i: _code)
		o_code.push_back(i);
}

static bool compileLispFragment(char const*& d, char const* e, bool _quiet, u256s& o_code, vector<unsigned>& o_locs)
{
	std::map<std::string, Instruction> const c_arith = { { "+", Instruction::ADD }, { "-", Instruction::SUB }, { "*", Instruction::MUL }, { "/", Instruction::DIV }, { "%", Instruction::MOD } };
	std::map<std::string, Instruction> const c_binary = { { "<", Instruction::LT }, { "<=", Instruction::LE }, { ">", Instruction::GT }, { ">=", Instruction::GE }, { "=", Instruction::EQ }, { "!=", Instruction::NOT } };
	std::map<std::string, Instruction> const c_unary = { { "!", Instruction::NOT } };
	std::set<char> const c_allowed = { '+', '-', '*', '/', '%', '<', '>', '=', '!' };

	bool exec = false;

	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '(' && *d != ')' && *d != '_' && *d != '"' && !c_allowed.count(*d) && *d != ';'; ++d) {}
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
		case ')':
			if (exec)
			{
				++d;
				return true;
			}
			else
				// unexpected - return false as we don't know what to do with it.
				return false;
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
					u256s codes;
					vector<unsigned> locs;
					if (compileLispFragment(d, e, _quiet, codes, locs))
					{
						appendCode(o_code, o_locs, codes, locs);
						while (compileLispFragment(d, e, _quiet, codes, locs))
							if (!_quiet)
								cwarn << "Additional items in bare store. Ignoring.";
						bareLoad = false;
					}
				}
				o_code.push_back(Instruction::PUSH);
				o_code.push_back(literalValue);
				if (exec)
					o_code.push_back(bareLoad ? Instruction::SLOAD : Instruction::SSTORE);
			}
			else
			{
				boost::algorithm::to_upper(t);
				if (t == "IF")
				{
					// Compile all the code...
					u256s codes[4];
					vector<unsigned> locs[4];
					for (int i = 0; i < 3; ++i)
						if (!compileLispFragment(d, e, _quiet, codes[i], locs[i]))
							return false;
					if (compileLispFragment(d, e, _quiet, codes[3], locs[3]))
						return false;

					// Push the positive location.
					o_code.push_back(Instruction::PUSH);
					unsigned posLocation = (unsigned)o_code.size();
					o_locs.push_back(posLocation);
					o_code.push_back(0);

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);

					// Jump to positive if true.
					o_code.push_back(Instruction::JMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[2], locs[2]);

					// Jump to end after negative.
					o_code.push_back(Instruction::PUSH);
					unsigned endLocation = (unsigned)o_code.size();
					o_locs.push_back(endLocation);
					o_code.push_back(0);
					o_code.push_back(Instruction::JMP);

					// Third fragment - positive.
					o_code[posLocation] = o_code.size();
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// At end now.
					o_code[endLocation] = o_code.size();
				}
				else if (t == "WHEN" || t == "UNLESS")
				{
					// Compile all the code...
					u256s codes[3];
					vector<unsigned> locs[3];
					for (int i = 0; i < 2; ++i)
						if (!compileLispFragment(d, e, _quiet, codes[i], locs[i]))
							return false;
					if (compileLispFragment(d, e, _quiet, codes[2], locs[2]))
						return false;

					// Push the positive location.
					o_code.push_back(Instruction::PUSH);
					unsigned endLocation = (unsigned)o_code.size();
					o_locs.push_back(endLocation);
					o_code.push_back(0);

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);

					// Jump to end...
					if (t == "WHEN")
						o_code.push_back(Instruction::NOT);
					o_code.push_back(Instruction::JMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// At end now.
					o_code[endLocation] = o_code.size();
				}
				else if (t == "FOR")
				{
					// Compile all the code...
					u256s codes[3];
					vector<unsigned> locs[3];
					for (int i = 0; i < 2; ++i)
						if (!compileLispFragment(d, e, _quiet, codes[i], locs[i]))
							return false;
					if (compileLispFragment(d, e, _quiet, codes[2], locs[2]))
						return false;

					unsigned startLocation = (unsigned)o_code.size();

					// Push the positive location.
					o_code.push_back(Instruction::PUSH);
					unsigned endInsertion = (unsigned)o_code.size();
					o_locs.push_back(endInsertion);
					o_code.push_back(0);

					// First fragment - predicate
					appendCode(o_code, o_locs, codes[0], locs[0]);

					// Jump to positive if true.
					o_code.push_back(Instruction::NOT);
					o_code.push_back(Instruction::JMPI);

					// Second fragment - negative.
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// Jump to end after negative.
					o_code.push_back(Instruction::PUSH);
					o_locs.push_back((unsigned)o_code.size());
					o_code.push_back(startLocation);
					o_code.push_back(Instruction::JMP);

					// At end now.
					o_code[endInsertion] = o_code.size();
				}
				else if (t == "SEQ")
				{
					while (d != e)
					{
						u256s codes;
						vector<unsigned> locs;
						if (compileLispFragment(d, e, _quiet, codes, locs))
							appendCode(o_code, o_locs, codes, locs);
						else
							break;
					}
				}
				else if (t == "AND")
				{
					vector<u256s> codes;
					vector<vector<unsigned>> locs;
					while (d != e)
					{
						codes.resize(codes.size() + 1);
						locs.resize(locs.size() + 1);
						if (!compileLispFragment(d, e, _quiet, codes.back(), locs.back()))
							break;
					}

					// last one is empty.
					if (codes.size() < 2)
						return false;

					codes.pop_back();
					locs.pop_back();

					vector<unsigned> ends;

					if (codes.size() > 1)
					{
						o_code.push_back(Instruction::PUSH);
						o_code.push_back(0);

						for (unsigned i = 1; i < codes.size(); ++i)
						{
							// Push the false location.
							o_code.push_back(Instruction::PUSH);
							ends.push_back((unsigned)o_code.size());
							o_locs.push_back(ends.back());
							o_code.push_back(0);

							// Check if true - predicate
							appendCode(o_code, o_locs, codes[i - 1], locs[i - 1]);

							// Jump to end...
							o_code.push_back(Instruction::NOT);
							o_code.push_back(Instruction::JMPI);
						}
						o_code.push_back(Instruction::POP);
					}

					// Check if true - predicate
					appendCode(o_code, o_locs, codes.back(), locs.back());

					// At end now.
					for (auto i: ends)
						o_code[i] = o_code.size();
				}
				else if (t == "OR")
				{
					vector<u256s> codes;
					vector<vector<unsigned>> locs;
					while (d != e)
					{
						codes.resize(codes.size() + 1);
						locs.resize(locs.size() + 1);
						if (!compileLispFragment(d, e, _quiet, codes.back(), locs.back()))
							break;
					}

					// last one is empty.
					if (codes.size() < 2)
						return false;

					codes.pop_back();
					locs.pop_back();

					vector<unsigned> ends;

					if (codes.size() > 1)
					{
						o_code.push_back(Instruction::PUSH);
						o_code.push_back(1);

						for (unsigned i = 1; i < codes.size(); ++i)
						{
							// Push the false location.
							o_code.push_back(Instruction::PUSH);
							ends.push_back((unsigned)o_code.size());
							o_locs.push_back(ends.back());
							o_code.push_back(0);

							// Check if true - predicate
							appendCode(o_code, o_locs, codes[i - 1], locs[i - 1]);

							// Jump to end...
							o_code.push_back(Instruction::JMPI);
						}
						o_code.push_back(Instruction::POP);
					}

					// Check if true - predicate
					appendCode(o_code, o_locs, codes.back(), locs.back());

					// At end now.
					for (auto i: ends)
						o_code[i] = o_code.size();
				}
				else
				{
					auto it = c_instructions.find(t);
					if (it != c_instructions.end())
					{
						if (exec)
						{
							vector<pair<u256s, vector<unsigned>>> codes(1);
							while (d != e && compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second))
								codes.push_back(pair<u256s, vector<unsigned>>());
							for (auto it = codes.rbegin(); it != codes.rend(); ++it)
								appendCode(o_code, o_locs, it->first, it->second);
							o_code.push_back((u256)it->second);
						}
						else
						{
							o_code.push_back(Instruction::PUSH);
							o_code.push_back(it->second);
						}
					}
					else
					{
						auto it = c_arith.find(t);
						if (it != c_arith.end())
						{
							int i = 0;
							while (d != e)
							{
								u256s codes;
								vector<unsigned> locs;
								if (compileLispFragment(d, e, _quiet, codes, locs))
								{
									appendCode(o_code, o_locs, codes, locs);
									if (i)
										o_code.push_back((u256)it->second);
									++i;
								}
								else
									break;
							}
						}
						else
						{
							auto it = c_binary.find(t);
							if (it != c_binary.end())
							{
								vector<pair<u256s, vector<unsigned>>> codes(1);
								while (d != e && compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second))
									codes.push_back(pair<u256s, vector<unsigned>>());
								codes.pop_back();
								int i = (int)codes.size();
								if (i > 2)
									cwarn << "Greater than two arguments given to binary operator" << t << "; using first two only.";
								for (auto it = codes.rbegin(); it != codes.rend(); ++it)
									if (--i < 2)
										appendCode(o_code, o_locs, it->first, it->second);
								if (it->second == Instruction::NOT)
									o_code.push_back(Instruction::EQ);
								o_code.push_back((u256)it->second);
							}
							else
							{
								auto it = c_unary.find(t);
								if (it != c_unary.end())
								{
									vector<pair<u256s, vector<unsigned>>> codes(1);
									while (d != e && compileLispFragment(d, e, _quiet, codes.back().first, codes.back().second))
										codes.push_back(pair<u256s, vector<unsigned>>());
									codes.pop_back();
									int i = (int)codes.size();
									if (i > 1)
										cwarn << "Greater than one argument given to unary operator" << t << "; using first only.";
									for (auto it = codes.rbegin(); it != codes.rend(); ++it)
										if (--i < 1)
											appendCode(o_code, o_locs, it->first, it->second);
									o_code.push_back(it->second);
								}
								else if (!_quiet)
									cwarn << "Unknown assembler token" << t;
							}
						}
					}
				}
			}

			if (!exec)
				return true;
		}
		}
	}
	return false;
}

u256s eth::compileLisp(std::string const& _code, bool _quiet)
{
	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	u256s ret;
	vector<unsigned> locs;
	compileLispFragment(d, e, _quiet, ret, locs);
	return ret;
}

string eth::disassemble(u256s const& _mem)
{
	stringstream ret;
	uint numerics = 0;
	for (auto it = _mem.begin(); it != _mem.end(); ++it)
	{
		u256 n = *it;
		auto iit = c_instructionInfo.find((Instruction)(uint)n);
		if (numerics || iit == c_instructionInfo.end() || (u256)(uint)iit->first != n)	// not an instruction or expecting an argument...
		{
			if (numerics)
				numerics--;
			ret << "0x" << hex << n << " ";
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
