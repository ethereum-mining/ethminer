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
using namespace std;
using namespace eth;

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
		for (; d != e && !isalnum(*d) && *d != '_' && *d != ':'; ++d) {}
		if (d == e)
			break;

		char const* s = d;
		for (; d != e && (isalnum(*d) || *d == '_' || *d == ':'); ++d) {}

		string t = string(s, d - s);
		if (isdigit(t[0]))
			try
			{
				ret.push_back(u256(t));
			}
			catch (...)
			{
				cwarn << "Invalid numeric" << t;
			}
		else if (t.back() == ':')
			known[t.substr(0, t.size() - 1)] = ret.size();
		else
		{
			auto it = c_instructions.find(boost::algorithm::to_upper_copy(t));
			if (it != c_instructions.end())
				ret.push_back((u256)it->second);
			else
			{
				req[ret.size()] = t;
				ret.push_back(0);
			}
		}
	}
	for (auto i: req)
		if (known.count(i.second))
			ret[i.first] = known[i.second];
		else
			cwarn << "Unknown assembler token" << i.second << "at address" << i.first;

	return ret;
}

static void appendCode(u256s& o_code, vector<unsigned>& o_locs, u256s _code, vector<unsigned>& _locs)
{
	o_locs.reserve(o_locs.size() + _locs.size());
	for (auto i: _locs)
	{
		_code[i] += (u256)o_code.size();
		o_locs.push_back(i + o_code.size());
	}
	o_code.reserve(o_code.size() + _code.size());
	for (auto i: _code)
		o_code.push_back(i);
}

static bool compileLispFragment(char const*& d, char const* e, bool _quiet, u256s& o_code, vector<unsigned>& o_locs)
{
	bool exec = false;

	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '(' && *d != ')' && *d != '_'; ++d) {}
		if (d == e)
			break;

		switch (*d)
		{
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
			char const* s = d;
			for (; d != e && (isalnum(*d) || *d == '_'); ++d) {}

			string t(s, d - s);
			if (isdigit(t[0]))
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
							cwarn << "Additional items in bare store. Ignoring.";
						bareLoad = false;
					}
				}
				o_code.push_back(Instruction::PUSH);
				try
				{
					o_code.push_back(u256(t));
				}
				catch (...)
				{
					cwarn << "Invalid numeric" << t;
				}
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
					unsigned posLocation = o_code.size();
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
					unsigned endLocation = o_code.size();
					o_locs.push_back(endLocation);
					o_code.push_back(0);
					o_code.push_back(Instruction::JMP);

					// Third fragment - positive.
					o_code[posLocation] = o_code.size();
					appendCode(o_code, o_locs, codes[1], locs[1]);

					// At end now.
					o_code[endLocation] = o_code.size();
				}
				if (t == "FOR")
				{
					// Compile all the code...
					u256s codes[3];
					vector<unsigned> locs[3];
					for (int i = 0; i < 2; ++i)
						if (!compileLispFragment(d, e, _quiet, codes[i], locs[i]))
							return false;
					if (compileLispFragment(d, e, _quiet, codes[2], locs[2]))
						return false;

					unsigned startLocation = o_code.size();

					// Push the positive location.
					o_code.push_back(Instruction::PUSH);
					unsigned endInsertion = o_code.size();
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
					o_locs.push_back(o_code.size());
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
							try
							{
								o_code.push_back((u256)it->second);
							}
							catch (...)
							{
								cwarn << "Invalid numeric" << t;
							}
						}
					}
					else if (!_quiet)
						cwarn << "Unknown assembler token" << t;
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
