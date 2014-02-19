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

u256s eth::assemble(std::string const& _code)
{
	u256s ret;
	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '_'; ++d) {}
		if (d == e)
			break;

		char const* s = d;
		for (; d != e && (isalnum(*d) || *d == '_'); ++d) {}

		string t = string(s, d - s);
		if (isdigit(t[0]))
			ret.push_back(u256(t));
		else
		{
			boost::algorithm::to_upper(t);
			auto it = c_instructions.find(t);
			if (it != c_instructions.end())
				ret.push_back((u256)it->second);
			else
				cwarn << "Unknown assembler token" << t;
		}
	}
	return ret;
}

u256s eth::compileLisp(std::string const& _code)
{
	u256s ret;
	vector<pair<Instruction, int>> inStack;


	char const* d = _code.data();
	char const* e = _code.data() + _code.size();
	while (d != e)
	{
		// skip to next token
		for (; d != e && !isalnum(*d) && *d != '(' && *d != ')' && *d != '_'; ++d) {}
		if (d == e)
			break;

		switch (*d)
		{
		case '(':
			inStack.push_back(make_pair(Instruction::STOP, 0));
			++d;
			break;
		case ')':
			++d;
			if (inStack.size())
				ret.push_back(inStack.back().first);
			inStack.pop_back();
			if (inStack.size())
				inStack.back().second++;
			break;
		default:
		{
			char const* s = d;
			for (; d != e && isalnum(*d); ++d) {}

			string t = string(s, d - s);
			if (isdigit(t[0]))
			{
				ret.push_back(u256(t));
				if (inStack.size() && !inStack.back().second)
					cwarn << "Cannot execute numeric" << t;
				else
				{
					ret.push_back(Instruction::PUSH);
					ret.push_back(u256(t));
				}
			}
			else
			{
				boost::algorithm::to_upper(t);
				auto it = c_instructions.find(t);
				if (it != c_instructions.end())
					if (inStack.size())
					{
						if (!inStack.back().second)
							inStack.back().first = it->second;
						else
							ret.push_back((u256)it->second);
						inStack.back().second++;
					}
					else
						cwarn << "Instruction outside parens" << t;
				else
					cwarn << "Unknown assembler token" << t;
			}
		}
		}
	}
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
			ret << " 0x" << hex << n;
		}
		else
		{
			auto const& ii = iit->second;
			ret << " " << ii.name;
			numerics = ii.additional;
		}
	}
	return ret.str();
}
