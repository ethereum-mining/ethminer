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
/** @file CommonData.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CommonData.h"

#include <random>
#include "Exceptions.h"
#include "Log.h"

using namespace std;
using namespace dev;

std::string dev::escaped(std::string const& _s, bool _all)
{
	static const map<char, char> prettyEscapes{{'\r', 'r'}, {'\n', 'n'}, {'\t', 't'}, {'\v', 'v'}};
	std::string ret;
	ret.reserve(_s.size() + 2);
	ret.push_back('"');
	for (auto i: _s)
		if (i == '"' && !_all)
			ret += "\\\"";
		else if (i == '\\' && !_all)
			ret += "\\\\";
		else if (prettyEscapes.count(i))
		{
			ret += '\\';
			ret += prettyEscapes.find(i)->second;
		}
		else if (i < ' ' || _all)
		{
			ret += "\\x";
			ret.push_back("0123456789abcdef"[(uint8_t)i / 16]);
			ret.push_back("0123456789abcdef"[(uint8_t)i % 16]);
		}
		else
			ret.push_back(i);
	ret.push_back('"');
	return ret;
}

std::string dev::randomWord()
{
	static std::mt19937_64 s_eng(0);
	std::string ret(std::uniform_int_distribution<int>(1, 5)(s_eng), ' ');
	char const n[] = "qwertyuiop";//asdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890";
	std::uniform_int_distribution<int> d(0, sizeof(n) - 2);
	for (char& c: ret)
		c = n[d(s_eng)];
	return ret;
}

int dev::fromHex(char _i)
{
	if (_i >= '0' && _i <= '9')
		return _i - '0';
	if (_i >= 'a' && _i <= 'f')
		return _i - 'a' + 10;
	if (_i >= 'A' && _i <= 'F')
		return _i - 'A' + 10;
	BOOST_THROW_EXCEPTION(BadHexCharacter() << errinfo_invalidSymbol(_i));
}

bytes dev::fromHex(std::string const& _s, ThrowType _throw)
{
	unsigned s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
	std::vector<uint8_t> ret;
	ret.reserve((_s.size() - s + 1) / 2);

	if (_s.size() % 2)
		try
		{
			ret.push_back(fromHex(_s[s++]));
		}
		catch (...)
		{ 
			ret.push_back(0);
			// msvc does not support it
#ifndef BOOST_NO_EXCEPTIONS
			cwarn << boost::current_exception_diagnostic_information(); 
#endif
			if (_throw == ThrowType::Throw)
				throw;
		}
	for (unsigned i = s; i < _s.size(); i += 2)
		try
		{
			ret.push_back((byte)(fromHex(_s[i]) * 16 + fromHex(_s[i + 1])));
		}
		catch (...){
			ret.push_back(0);
#ifndef BOOST_NO_EXCEPTIONS
			cwarn << boost::current_exception_diagnostic_information();
#endif
			if (_throw == ThrowType::Throw)
				throw;
		}
	return ret;
}

bytes dev::asNibbles(std::string const& _s)
{
	std::vector<uint8_t> ret;
	ret.reserve(_s.size() * 2);
	for (auto i: _s)
	{
		ret.push_back(i / 16);
		ret.push_back(i % 16);
	}
	return ret;
}

std::string dev::toString(string32 const& _s)
{
	std::string ret;
	for (unsigned i = 0; i < 32 && _s[i]; ++i)
		ret.push_back(_s[i]);
	return ret;
}
