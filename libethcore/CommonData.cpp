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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CommonData.h"

#include <random>
#include "Exceptions.h"
using namespace std;
using namespace eth;

std::string eth::escaped(std::string const& _s, bool _all)
{
	std::string ret;
	ret.reserve(_s.size());
	ret.push_back('"');
	for (auto i: _s)
		if (i == '"' && !_all)
			ret += "\\\"";
		else if (i == '\\' && !_all)
			ret += "\\\\";
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

std::string eth::randomWord()
{
	static std::mt19937_64 s_eng(0);
	std::string ret(std::uniform_int_distribution<int>(4, 10)(s_eng), ' ');
	char const n[] = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890";
	std::uniform_int_distribution<int> d(0, sizeof(n) - 2);
	for (char& c: ret)
		c = n[d(s_eng)];
	return ret;
}

int eth::fromHex(char _i)
{
	if (_i >= '0' && _i <= '9')
		return _i - '0';
	if (_i >= 'a' && _i <= 'f')
		return _i - 'a' + 10;
	if (_i >= 'A' && _i <= 'F')
		return _i - 'A' + 10;
	throw BadHexCharacter();
}

bytes eth::fromHex(std::string const& _s)
{
	if (_s.size() % 2 || _s.size() < 2)
		return bytes();
	uint s = (_s[0] == '0' && _s[1] == 'x') ? 2 : 0;
	std::vector<uint8_t> ret;
	ret.reserve((_s.size() - s) / 2);
	for (uint i = s; i < _s.size(); i += 2)
		try
		{
			ret.push_back((byte)(fromHex(_s[i]) * 16 + fromHex(_s[i + 1])));
		}
		catch (...){}
	return ret;
}

bytes eth::asNibbles(std::string const& _s)
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
