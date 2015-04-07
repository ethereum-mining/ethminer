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
/** @file CommonJS.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#include "CommonJS.h"

using namespace std;

namespace dev
{

bytes jsToBytes(string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return fromHex(_s.substr(2));
	else if (_s.find_first_not_of("0123456789") == string::npos)
		// Decimal
		return toCompactBigEndian(bigint(_s));
	else
		return bytes();
}

bytes padded(bytes _b, unsigned _l)
{
	while (_b.size() < _l)
		_b.insert(_b.begin(), 0);
	return asBytes(asString(_b).substr(_b.size() - max(_l, _l)));
}

bytes paddedRight(bytes _b, unsigned _l)
{
	_b.resize(_l);
	return _b;
}

bytes unpadded(bytes _b)
{
	auto p = asString(_b).find_last_not_of((char)0);
	_b.resize(p == string::npos ? 0 : (p + 1));
	return _b;
}

bytes unpadLeft(bytes _b)
{
	unsigned int i = 0;
	if (_b.size() == 0)
		return _b;

	while (i < _b.size() && _b[i] == byte(0))
		i++;

	if (i != 0)
		_b.erase(_b.begin(), _b.begin() + i);
	return _b;
}

string fromRaw(h256 _n, unsigned* _inc)
{
	if (_n)
	{
		string s((char const*)_n.data(), 32);
		auto l = s.find_first_of('\0');
		if (!l)
			return "";
		if (l != string::npos)
		{
			auto p = s.find_first_not_of('\0', l);
			if (!(p == string::npos || (_inc && p == 31)))
				return "";
			if (_inc)
				*_inc = (byte)s[31];
			s.resize(l);
		}
		for (auto i: s)
			if (i < 32)
				return "";
		return s;
	}
	return "";
}

}

