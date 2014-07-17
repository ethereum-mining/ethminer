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
 * @date 2014
 */

#include "CommonJS.h"
using namespace std;
using namespace eth;

bytes eth::jsToBytes(string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return fromHex(_s.substr(2));
	else if (_s.find_first_not_of("0123456789") == string::npos)
		// Decimal
		return toCompactBigEndian(bigint(_s));
	else
		// Binary
		return asBytes(_s);
}

string eth::jsPadded(string const& _s, unsigned _l, unsigned _r)
{
	bytes b = jsToBytes(_s);
	while (b.size() < _l)
		b.insert(b.begin(), 0);
	while (b.size() < _r)
		b.push_back(0);
	return asString(b).substr(b.size() - max(_l, _r));
}

string eth::jsPadded(string const& _s, unsigned _l)
{
	if (_s.substr(0, 2) == "0x" || _s.find_first_not_of("0123456789") == string::npos)
		// Numeric: pad to right
		return jsPadded(_s, _l, _l);
	else
		// Text: pad to the left
		return jsPadded(_s, 0, _l);
}

string eth::jsUnpadded(string _s)
{
	auto p = _s.find_last_not_of((char)0);
	_s.resize(p == string::npos ? 0 : (p + 1));
	return _s;
}
