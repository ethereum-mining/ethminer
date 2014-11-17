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

namespace dev
{

bytes jsToBytes(std::string const& _s)
{
	if (_s.substr(0, 2) == "0x")
		// Hex
		return fromHex(_s.substr(2));
	else if (_s.find_first_not_of("0123456789") == std::string::npos)
		// Decimal
		return toCompactBigEndian(bigint(_s));
	else
		return bytes();
}

bytes padded(bytes _b, unsigned _l)
{
	while (_b.size() < _l)
		_b.insert(_b.begin(), 0);
	while (_b.size() < _l)
		_b.push_back(0);
	return asBytes(asString(_b).substr(_b.size() - std::max(_l, _l)));
}

bytes unpadded(bytes _b)
{
	auto p = asString(_b).find_last_not_of((char)0);
	_b.resize(p == std::string::npos ? 0 : (p + 1));
	return _b;
}

}
