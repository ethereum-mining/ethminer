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

std::string prettyU256(u256 _n)
{
	unsigned inc = 0;
	std::string raw;
	std::ostringstream s;
	if (!(_n >> 64))
		s << " " << (uint64_t)_n << " (0x" << (uint64_t)_n << ")";
	else if (!~(_n >> 64))
		s << " " << (int64_t)_n << " (0x" << (int64_t)_n << ")";
	else if ((_n >> 160) == 0)
	{
		Address a = right160(_n);

		std::string n = a.abridged();
		if (n.empty())
			s << "0x" << a;
		else
			s << n << "(0x" << a.abridged() << ")";
	}
	else if ((raw = fromRaw((h256)_n, &inc)).size())
		return "\"" + raw + "\"" + (inc ? " + " + std::to_string(inc) : "");
	else
		s << "" << (h256)_n;
	return s.str();
}

std::string fromRaw(h256 _n, unsigned* _inc)
{
	if (_n)
	{
		std::string s((char const*)_n.data(), 32);
		auto l = s.find_first_of('\0');
		if (!l)
			return NULL;
		if (l != std::string::npos)
		{
			auto p = s.find_first_not_of('\0', l);
			if (!(p == std::string::npos || (_inc && p == 31)))
				return NULL;
			if (_inc)
				*_inc = (byte)s[31];
			s.resize(l);
		}
		for (auto i: s)
			if (i < 32)
				return NULL;
		return s;
	}
	return NULL;
}

Address fromString(std::string _sn)
{
	if (_sn.size() > 32)
		_sn.resize(32);
	h256 n;
	memcpy(n.data(), _sn.data(), _sn.size());
	memset(n.data() + _sn.size(), 0, 32 - _sn.size());
	if (_sn.size() == 40)
		return Address(fromHex(_sn));
	else
		return Address();
}


}
