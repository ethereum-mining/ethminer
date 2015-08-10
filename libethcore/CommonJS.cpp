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
#include "ICAP.h"

namespace dev
{

Address jsToAddress(std::string const& _s)
{
	try
	{
		eth::ICAP i = eth::ICAP::decoded(_s);
		return i.direct();
	}
	catch (eth::InvalidICAP&) {}
	try
	{
		auto b = fromHex(_s.substr(0, 2) == "0x" ? _s.substr(2) : _s, WhenError::Throw);
		if (b.size() == 20)
			return Address(b);
	}
	catch (BadHexCharacter&) {}
	BOOST_THROW_EXCEPTION(InvalidAddress());
}

std::string prettyU256(u256 _n, bool _abridged)
{
	unsigned inc = 0;
	std::string raw;
	std::ostringstream s;
	if (!(_n >> 64))
		s << " " << (uint64_t)_n << " (0x" << std::hex << (uint64_t)_n << ")";
	else if (!~(_n >> 64))
		s << " " << (int64_t)_n << " (0x" << std::hex << (int64_t)_n << ")";
	else if ((_n >> 160) == 0)
	{
		Address a = right160(_n);

		std::string n;
		if (_abridged)
			n =  a.abridged();
		else
			n = toHex(a.ref());

		if (n.empty())
			s << "0";
		else
			s << _n << "(0x" << n << ")";
	}
	else if ((raw = fromRaw((h256)_n, &inc)).size())
		return "\"" + raw + "\"" + (inc ? " + " + std::to_string(inc) : "");
	else
		s << "" << (h256)_n;
	return s.str();
}

namespace eth
{

BlockNumber jsToBlockNumber(std::string const& _js)
{
	if (_js == "latest")
		return LatestBlock;
	else if (_js == "earliest")
		return 0;
	else if (_js == "pending")
		return PendingBlock;
	else
		return (unsigned)jsToInt(_js);
}

}

}

