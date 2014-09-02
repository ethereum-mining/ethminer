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
/** @file Manifest.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <iostream>
#include <sstream>
#include <libethential/RLP.h>
#include <libethcore/CommonEth.h>

namespace eth
{

struct Manifest;
using Manifests = std::vector<Manifest>;

/**
 * @brief A record of the state-interaction of a transaction/call/create.
 */
struct Manifest
{
	Manifest() {}
	Manifest(bytesConstRef _r);
	void streamOut(RLPStream& _s) const;

	h256 bloom() const { h256 ret = from.bloom() | to.bloom(); for (auto const& i: internal) ret |= i.bloom(); for (auto const& i: altered) ret |= h256(i).bloom(); return ret; }

	std::string toString(unsigned _indent = 0) const
	{
		std::ostringstream oss;
		oss << std::string(_indent * 3, ' ') << from << " -> " << to << " [" << value << "]: {";
		if (internal.size())
		{
			oss << std::endl;
			for (auto const& m: internal)
				oss << m.toString(_indent + 1) << std::endl;
			oss << std::string(_indent * 3, ' ');
		}
		oss << "} I:" << toHex(input) << "; O:" << toHex(output);
		return oss.str();
	}

	Address from;
	Address to;
	u256 value;
	u256s altered;
	bytes input;
	bytes output;
	Manifests internal;
};

}
