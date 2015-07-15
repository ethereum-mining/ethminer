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
/** @file Utility.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <libdevcore/Common.h>
#include <libdevcore/FixedHash.h>

namespace dev
{
namespace eth
{

/**
 * Takes a user-authorable string with several whitespace delimited arguments and builds a byte array
 * from it. Arguments can be hex data/numerals, decimal numbers or ASCII strings. Literals are padded
 * to 32 bytes if prefixed by a '@' (or not prefixed at all), and tightly packed if prefixed by a '$'.
 * Currency multipliers can be provided.
 *
 * Example:
 * @code
 * parseData("$42 0x42 $\"Hello\"");	// == bytes(1, 0x2a) + bytes(31, 0) + bytes(1, 0x42) + asBytes("Hello");
 * @endcode
 */
bytes parseData(std::string const& _args);

void upgradeDatabase(std::string const& _basePath, h256 const& _genesisHash);

}
}
