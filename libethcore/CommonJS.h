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
/** @file CommonJS.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <libdevcore/CommonJS.h>
#include <libdevcrypto/Common.h>
#include "Common.h"

// devcrypto

namespace dev
{

/// Leniently convert string to Public (h512). Accepts integers, "0x" prefixing, non-exact length.
inline Public jsToPublic(std::string const& _s) { return jsToFixed<sizeof(dev::Public)>(_s); }

/// Leniently convert string to Secret (h256). Accepts integers, "0x" prefixing, non-exact length.
inline Secret jsToSecret(std::string const& _s) { h256 d = jsToFixed<sizeof(dev::Secret)>(_s); Secret ret(d); d.ref().cleanse(); return ret; }

/// Leniently convert string to Address (h160). Accepts integers, "0x" prefixing, non-exact length.
inline Address jsToAddress(std::string const& _s) { return eth::toAddress(_s); }

/// Convert u256 into user-readable string. Returns int/hex value of 64 bits int, hex of 160 bits FixedHash. As a fallback try to handle input as h256.
std::string prettyU256(u256 _n, bool _abridged = true);

}


// ethcore
namespace dev
{
namespace eth
{

/// Convert to a block number, a bit like jsToInt, except that it correctly recognises "pending" and "latest".
BlockNumber jsToBlockNumber(std::string const& _js);

}
}
