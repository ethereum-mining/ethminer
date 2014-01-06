/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Transaction.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
#include "sha256.h"
#include "RLP.h"

namespace eth
{

using PrivateKey = u256;
using Address = u160;

struct Signature
{
	byte v;
	u256 r;
	u256 s;
};

// [ nonce, receiving_address, value, fee, [ data item 0, data item 1 ... data item n ], v, r, s ]
struct Transaction
{
	Transaction() {}
	Transaction(bytes const& _rlp);

	u256 nonce;
	Address receiveAddress;
	u256 value;
	u256 fee;
	u256s data;
	Signature vrs;

	Address sender() const;
	void sign(PrivateKey _priv);

	static u256 kFromMessage(u256 _msg, u256 _priv);

	void fillStream(RLPStream& _s, bool _sig = true) const;
	bytes rlp(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return s.out(); }
	std::string rlpString(bool _sig = true) const { return asString(rlp()); }
	u256 sha256(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha256(s.out()); }
	bytes sha256Bytes(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha256Bytes(s.out()); }
};

}


