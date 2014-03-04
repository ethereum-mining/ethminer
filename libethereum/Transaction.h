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
/** @file Transaction.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "CommonEth.h"
#include "RLP.h"

namespace eth
{

struct Signature
{
	byte v;
	u256 r;
	u256 s;
};

// [ nonce, receiving_address, value, [ data item 0, data item 1 ... data item n ], v, r, s ]
struct Transaction
{
	Transaction() {}
	Transaction(bytesConstRef _rlp);
	Transaction(bytes const& _rlp): Transaction(&_rlp) {}

	bool operator==(Transaction const& _c) const { return receiveAddress == _c.receiveAddress && value == _c.value && data == _c.data; }
	bool operator!=(Transaction const& _c) const { return !operator==(_c); }

	u256 nonce;
	Address receiveAddress;
	u256 value;
	u256s data;
	Signature vrs;

	Address safeSender() const noexcept;
	Address sender() const;
	void sign(Secret _priv);

	static h256 kFromMessage(h256 _msg, h256 _priv);

	void fillStream(RLPStream& _s, bool _sig = true) const;
	bytes rlp(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return s.out(); }
	std::string rlpString(bool _sig = true) const { return asString(rlp(_sig)); }
	h256 sha3(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha3(s.out()); }
	bytes sha3Bytes(bool _sig = true) const { RLPStream s; fillStream(s, _sig); return eth::sha3Bytes(s.out()); }
};

using Transactions = std::vector<Transaction>;

inline std::ostream& operator<<(std::ostream& _out, Transaction const& _t)
{
	_out << "{";
	if (_t.receiveAddress)
		_out << _t.receiveAddress.abridged();
	else
		_out << "[CREATE]";

	_out << "/" << _t.nonce << "*" << _t.value;
	Address s;
	try
	{
		_out << "<-" << _t.sender().abridged();
	}
	catch (...) {}
	_out << "}";
	return _out;
}

}


