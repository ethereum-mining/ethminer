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

#include <libethcore/RLP.h>
#include "CommonEth.h"

namespace eth
{

struct Signature
{
	byte v;
	u256 r;
	u256 s;
};

// [ nonce, value, receiveAddress, gasPrice, gasDeposit, data, v, r, s ]
// or
// [ nonce, endowment, 0, gasPrice, gasDeposit (for init), body, init, v, r, s ]
struct Transaction
{
	Transaction() {}
	Transaction(bytesConstRef _rlp);
	Transaction(bytes const& _rlp): Transaction(&_rlp) {}

	bool operator==(Transaction const& _c) const { return receiveAddress == _c.receiveAddress && value == _c.value && data == _c.data; }
	bool operator!=(Transaction const& _c) const { return !operator==(_c); }

	u256 nonce;			///< The transaction-count of the sender.
	u256 value;			///< The amount of ETH to be transferred by this transaction. Called 'endowment' for contract-creation transactions.
	Address receiveAddress;	///< The receiving address of the transaction.
	u256 gasPrice;		///< The base fee and thus the implied exchange rate of ETH to GAS.
	u256 gas;			///< The total gas to convert, paid for from sender's account. Any unused gas gets refunded once the contract is ended.

	bytes data;			///< The data associated with the transaction, or the main body if it's a creation transaction.
	bytes init;			///< The initialisation associated with the transaction.

	Signature vrs;		///< The signature of the transaction. Encodes the sender.

	Address safeSender() const noexcept;	///< Like sender() but will never throw.
	Address sender() const;	///< Determine the sender of the transaction from the signature (and hash).
	void sign(Secret _priv);	///< Sign the transaction.

	bool isCreation() const { return !receiveAddress; }

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

	_out << "/" << _t.nonce << "$" << _t.value << "+" << _t.gas << "@" << _t.gasPrice;
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


