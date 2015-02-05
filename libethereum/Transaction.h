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

#include <libdevcore/RLP.h>
#include <libdevcrypto/SHA3.h>
#include <libethcore/CommonEth.h>

namespace dev
{
namespace eth
{

/// Named-boolean type to encode whether a signature be included in the serialisation process.
enum IncludeSignature
{
	WithoutSignature = 0,	///< Do not include a signature.
	WithSignature = 1,		///< Do include a signature.
};

enum class CheckSignature
{
	None,
	Range,
	Sender
};

/// Encodes a transaction, ready to be exported to or freshly imported from RLP.
class Transaction
{
public:
	/// Constructs a null transaction.
	Transaction() {}

	/// Constructs a signed message-call transaction.
	Transaction(u256 _value, u256 _gasPrice, u256 _gas, Address const& _dest, bytes const& _data, u256 _nonce, Secret const& _secret): m_type(MessageCall), m_nonce(_nonce), m_value(_value), m_receiveAddress(_dest), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) { sign(_secret); }

	/// Constructs a signed contract-creation transaction.
	Transaction(u256 _value, u256 _gasPrice, u256 _gas, bytes const& _data, u256 _nonce, Secret const& _secret): m_type(ContractCreation), m_nonce(_nonce), m_value(_value), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) { sign(_secret); }

	/// Constructs an unsigned message-call transaction.
	Transaction(u256 _value, u256 _gasPrice, u256 _gas, Address const& _dest, bytes const& _data): m_type(MessageCall), m_value(_value), m_receiveAddress(_dest), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) {}

	/// Constructs an unsigned contract-creation transaction.
	Transaction(u256 _value, u256 _gasPrice, u256 _gas, bytes const& _data): m_type(ContractCreation), m_value(_value), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) {}

	/// Constructs a transaction from the given RLP.
	explicit Transaction(bytesConstRef _rlp, CheckSignature _checkSig);

	/// Constructs a transaction from the given RLP.
	explicit Transaction(bytes const& _rlp, CheckSignature _checkSig): Transaction(&_rlp, _checkSig) {}


	/// Checks equality of transactions.
	bool operator==(Transaction const& _c) const { return m_type == _c.m_type && (m_type == ContractCreation || m_receiveAddress == _c.m_receiveAddress) && m_value == _c.m_value && m_data == _c.m_data; }
	/// Checks inequality of transactions.
	bool operator!=(Transaction const& _c) const { return !operator==(_c); }

	/// @returns sender of the transaction from the signature (and hash).
	Address sender() const;
	/// Like sender() but will never throw. @returns a null Address if the signature is invalid.
	Address safeSender() const noexcept;

	/// @returns true if transaction is non-null.
	operator bool() const { return m_type != NullTransaction; }

	/// @returns true if transaction is contract-creation.
	bool isCreation() const { return m_type == ContractCreation; }

	/// @returns true if transaction is message-call.
	bool isMessageCall() const { return m_type == MessageCall; }

	/// Serialises this transaction to an RLPStream.
	void streamRLP(RLPStream& _s, IncludeSignature _sig = WithSignature) const;

	/// @returns the RLP serialisation of this transaction.
	bytes rlp(IncludeSignature _sig = WithSignature) const { RLPStream s; streamRLP(s, _sig); return s.out(); }

	/// @returns the SHA3 hash of the RLP serialisation of this transaction.
	h256 sha3(IncludeSignature _sig = WithSignature) const { RLPStream s; streamRLP(s, _sig); return dev::sha3(s.out()); }

	/// @returns the amount of ETH to be transferred by this (message-call) transaction, in Wei. Synonym for endowment().
	u256 value() const { return m_value; }
	/// @returns the amount of ETH to be endowed by this (contract-creation) transaction, in Wei. Synonym for value().
	u256 endowment() const { return m_value; }

	/// @returns the base fee and thus the implied exchange rate of ETH to GAS.
	u256 gasPrice() const { return m_gasPrice; }

	/// @returns the total gas to convert, paid for from sender's account. Any unused gas gets refunded once the contract is ended.
	u256 gas() const { return m_gas; }

	/// @returns the receiving address of the message-call transaction (undefined for contract-creation transactions).
	Address receiveAddress() const { return m_receiveAddress; }

	/// @returns the data associated with this (message-call) transaction. Synonym for initCode().
	bytes const& data() const { return m_data; }
	/// @returns the initialisation code associated with this (contract-creation) transaction. Synonym for data().
	bytes const& initCode() const { return m_data; }

	/// @returns the transaction-count of the sender.
	u256 nonce() const { return m_nonce; }

	/// @returns the signature of the transaction. Encodes the sender.
	SignatureStruct const& signature() const { return m_vrs; }

private:
	/// Type of transaction.
	enum Type
	{
		NullTransaction,			///< Null transaction.
		ContractCreation,			///< Transaction to create contracts - receiveAddress() is ignored.
		MessageCall					///< Transaction to invoke a message call - receiveAddress() is used.
	};

	void sign(Secret _priv);		///< Sign the transaction.

	Type m_type = NullTransaction;	///< Is this a contract-creation transaction or a message-call transaction?
	u256 m_nonce;					///< The transaction-count of the sender.
	u256 m_value;					///< The amount of ETH to be transferred by this transaction. Called 'endowment' for contract-creation transactions.
	Address m_receiveAddress;		///< The receiving address of the transaction.
	u256 m_gasPrice;				///< The base fee and thus the implied exchange rate of ETH to GAS.
	u256 m_gas;						///< The total gas to convert, paid for from sender's account. Any unused gas gets refunded once the contract is ended.
	bytes m_data;					///< The data associated with the transaction, or the initialiser if it's a creation transaction.
	SignatureStruct m_vrs;			///< The signature of the transaction. Encodes the sender.

	mutable Address m_sender;		///< Cached sender, determined from signature.
};

/// Nice name for vector of Transaction.
using Transactions = std::vector<Transaction>;

/// Simple human-readable stream-shift operator.
inline std::ostream& operator<<(std::ostream& _out, Transaction const& _t)
{
	_out << "{";
	if (_t.receiveAddress())
		_out << _t.receiveAddress().abridged();
	else
		_out << "[CREATE]";

	_out << "/" << _t.nonce() << "$" << _t.value() << "+" << _t.gas() << "@" << _t.gasPrice();
	try
	{
		_out << "<-" << _t.sender().abridged();
	}
	catch (...) {}
	_out << " #" << _t.data().size() << "}";
	return _out;
}

}
}
