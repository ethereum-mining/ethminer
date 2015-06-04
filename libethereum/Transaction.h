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
#include <libdevcore/SHA3.h>
#include <libethcore/Common.h>
#include <libevmcore/Params.h>
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

enum class CheckTransaction
{
	None,
	Cheap,
	Everything
};

enum class TransactionException
{
	None = 0,
	Unknown,
	InvalidSignature,
	InvalidNonce,
	NotEnoughCash,
	OutOfGasBase,			///< Too little gas to pay for the base transaction cost.
	BlockGasLimitReached,
	BadInstruction,
	BadJumpDestination,
	OutOfGas,				///< Ran out of gas executing code of the transaction.
	OutOfStack,				///< Ran out of stack executing code of the transaction.
	StackUnderflow
};

enum class CodeDeposit
{
	None = 0,
	Failed,
	Success
};

struct VMException;

TransactionException toTransactionException(VMException const& _e);

/// Description of the result of executing a transaction.
struct ExecutionResult
{
	ExecutionResult() = default;
	ExecutionResult(u256 const& _gasUsed, TransactionException _excepted, Address const& _newAddress, bytesConstRef _output, CodeDeposit _codeDeposit, u256 const& _gasRefund, unsigned _depositSize, u256 const& _gasForDeposit):
		gasUsed(_gasUsed),
		excepted(_excepted),
		newAddress(_newAddress),
		output(_output.toBytes()),
		codeDeposit(_codeDeposit),
		gasRefunded(_gasRefund),
		depositSize(_depositSize),
		gasForDeposit(_gasForDeposit)
	{}
	u256 gasUsed = 0;
	TransactionException excepted = TransactionException::Unknown;
	Address newAddress;
	bytes output;
	CodeDeposit codeDeposit = CodeDeposit::None;
	u256 gasRefunded = 0;
	unsigned depositSize = 0;
	u256 gasForDeposit;
};

std::ostream& operator<<(std::ostream& _out, ExecutionResult const& _er);

/// Encodes a transaction, ready to be exported to or freshly imported from RLP.
class Transaction
{
public:
	/// Constructs a null transaction.
	Transaction() {}

	/// Constructs a signed message-call transaction.
	Transaction(u256 const& _value, u256 const& _gasPrice, u256 const& _gas, Address const& _dest, bytes const& _data, u256 const& _nonce, Secret const& _secret): m_type(MessageCall), m_nonce(_nonce), m_value(_value), m_receiveAddress(_dest), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) { sign(_secret); }

	/// Constructs a signed contract-creation transaction.
	Transaction(u256 const& _value, u256 const& _gasPrice, u256 const& _gas, bytes const& _data, u256 const& _nonce, Secret const& _secret): m_type(ContractCreation), m_nonce(_nonce), m_value(_value), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) { sign(_secret); }

	/// Constructs an unsigned message-call transaction.
	Transaction(u256 const& _value, u256 const& _gasPrice, u256 const& _gas, Address const& _dest, bytes const& _data, u256 const& _nonce = 0): m_type(MessageCall), m_nonce(_nonce), m_value(_value), m_receiveAddress(_dest), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) {}

	/// Constructs an unsigned contract-creation transaction.
	Transaction(u256 const& _value, u256 const& _gasPrice, u256 const& _gas, bytes const& _data, u256 const& _nonce = 0): m_type(ContractCreation), m_nonce(_nonce), m_value(_value), m_gasPrice(_gasPrice), m_gas(_gas), m_data(_data) {}

	/// Constructs a transaction from the given RLP.
	explicit Transaction(bytesConstRef _rlp, CheckTransaction _checkSig);

	/// Constructs a transaction from the given RLP.
	explicit Transaction(bytes const& _rlp, CheckTransaction _checkSig): Transaction(&_rlp, _checkSig) {}


	/// Checks equality of transactions.
	bool operator==(Transaction const& _c) const { return m_type == _c.m_type && (m_type == ContractCreation || m_receiveAddress == _c.m_receiveAddress) && m_value == _c.m_value && m_data == _c.m_data; }
	/// Checks inequality of transactions.
	bool operator!=(Transaction const& _c) const { return !operator==(_c); }

	/// @returns sender of the transaction from the signature (and hash).
	Address const& sender() const;
	/// Like sender() but will never throw. @returns a null Address if the signature is invalid.
	Address const& safeSender() const noexcept;
	/// Force the sender to a particular value. This will result in an invalid transaction RLP.
	void forceSender(Address const& _a) { m_sender = _a; }

	/// @returns true if transaction is non-null.
	explicit operator bool() const { return m_type != NullTransaction; }

	/// @returns true if transaction is contract-creation.
	bool isCreation() const { return m_type == ContractCreation; }

	/// @returns true if transaction is message-call.
	bool isMessageCall() const { return m_type == MessageCall; }

	/// Serialises this transaction to an RLPStream.
	void streamRLP(RLPStream& _s, IncludeSignature _sig = WithSignature) const;

	/// @returns the RLP serialisation of this transaction.
	bytes rlp(IncludeSignature _sig = WithSignature) const { RLPStream s; streamRLP(s, _sig); return s.out(); }

	/// @returns the SHA3 hash of the RLP serialisation of this transaction.
	h256 sha3(IncludeSignature _sig = WithSignature) const { if (_sig == WithSignature && m_hashWith) return m_hashWith; RLPStream s; streamRLP(s, _sig); auto ret = dev::sha3(s.out()); if (_sig == WithSignature) m_hashWith = ret; return ret; }

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

	/// Synonym for receiveAddress().
	Address to() const { return m_receiveAddress; }

	/// Synonym for safeSender().
	Address from() const { return safeSender(); }

	/// @returns the data associated with this (message-call) transaction. Synonym for initCode().
	bytes const& data() const { return m_data; }
	/// @returns the initialisation code associated with this (contract-creation) transaction. Synonym for data().
	bytes const& initCode() const { return m_data; }

	/// @returns the transaction-count of the sender.
	u256 nonce() const { return m_nonce; }

	/// @returns the signature of the transaction. Encodes the sender.
	SignatureStruct const& signature() const { return m_vrs; }

	/// @returns true if the transaction contains enough gas for the basic payment.
	bool checkPayment() const { return m_gas >= gasRequired(); }

	/// @returns the gas required to run this transaction.
	bigint gasRequired() const;

	/// Get the fee associated for a transaction with the given data.
	template <class T> static bigint gasRequired(T const& _data, u256 _gas = 0) { bigint ret = c_txGas + _gas; for (auto i: _data) ret += i ? c_txDataNonZeroGas : c_txDataZeroGas; return ret; }

private:
	/// Type of transaction.
	enum Type
	{
		NullTransaction,				///< Null transaction.
		ContractCreation,				///< Transaction to create contracts - receiveAddress() is ignored.
		MessageCall						///< Transaction to invoke a message call - receiveAddress() is used.
	};

	void sign(Secret _priv);			///< Sign the transaction.

	Type m_type = NullTransaction;		///< Is this a contract-creation transaction or a message-call transaction?
	u256 m_nonce;						///< The transaction-count of the sender.
	u256 m_value;						///< The amount of ETH to be transferred by this transaction. Called 'endowment' for contract-creation transactions.
	Address m_receiveAddress;			///< The receiving address of the transaction.
	u256 m_gasPrice;					///< The base fee and thus the implied exchange rate of ETH to GAS.
	u256 m_gas;							///< The total gas to convert, paid for from sender's account. Any unused gas gets refunded once the contract is ended.
	bytes m_data;						///< The data associated with the transaction, or the initialiser if it's a creation transaction.
	SignatureStruct m_vrs;				///< The signature of the transaction. Encodes the sender.

	mutable h256 m_hashWith;			///< Cached hash of transaction with signature.
	mutable Address m_sender;			///< Cached sender, determined from signature.
	mutable bigint m_gasRequired = 0;	///< Memoised amount required for the transaction to run.
};

/// Nice name for vector of Transaction.
using Transactions = std::vector<Transaction>;

/// Simple human-readable stream-shift operator.
inline std::ostream& operator<<(std::ostream& _out, Transaction const& _t)
{
	_out << _t.sha3().abridged() << "{";
	if (_t.receiveAddress())
		_out << _t.receiveAddress().abridged();
	else
		_out << "[CREATE]";

	_out << "/" << _t.data().size() << "$" << _t.value() << "+" << _t.gas() << "@" << _t.gasPrice();
	_out << "<-" << _t.safeSender().abridged() << " #" << _t.nonce() << "}";
	return _out;
}

void badTransaction(bytesConstRef _tx, std::string const& _err);
inline void badTransaction(bytes const& _tx, std::string const& _err) { badTransaction(&_tx, _err); }

}
}
