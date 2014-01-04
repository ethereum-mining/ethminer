#pragma once

#include <exception>
#include <array>
#include <map>
#include <unordered_map>
#include "Trie.h"
#include "RLP.h"
#include "Common.h"

namespace eth
{

/// Virtual machine bytecode instruction.
enum class Instruction: uint8_t
{
	STOP = 0x00,		///< halts execution
	ADD,				///< Rx Ry Rz - sets Rz <- Rx + Ry mod 2^256
	SUB,				///< Rx Ry Rz - sets Rz <- Rx - Ry mod 2^256
	MUL,				///< Rx Ry Rz - sets Rz <- Rx * Ry mod 2^256
	DIV,				///< Rx Ry Rz - sets Rz <- floor(Rx / Ry)
	SDIV,				///< Rx Ry Rz - like DIV, except it treats values above 2^255 as negative (ie. 2^256 - x -> -x)
	MOD,				///< Rx Ry Rz - sets Rz <- Rx mod Ry
	SMOD,				///< Rx Ry Rz - like MOD, but for signed values just like SDIV (using Python's convention with negative numbers)
	EXP,				///< Rx Ry Rz - sets Rz <- Rx ^ Ry mod 2^256
	NEG,				///< Rx Ry - sets Ry <- 2^256 - Rx
	LT,					///< Rx Ry Rz - sets Rz <- 1 if Rx < Ry else 0
	LE,					///< Rx Ry Rz - sets Rz <- 1 if Rx <= Ry else 0
	GT,					///< Rx Ry Rz - sets Rz <- 1 if Rx > Ry else 0
	GE,					///< Rx Ry Rz - sets Rz <- 1 if Rx >= Ry else 0
	EQ,					///< Rx Ry Rz - sets Rz <- 1 if Rx = Ry else 0
	NOT,				///< Rx Ry - sets Ry <- 1 if Rx = 0 else 0
	MYADDRESS = 0x10,	///< Rx - sets Rx to the contract's own address
	TXSENDER,			///< pushes the transaction sender
	TXVALUE	,			///< pushes the transaction value
	TXFEE,				///< pushes the transaction fee
	TXDATAN,			///< pushes the number of data items
	TXDATA,				///< pops one item and pushes data item S[-1], or zero if index out of range
	BLK_PREVHASH,		///< pushes the hash of the previous block (NOT the current one since that's impossible!)
	BLK_COINBASE,		///< pushes the coinbase of the current block
	BLK_TIMESTAMP,		///< pushes the timestamp of the current block
	BLK_NUMBER,			///< pushes the current block number
	BLK_DIFFICULTY,		///< pushes the difficulty of the current block
	SHA256 = 0x20,		///< sets Ry <- SHA256(Rx)
	RIPEMD160,			///< Rx Ry - sets Ry <- RIPEMD160(Rx)
	ECMUL,				///< Rx Ry Rz Ra Rb - sets (Ra, Rb) = Rz * (Rx, Ry) in secp256k1, using (0,0) for the point at infinity
	ECADD,				///< Rx Ry Rz Ra Rb Rc - sets (Rb, Rc) = (Rx, Ry) + (Ra, Rb)
	ECSIGN,				///< Rx Ry Rz Ra Rb - sets(Rz, Ra, Rb)as the(r,s,prefix)values of an Electrum-style RFC6979 deterministic signature ofRxwith private keyRy`
	ECRECOVER,			///< Rx Ry Rz Ra Rb Rc - sets(Rb, Rc)as the public key from the signature(Ry, Rz, Ra)of the message hashRx`
	ECVALID,			///< Rx Ry Rz Ra Rb Rc - sets(Rb, Rc)as the public key from the signature(Ry, Rz, Ra)of the message hashRx`
	PUSH = 0x30,
	POP,
	DUP,
	DUPN,
	SWAP,
	SWAPN,
	LOAD,
	STORE,
	JMP = 0x40,			///< Rx - sets the index pointer to the value at Rx
	JMPI,				///< Rx Ry - if Rx != 0, sets the index pointer to Ry
	IND,				///< pushes the index pointer.
	EXTRO = 0x50,		///< Rx Ry Rz - looks at the contract at address Rx and its memory state Ry, and outputs the result to Rz
	BALANCE,			///< Rx - returns the ether balance of address Rx
	MKTX = 0x60,		///< Rx Ry Rz Rw Rv - sends Ry ether to Rx plus Rz fee with Rw data items starting from memory index Rv (and then reading to (Rv + 1), (Rv + 2) etc). Note that if Rx = 0 then this creates a new contract.
	SUICIDE = 0xff		///< Rx - destroys the contract and clears all memory, sending the entire balance plus the negative fee from clearing memory minus TXFEE to the address
};

class BadInstruction: public std::exception {};
class StackTooSmall: public std::exception { public: StackTooSmall(u256 _req, u256 _got): req(_req), got(_got) {} u256 req; u256 got; };
class OperandOutOfRange: public std::exception { public: OperandOutOfRange(u256 _min, u256 _max, u256 _got): mn(_min), mx(_max), got(_got) {} u256 mn; u256 mx; u256 got; };
class ExecutionException: public std::exception {};
class NoSuchContract: public std::exception {};
class InvalidTransactionFormat: public std::exception {};
class InvalidBlockFormat: public std::exception {};
class InvalidUnclesHash: public std::exception {};
class InvalidTransactionsHash: public std::exception {};
class InvalidTransaction: public std::exception {};
class InvalidDifficulty: public std::exception {};
class InvalidTimestamp: public std::exception {};
class InvalidNonce: public std::exception {};

struct BlockInfo
{
public:
	u256 hash;
	u256 parentHash;
	u256 sha256Uncles;
	u256 coinbaseAddress;
	u256 sha256Transactions;
	u256 difficulty;
	u256 timestamp;
	u256 nonce;
	u256 number;

	void populateAndVerify(bytesConstRef _block, u256 _number)
	{
		number = _number;

		RLP root(_block);
		try
		{
			RLP header = root[0];
			hash = eth::sha256(_block);
			parentHash = header[0].toFatInt();
			sha256Uncles = header[1].toFatInt();
			coinbaseAddress = header[2].toFatInt();
			sha256Transactions = header[3].toFatInt();
			difficulty = header[4].toFatInt();
			timestamp = header[5].toFatInt();
			nonce = header[6].toFatInt();
		}
		catch (RLP::BadCast)
		{
			throw InvalidBlockFormat();
		}

		if (sha256Transactions != sha256(root[1].data()))
			throw InvalidTransactionsHash();

		if (sha256Uncles != sha256(root[2].data()))
			throw InvalidUnclesHash();

		// TODO: check timestamp.
		// TODO: check difficulty against timestamp.
		// TODO: check proof of work.

		// TODO: check each transaction.
	}
};

enum class AddressType
{
	Normal,
	Contract
};

class AddressState
{
public:
	AddressState(AddressType _type = AddressType::Normal): m_type(_type), m_balance(0), m_nonce(0) {}

	AddressType type() const { return m_type; }
	u256& balance() { return m_balance; }
	u256 const& balance() const { return m_balance; }
	u256& nonce() { return m_nonce; }
	u256 const& nonce() const { return m_nonce; }
	std::map<u256, u256>& memory() { assert(m_type == AddressType::Contract); return m_memory; }
	std::map<u256, u256> const& memory() const { assert(m_type == AddressType::Contract); return m_memory; }

	u256 memoryHash() const
	{
		return hash256(m_memory);
	}

	std::string toString() const
	{
		if (m_type == AddressType::Normal)
			return rlpList(m_balance, toCompactBigEndianString(m_nonce));
		if (m_type == AddressType::Contract)
			return rlpList(m_balance, toCompactBigEndianString(m_nonce), toCompactBigEndianString(memoryHash()));
		return "";
	}

private:
	AddressType m_type;
	u256 m_balance;
	u256 m_nonce;
	u256Map m_memory;
};

template <class _T>
inline u160 low160(_T const& _t)
{
	return (u160)(_t & ((((_T)1) << 160) - 1));
}

template <class _T>
inline u160 as160(_T const& _t)
{
	return (u160)(_t & ((((_T)1) << 160) - 1));
}


struct Signature
{
	u256 v;
	u256 r;
	u256 s;

	u160 address(bytesConstRef _tx) const
	{
		return as160(s);
	}
};


// [ nonce, receiving_address, value, fee, [ data item 0, data item 1 ... data item n ], v, r, s ]
struct Transaction
{
	Transaction() {}
	Transaction(bytes const& _rlp);

	u256 nonce;
	u160 receiveAddress;
	u256 value;
	u256 fee;
	u256s data;
	Signature vrs;

	bytes rlp() const;
	u256 sha256() const { return eth::sha256(rlp()); }
};

class State
{
public:
	explicit State(u256 _minerAddress): m_minerAddress(_minerAddress) {}

	bool verify(bytes const& _block);
	bool execute(bytes const& _rlp) { try { Transaction t(_rlp); u160 sender = t.vrs.address(bytesConstRef(const_cast<bytes*>(&_rlp))); return execute(t, sender); } catch (...) { return false; } }	// remove const_cast once vector_ref can handle const vector* properly.

private:
	bool execute(Transaction const& _t);
	bool execute(Transaction const& _t, u160 _sender);

	bool isNormalAddress(u160 _address) const { auto it = m_current.find(_address); return it != m_current.end() && it->second.type() == AddressType::Normal; }
	bool isContractAddress(u160 _address) const { auto it = m_current.find(_address); return it != m_current.end() && it->second.type() == AddressType::Contract; }

	u256 balance(u160 _id) const { auto it = m_current.find(_id); return it == m_current.end() ? 0 : it->second.balance(); }
	void addBalance(u160 _id, u256 _amount) { auto it = m_current.find(_id); if (it == m_current.end()) it->second.balance() = _amount; else it->second.balance() += _amount; }
	// bigint as we don't want any accidental problems with -ve numbers.
	bool subBalance(u160 _id, bigint _amount) { auto it = m_current.find(_id); if (it == m_current.end() || (bigint)it->second.balance() < _amount) return false; it->second.balance() = (u256)((bigint)it->second.balance() - _amount); return true; }

	u256 contractMemory(u160 _contract, u256 _memory) const
	{
		auto m = m_current.find(_contract);
		if (m == m_current.end())
			return 0;
		auto i = m->second.memory().find(_memory);
		return i == m->second.memory().end() ? 0 : i->second;
	}

	u256 transactionsFrom(u160 _address) { auto it = m_current.find(_address); return it == m_current.end() ? 0 : it->second.nonce(); }

	void execute(u160 _myAddress, u160 _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* o_totalFee);

	std::map<u160, AddressState> m_current;


	BlockInfo m_previousBlock;
	BlockInfo m_currentBlock;

	u160 m_minerAddress;

	static const u256 c_stepFee;
	static const u256 c_dataFee;
	static const u256 c_memoryFee;
	static const u256 c_extroFee;
	static const u256 c_cryptoFee;
	static const u256 c_newContractFee;
};

}


