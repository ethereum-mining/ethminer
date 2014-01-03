#pragma once

#include <exception>
#include <array>
#include <map>
#include <unordered_map>
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

struct BlockInfo
{
	u256 hash;
	u256 coinbase;
	u256 timestamp;
	u256 number;
	u256 difficulty;
};

class State
{
public:
	explicit State(u256 _minerAddress): m_minerAddress(_minerAddress) {}

	bool transact(bytes const& _rlp);

private:
	bool isContractAddress(u256 _address) const { return m_contractMemory.count(_address); }

	u256 balance(u256 _id) const { auto it = m_balance.find(_id); return it == m_balance.end() ? 0 : it->second; }
	void addBalance(u256 _id, u256 _amount) { auto it = m_balance.find(_id); if (it == m_balance.end()) it->second = _amount; else it->second += _amount; }
	// bigint as we don't want any accidental problems with -ve numbers.
	bool subBalance(u256 _id, bigint _amount) { auto it = m_balance.find(_id); if (it == m_balance.end() || (bigint)it->second < _amount) return false; it->second = (u256)((bigint)it->second - _amount); return true; }

	u256 memory(u256 _contract, u256 _memory) const
	{
		auto m = m_contractMemory.find(_contract);
		if (m == m_contractMemory.end())
			return 0;
		auto i = m->second.find(_memory);
		return i == m->second.end() ? 0 : i->second;
	}

	u256 transactionsFrom(u256 _address) { return 0; } // TODO

	void execute(u256 _myAddress, u256 _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* _totalFee);

	std::map<u256, u256>& ensureMemory(u256 _contract) { return m_contractMemory[_contract]; }

	std::map<u256, std::map<u256, u256>> m_contractMemory;
	std::map<u256, u256> m_balance;	// for now - might end up using Trie?

	BlockInfo m_previousBlock;
	BlockInfo m_currentBlock;

	u256 m_minerAddress;

	static const u256 c_stepFee;
	static const u256 c_dataFee;
	static const u256 c_memoryFee;
	static const u256 c_extroFee;
	static const u256 c_cryptoFee;
	static const u256 c_newContractFee;
};

}


