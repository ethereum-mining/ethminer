#pragma once

namespace eth
{

// TODO: Update comments.

/// Virtual machine bytecode instruction.
enum class Instruction: uint8_t
{
	STOP = 0x00,		///< halts execution
	ADD,
	SUB,
	MUL,
	DIV,
	SDIV,
	MOD,
	SMOD,
	EXP,
	NEG,
	LT,
	LE,
	GT,
	GE,
	EQ,
	NOT,
	MYADDRESS = 0x10,
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
	SHA256 = 0x20,
	RIPEMD160,
	ECMUL,
	ECADD,
	ECSIGN,
	ECRECOVER,
	ECVALID,
	SHA3,
	PUSH = 0x30,
	POP,
	DUP,
	DUPN,
	SWAP,
	SWAPN,
	LOAD,
	STORE,
	JMP = 0x40,
	JMPI,
	IND,
	EXTRO = 0x50,
	BALANCE,
	MKTX = 0x60,
	SUICIDE = 0xff
};

}
