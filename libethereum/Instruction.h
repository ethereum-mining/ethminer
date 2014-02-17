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
/** @file Instruction.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"

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

static const std::map<std::string, Instruction> c_instructions =
{
	{ "STOP", (Instruction)0x00 },
	{ "ADD", (Instruction)0x01 },
	{ "SUB", (Instruction)0x02 },
	{ "MUL", (Instruction)0x03 },
	{ "DIV", (Instruction)0x04 },
	{ "SDIV", (Instruction)0x05 },
	{ "MOD", (Instruction)0x06 },
	{ "SMOD", (Instruction)0x07 },
	{ "EXP", (Instruction)0x08 },
	{ "NEG", (Instruction)0x09 },
	{ "LT", (Instruction)0x0a },
	{ "LE", (Instruction)0x0b },
	{ "GT", (Instruction)0x0c },
	{ "GE", (Instruction)0x0d },
	{ "EQ", (Instruction)0x0e },
	{ "NOT", (Instruction)0x0f },
	{ "MYADDRESS", (Instruction)0x10 },
	{ "TXSENDER", (Instruction)0x11 },
	{ "TXVALUE", (Instruction)0x12 },
	{ "TXFEE", (Instruction)0x13 },
	{ "TXDATAN", (Instruction)0x14 },
	{ "TXDATA", (Instruction)0x15 },
	{ "BLK_PREVHASH", (Instruction)0x16 },
	{ "BLK_COINBASE", (Instruction)0x17 },
	{ "BLK_TIMESTAMP", (Instruction)0x18 },
	{ "BLK_NUMBER", (Instruction)0x19 },
	{ "BLK_DIFFICULTY", (Instruction)0x1a },
	{ "SHA256", (Instruction)0x20 },
	{ "RIPEMD160", (Instruction)0x21 },
	{ "ECMUL", (Instruction)0x22 },
	{ "ECADD", (Instruction)0x23 },
	{ "ECSIGN", (Instruction)0x24 },
	{ "ECRECOVER", (Instruction)0x25 },
	{ "ECVALID", (Instruction)0x26 },
	{ "SHA3", (Instruction)0x27 },
	{ "PUSH", (Instruction)0x30 },
	{ "POP", (Instruction)0x31 },
	{ "DUP", (Instruction)0x32 },
	{ "DUPN", (Instruction)0x33 },
	{ "SWAP", (Instruction)0x34 },
	{ "SWAPN", (Instruction)0x35 },
	{ "LOAD", (Instruction)0x36 },
	{ "STORE", (Instruction)0x37 },
	{ "JMP", (Instruction)0x40 },
	{ "JMPI", (Instruction)0x41 },
	{ "IND", (Instruction)0x42 },
	{ "EXTRO", (Instruction)0x50 },
	{ "BALANCE", (Instruction)0x51 },
	{ "MKTX", (Instruction)0x60 },
	{ "SUICIDE", (Instruction)0xff }
};

u256s assemble(std::string const& _code);

}
