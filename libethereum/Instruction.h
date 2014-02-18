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
	TXSENDER,			///< pushes the transaction sender
	TXVALUE	,			///< pushes the transaction value
	TXDATAN,			///< pushes the number of data items
	TXDATA,				///< pops one item and pushes data item S[-1], or zero if index out of range
	BLK_PREVHASH,		///< pushes the hash of the previous block (NOT the current one since that's impossible!)
	BLK_COINBASE,		///< pushes the coinbase of the current block
	BLK_TIMESTAMP,		///< pushes the timestamp of the current block
	BLK_NUMBER,			///< pushes the current block number
	BLK_DIFFICULTY,		///< pushes the difficulty of the current block
	BLK_NONCE,
	BASEFEE,
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
	SWAP,
	MLOAD,
	MSTORE,
	SLOAD,
	SSTORE,
	JMP,
	JMPI,
	IND,
	EXTRO,
	BALANCE,
	MKTX,
	SUICIDE = 0x3f
};

static const std::map<std::string, Instruction> c_instructions =
{
	{ "STOP", Instruction::STOP },
	{ "ADD", Instruction::ADD },
	{ "SUB", Instruction::SUB },
	{ "MUL", Instruction::MUL },
	{ "DIV", Instruction::DIV },
	{ "SDIV", Instruction::SDIV },
	{ "MOD", Instruction::MOD },
	{ "SMOD", Instruction::SMOD },
	{ "EXP", Instruction::EXP },
	{ "NEG", Instruction::NEG },
	{ "LT", Instruction::LT },
	{ "LE", Instruction::LE },
	{ "GT", Instruction::GT },
	{ "GE", Instruction::GE },
	{ "EQ", Instruction::EQ },
	{ "NOT", Instruction::NOT },
	{ "TXSENDER", Instruction::TXSENDER },
	{ "TXVALUE", Instruction::TXVALUE },
	{ "TXDATAN", Instruction::TXDATAN },
	{ "TXDATA", Instruction::TXDATA },
	{ "BLK_PREVHASH", Instruction::BLK_PREVHASH },
	{ "BLK_COINBASE", Instruction::BLK_COINBASE },
	{ "BLK_TIMESTAMP", Instruction::BLK_TIMESTAMP },
	{ "BLK_NUMBER", Instruction::BLK_NUMBER },
	{ "BLK_DIFFICULTY", Instruction::BLK_DIFFICULTY },
	{ "BLK_NONCE", Instruction::BLK_NONCE },
	{ "BASFEEE", Instruction::BASEFEE },
	{ "SHA256", Instruction::SHA256 },
	{ "RIPEMD160", Instruction::RIPEMD160 },
	{ "ECMUL", Instruction::ECMUL },
	{ "ECADD", Instruction::ECADD },
	{ "ECSIGN", Instruction::ECSIGN },
	{ "ECRECOVER", Instruction::ECRECOVER },
	{ "ECVALID", Instruction::ECVALID },
	{ "SHA3", Instruction::SHA3 },
	{ "PUSH", Instruction::PUSH },
	{ "POP", Instruction::POP },
	{ "DUP", Instruction::DUP },
	{ "SWAP", Instruction::SWAP },
	{ "MLOAD", Instruction::MLOAD },
	{ "MSTORE", Instruction::MSTORE },
	{ "SLOAD", Instruction::SLOAD },
	{ "SSTORE", Instruction::SSTORE },
	{ "JMP", Instruction::JMP },
	{ "JMPI", Instruction::JMPI },
	{ "IND", Instruction::IND },
	{ "EXTRO", Instruction::EXTRO },
	{ "BALANCE", Instruction::BALANCE },
	{ "MKTX", Instruction::MKTX },
	{ "SUICIDE", Instruction::SUICIDE }
};

u256s assemble(std::string const& _code);

}
