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
	MUL,
	SUB,
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
	MYADDRESS,			///< pushes the transaction sender
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

/// Information structure for a particular instruction.
struct InstructionInfo
{
	std::string name;	///< The name of the instruction.
	int additional;		///< Additional items required in memory for this instructions (only for PUSH).
	int args;			///< Number of items required on the stack for this instruction (and, for the purposes of ret, the number taken from the stack).
	int ret;			///< Number of items placed (back) on the stack by this instruction, assuming args items were removed.
};

/// Information on all the instructions.
static const std::map<Instruction, InstructionInfo> c_instructionInfo =
{
	{ Instruction::STOP, { "STOP", 0, 0, 0 } },
	{ Instruction::ADD, { "ADD", 0, 2, 1 } },
	{ Instruction::SUB, { "SUB", 0, 2, 1 } },
	{ Instruction::MUL, { "MUL", 0, 2, 1 } },
	{ Instruction::DIV, { "DIV", 0, 2, 1 } },
	{ Instruction::SDIV, { "SDIV", 0, 2, 1 } },
	{ Instruction::MOD, { "MOD", 0, 2, 1 } },
	{ Instruction::SMOD, { "SMOD", 0, 2, 1 } },
	{ Instruction::EXP, { "EXP", 0, 2, 1 } },
	{ Instruction::NEG, { "NEG", 0, 1, 1 } },
	{ Instruction::LT, { "LT", 0, 2, 1 } },
	{ Instruction::LE, { "LE", 0, 2, 1 } },
	{ Instruction::GT, { "GT", 0, 2, 1 } },
	{ Instruction::GE, { "GE", 0, 2, 1 } },
	{ Instruction::EQ, { "EQ", 0, 2, 1 } },
	{ Instruction::NOT, { "NOT", 0, 1, 1 } },
	{ Instruction::MYADDRESS, { "MYADDRESS", 0, 0, 1 } },
	{ Instruction::TXSENDER, { "TXSENDER", 0, 0, 1 } },
	{ Instruction::TXVALUE, { "TXVALUE", 0, 0, 1 } },
	{ Instruction::TXDATAN, { "TXDATAN", 0, 0, 1 } },
	{ Instruction::TXDATA, { "TXDATA", 0, 1, 1 } },
	{ Instruction::BLK_PREVHASH, { "BLK_PREVHASH", 0, 0, 1 } },
	{ Instruction::BLK_COINBASE, { "BLK_COINBASE", 0, 0, 1 } },
	{ Instruction::BLK_TIMESTAMP, { "BLK_TIMESTAMP", 0, 0, 1 } },
	{ Instruction::BLK_NUMBER, { "BLK_NUMBER", 0, 0, 1 } },
	{ Instruction::BLK_DIFFICULTY, { "BLK_DIFFICULTY", 0, 0, 1 } },
	{ Instruction::BLK_NONCE, { "BLK_NONCE", 0, 0, 1 } },
	{ Instruction::BASEFEE, { "BASEFEE", 0, 0, 1 } },
	{ Instruction::SHA256, { "SHA256", 0, -1, 1 } },
	{ Instruction::RIPEMD160, { "RIPEMD160", 0, -1, 1 } },
	{ Instruction::ECMUL, { "ECMUL", 0, 3, 1 } },
	{ Instruction::ECADD, { "ECADD", 0, 4, 1 } },
	{ Instruction::ECSIGN, { "ECSIGN", 0, 2, 1 } },
	{ Instruction::ECRECOVER, { "ECRECOVER", 0, 4, 1 } },
	{ Instruction::ECVALID, { "ECVALID", 0, 2, 1 } },
	{ Instruction::SHA3, { "SHA3", 0, -1, 1 } },
	{ Instruction::PUSH, { "PUSH", 1, 0, 1 } },
	{ Instruction::POP, { "POP", 0, 1, 0 } },
	{ Instruction::DUP, { "DUP", 0, 1, 2 } },
	{ Instruction::SWAP, { "SWAP", 0, 2, 2 } },
	{ Instruction::MLOAD, { "MLOAD", 0, 1, 1 } },
	{ Instruction::MSTORE, { "MSTORE", 0, 2, 0 } },
	{ Instruction::SLOAD, { "SLOAD", 0, 1, 1 } },
	{ Instruction::SSTORE, { "SSTORE", 0, 2, 0 } },
	{ Instruction::JMP, { "JMP", 0, 1, 0 } },
	{ Instruction::JMPI, { "JMPI", 0, 2, 0 } },
	{ Instruction::IND, { "IND", 0, 0, 1 } },
	{ Instruction::EXTRO, { "EXTRO", 0, 2, 1 } },
	{ Instruction::BALANCE, { "BALANCE", 0, 1, 1 } },
	{ Instruction::MKTX, { "MKTX", 0, -3, 0 } },
	{ Instruction::SUICIDE, { "SUICIDE", 0, 1, 0} }
};

/// Convert from string mnemonic to Instruction type.
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
	{ "MYADDRESS", Instruction::MYADDRESS },
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
	{ "BASEFEE", Instruction::BASEFEE },
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

/// Convert from simple EVM assembly language to EVM code.
u256s assemble(std::string const& _code, bool _quiet = false);

/// Convert from EVM code to simple EVM assembly language.
std::string disassemble(u256s const& _mem);

/// Compile a Low-level Lisp-like Language program into EVM-code.
u256s compileLisp(std::string const& _code, bool _quiet = false);

}
