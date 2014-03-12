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
	GT,
	EQ,
	NOT,
	ROTATE,
	AND,
	OR,
	XOR,
	SHA3,
	ADDRESS = 0x20,		///< pushes the transaction sender
	BALANCE,
	ORIGIN,				///< pushes the transaction sender
	CALLER,				///< pushes the transaction sender
	CALLVALUE,			///< pushes the transaction value
	CALLDATA,			///< pushes the transaction value
	CALLDATASIZE,		///< pushes the transaction value
	BASEFEE,
	MEMSIZE,
	PREVHASH,			///< pushes the hash of the previous block (NOT the current one since that's impossible!)
	PREVNONCE,
	COINBASE,			///< pushes the coinbase of the current block
	TIMESTAMP,			///< pushes the timestamp of the current block
	NUMBER,				///< pushes the current block number
	DIFFICULTY,			///< pushes the difficulty of the current block
	PUSH = 0x30,
	POP,
	DUP,
	SWAP,
	MLOAD,
	MSTORE,
	MSTORE8,
	SLOAD,
	SSTORE,
	JUMP,
	JUMPI,
	PC,
	CALL,
	RETURN,
	SUICIDE = 0x3f
};

/// Information structure for a particular instruction.
struct InstructionInfo
{
	char const* name;	///< The name of the instruction.
	int additional;		///< Additional items required in memory for this instructions (only for PUSH).
	int args;			///< Number of items required on the stack for this instruction (and, for the purposes of ret, the number taken from the stack).
	int ret;			///< Number of items placed (back) on the stack by this instruction, assuming args items were removed.
};

/// Information on all the instructions.
extern const std::map<Instruction, InstructionInfo> c_instructionInfo;

/// Convert from string mnemonic to Instruction type.
extern const std::map<std::string, Instruction> c_instructions;

/// Convert from simple EVM assembly language to EVM code.
u256s assemble(std::string const& _code, bool _quiet = false);

/// Convert from EVM code to simple EVM assembly language.
std::string disassemble(u256s const& _mem);

/// Compile a Low-level Lisp-like Language program into EVM-code.
u256s compileLisp(std::string const& _code, bool _quiet = false);

}
