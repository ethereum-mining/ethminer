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

#include <libethcore/Common.h>

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

	AND = 0x10,
	OR,
	XOR,
	BYTE,

	SHA3 = 0x20,

	ADDRESS = 0x30,
	BALANCE,
	ORIGIN,
	CALLER,
	CALLVALUE,
	CALLDATALOAD,
	CALLDATASIZE,
	GASPRICE,

	PREVHASH = 0x40,
	COINBASE,
	TIMESTAMP,
	NUMBER,
	DIFFICULTY,
	GASLIMIT,

	POP = 0x50,
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
	MEMSIZE,
	GAS,

	PUSH1 = 0x60,
	PUSH2,
	PUSH3,
	PUSH4,
	PUSH5,
	PUSH6,
	PUSH7,
	PUSH8,
	PUSH9,
	PUSH10,
	PUSH11,
	PUSH12,
	PUSH13,
	PUSH14,
	PUSH15,
	PUSH16,
	PUSH17,
	PUSH18,
	PUSH19,
	PUSH20,
	PUSH21,
	PUSH22,
	PUSH23,
	PUSH24,
	PUSH25,
	PUSH26,
	PUSH27,
	PUSH28,
	PUSH29,
	PUSH30,
	PUSH31,
	PUSH32,

	CREATE = 0xf0,
	CALL,
	RETURN,
	SUICIDE = 0xff
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
bytes assemble(std::string const& _code, bool _quiet = false);

/// Convert from EVM code to simple EVM assembly language.
std::string disassemble(bytes const& _mem);

/// Compile a Low-level Lisp-like Language program into EVM-code.
bytes compileLisp(std::string const& _code, bool _quiet, bytes& _init);

}
