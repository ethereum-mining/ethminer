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
#include "Exceptions.h"

namespace boost { namespace spirit { class utree; } }
namespace sp = boost::spirit;

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
	SLT,
	SGT,
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
	CALLDATACOPY,
	CODESIZE,
	CODECOPY,
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

/// Convert from EVM code to simple EVM assembly language.
std::string disassemble(bytes const& _mem);

/// Compile a Low-level Lisp-like Language program into EVM-code.
class CompilerException: public Exception {};
class InvalidOperation: public CompilerException {};
class SymbolNotFirst: public CompilerException {};
class IntegerOutOfRange: public CompilerException {};
class StringTooLong: public CompilerException {};
class EmptyList: public CompilerException {};
class DataNotExecutable: public CompilerException {};
class IncorrectParameterCount: public CompilerException {};
class InvalidDeposit: public CompilerException {};
class InvalidOpCode: public CompilerException {};
class InvalidName: public CompilerException {};
class InvalidMacroArgs: public CompilerException {};
class BareSymbol: public CompilerException {};
bytes compileLLL(std::string const& _s, std::vector<std::string>* _errors = nullptr);

class CompilerState;
class CodeFragment;

class CodeLocation
{
	friend class CodeFragment;

public:
	CodeLocation(CodeFragment* _f);
	CodeLocation(CodeFragment* _f, unsigned _p): m_f(_f), m_pos(_p) {}

	unsigned get() const;
	void increase(unsigned _val);
	void set(unsigned _val);
	void set(CodeLocation _loc) { assert(_loc.m_f == m_f); set(_loc.m_pos); }
	void anchor();

	CodeLocation operator+(unsigned _i) const { return CodeLocation(m_f, m_pos + _i); }

private:
	CodeFragment* m_f;
	unsigned m_pos;
};

class CompilerState;

class CodeFragment
{
	friend class CodeLocation;

public:
	CodeFragment(sp::utree const& _t, CompilerState& _s, bool _allowASM = false);
	CodeFragment(bytes const& _c = bytes()): m_code(_c) {}

	bytes const& code() const { return m_code; }

	unsigned appendPush(u256 _l);
	void appendFragment(CodeFragment const& _f);
	void appendFragment(CodeFragment const& _f, unsigned _i);
	void appendInstruction(Instruction _i);

	CodeLocation appendPushLocation(unsigned _l = 0);
	void appendPushLocation(CodeLocation _l) { assert(_l.m_f == this); appendPushLocation(_l.m_pos); }

	CodeLocation appendJump() { auto ret = appendPushLocation(0); appendInstruction(Instruction::JUMP); return ret; }
	CodeLocation appendJumpI() { auto ret = appendPushLocation(0); appendInstruction(Instruction::JUMPI); return ret; }
	CodeLocation appendJump(CodeLocation _l) { auto ret = appendPushLocation(_l.m_pos); appendInstruction(Instruction::JUMP); return ret; }
	CodeLocation appendJumpI(CodeLocation _l) { auto ret = appendPushLocation(_l.m_pos); appendInstruction(Instruction::JUMPI); return ret; }

	void onePath() { assert(!m_totalDeposit && !m_baseDeposit); m_baseDeposit = m_deposit; m_totalDeposit = INT_MAX; }
	void otherPath() { donePath(); m_totalDeposit = m_deposit; m_deposit = m_baseDeposit; }
	void donePaths() { donePath(); m_totalDeposit = m_baseDeposit = 0; }
	void ignored() { m_baseDeposit = m_deposit; }
	void endIgnored() { m_deposit = m_baseDeposit; m_baseDeposit = 0; }

	unsigned size() const { return m_code.size(); }

private:
	template <class T> void error() { throw T(); }
	void constructOperation(sp::utree const& _t, CompilerState& _s);

	void donePath() { if (m_totalDeposit != INT_MAX && m_totalDeposit != m_deposit) error<InvalidDeposit>(); }

	int m_deposit = 0;
	int m_baseDeposit = 0;
	int m_totalDeposit = 0;
	bytes m_code;
	std::vector<unsigned> m_locs;
};

}
