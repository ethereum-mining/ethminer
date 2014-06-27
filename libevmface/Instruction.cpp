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
/** @file Instruction.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Instruction.h"

#include <libethential/Common.h>
using namespace std;
using namespace eth;

const std::map<std::string, Instruction> eth::c_instructions =
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
	{ "GT", Instruction::GT },
	{ "SLT", Instruction::SLT },
	{ "SGT", Instruction::SGT },
	{ "EQ", Instruction::EQ },
	{ "NOT", Instruction::NOT },
	{ "AND", Instruction::AND },
	{ "OR", Instruction::OR },
	{ "XOR", Instruction::XOR },
	{ "BYTE", Instruction::BYTE },
	{ "SHA3", Instruction::SHA3 },
	{ "ADDRESS", Instruction::ADDRESS },
	{ "BALANCE", Instruction::BALANCE },
	{ "ORIGIN", Instruction::ORIGIN },
	{ "CALLER", Instruction::CALLER },
	{ "CALLVALUE", Instruction::CALLVALUE },
	{ "CALLDATALOAD", Instruction::CALLDATALOAD },
	{ "CALLDATASIZE", Instruction::CALLDATASIZE },
	{ "CALLDATACOPY", Instruction::CALLDATACOPY },
	{ "CODESIZE", Instruction::CODESIZE },
	{ "CODECOPY", Instruction::CODECOPY },
	{ "GASPRICE", Instruction::GASPRICE },
	{ "PREVHASH", Instruction::PREVHASH },
	{ "COINBASE", Instruction::COINBASE },
	{ "TIMESTAMP", Instruction::TIMESTAMP },
	{ "NUMBER", Instruction::NUMBER },
	{ "DIFFICULTY", Instruction::DIFFICULTY },
	{ "GASLIMIT", Instruction::GASLIMIT },
	{ "POP", Instruction::POP },
	{ "DUP", Instruction::DUP },
	{ "SWAP", Instruction::SWAP },
	{ "MLOAD", Instruction::MLOAD },
	{ "MSTORE", Instruction::MSTORE },
	{ "MSTORE8", Instruction::MSTORE8 },
	{ "SLOAD", Instruction::SLOAD },
	{ "SSTORE", Instruction::SSTORE },
	{ "JUMP", Instruction::JUMP },
	{ "JUMPI", Instruction::JUMPI },
	{ "PC", Instruction::PC },
	{ "MEMSIZE", Instruction::MEMSIZE },
	{ "GAS", Instruction::GAS },
	{ "PUSH1", Instruction::PUSH1 },
	{ "PUSH2", Instruction::PUSH2 },
	{ "PUSH3", Instruction::PUSH3 },
	{ "PUSH4", Instruction::PUSH4 },
	{ "PUSH5", Instruction::PUSH5 },
	{ "PUSH6", Instruction::PUSH6 },
	{ "PUSH7", Instruction::PUSH7 },
	{ "PUSH8", Instruction::PUSH8 },
	{ "PUSH9", Instruction::PUSH9 },
	{ "PUSH10", Instruction::PUSH10 },
	{ "PUSH11", Instruction::PUSH11 },
	{ "PUSH12", Instruction::PUSH12 },
	{ "PUSH13", Instruction::PUSH13 },
	{ "PUSH14", Instruction::PUSH14 },
	{ "PUSH15", Instruction::PUSH15 },
	{ "PUSH16", Instruction::PUSH16 },
	{ "PUSH17", Instruction::PUSH17 },
	{ "PUSH18", Instruction::PUSH18 },
	{ "PUSH19", Instruction::PUSH19 },
	{ "PUSH20", Instruction::PUSH20 },
	{ "PUSH21", Instruction::PUSH21 },
	{ "PUSH22", Instruction::PUSH22 },
	{ "PUSH23", Instruction::PUSH23 },
	{ "PUSH24", Instruction::PUSH24 },
	{ "PUSH25", Instruction::PUSH25 },
	{ "PUSH26", Instruction::PUSH26 },
	{ "PUSH27", Instruction::PUSH27 },
	{ "PUSH28", Instruction::PUSH28 },
	{ "PUSH29", Instruction::PUSH29 },
	{ "PUSH30", Instruction::PUSH30 },
	{ "PUSH31", Instruction::PUSH31 },
	{ "PUSH32", Instruction::PUSH32 },
	{ "CREATE", Instruction::CREATE },
	{ "CALL", Instruction::CALL },
	{ "RETURN", Instruction::RETURN },
	{ "SUICIDE", Instruction::SUICIDE }
};

const std::map<Instruction, InstructionInfo> eth::c_instructionInfo =
{ //                                           Add, Args, Ret
	{ Instruction::STOP,         { "STOP",         0, 0, 0 } },
	{ Instruction::ADD,          { "ADD",          0, 2, 1 } },
	{ Instruction::SUB,          { "SUB",          0, 2, 1 } },
	{ Instruction::MUL,          { "MUL",          0, 2, 1 } },
	{ Instruction::DIV,          { "DIV",          0, 2, 1 } },
	{ Instruction::SDIV,         { "SDIV",         0, 2, 1 } },
	{ Instruction::MOD,          { "MOD",          0, 2, 1 } },
	{ Instruction::SMOD,         { "SMOD",         0, 2, 1 } },
	{ Instruction::EXP,          { "EXP",          0, 2, 1 } },
	{ Instruction::NEG,          { "NEG",          0, 1, 1 } },
	{ Instruction::LT,           { "LT",           0, 2, 1 } },
	{ Instruction::GT,           { "GT",           0, 2, 1 } },
	{ Instruction::SLT,          { "SLT",          0, 2, 1 } },
	{ Instruction::SGT,          { "SGT",          0, 2, 1 } },
	{ Instruction::EQ,           { "EQ",           0, 2, 1 } },
	{ Instruction::NOT,          { "NOT",          0, 1, 1 } },
	{ Instruction::AND,          { "AND",          0, 2, 1 } },
	{ Instruction::OR,           { "OR",           0, 2, 1 } },
	{ Instruction::XOR,          { "XOR",          0, 2, 1 } },
	{ Instruction::BYTE,         { "BYTE",         0, 2, 1 } },
	{ Instruction::SHA3,         { "SHA3",         0, 2, 1 } },
	{ Instruction::ADDRESS,      { "ADDRESS",      0, 0, 1 } },
	{ Instruction::BALANCE,      { "BALANCE",      0, 1, 1 } },
	{ Instruction::ORIGIN,       { "ORIGIN",       0, 0, 1 } },
	{ Instruction::CALLER,       { "CALLER",       0, 0, 1 } },
	{ Instruction::CALLVALUE,    { "CALLVALUE",    0, 0, 1 } },
	{ Instruction::CALLDATALOAD, { "CALLDATALOAD", 0, 1, 1 } },
	{ Instruction::CALLDATASIZE, { "CALLDATASIZE", 0, 0, 1 } },
	{ Instruction::CALLDATACOPY, { "CALLDATACOPY", 0, 3, 0 } },
	{ Instruction::CODESIZE,     { "CODESIZE",     0, 0, 1 } },
	{ Instruction::CODECOPY,     { "CODECOPY",     0, 3, 0 } },
	{ Instruction::GASPRICE,     { "GASPRICE",     0, 0, 1 } },
	{ Instruction::PREVHASH,     { "PREVHASH",     0, 0, 1 } },
	{ Instruction::COINBASE,     { "COINBASE",     0, 0, 1 } },
	{ Instruction::TIMESTAMP,    { "TIMESTAMP",    0, 0, 1 } },
	{ Instruction::NUMBER,       { "NUMBER",       0, 0, 1 } },
	{ Instruction::DIFFICULTY,   { "DIFFICULTY",   0, 0, 1 } },
	{ Instruction::GASLIMIT,     { "GASLIMIT",     0, 0, 1 } },
	{ Instruction::POP,          { "POP",          0, 1, 0 } },
	{ Instruction::DUP,          { "DUP",          0, 1, 2 } },
	{ Instruction::SWAP,         { "SWAP",         0, 2, 2 } },
	{ Instruction::MLOAD,        { "MLOAD",        0, 1, 1 } },
	{ Instruction::MSTORE,       { "MSTORE",       0, 2, 0 } },
	{ Instruction::MSTORE8,      { "MSTORE8",      0, 2, 0 } },
	{ Instruction::SLOAD,        { "SLOAD",        0, 1, 1 } },
	{ Instruction::SSTORE,       { "SSTORE",       0, 2, 0 } },
	{ Instruction::JUMP,         { "JUMP",         0, 1, 0 } },
	{ Instruction::JUMPI,        { "JUMPI",        0, 2, 0 } },
	{ Instruction::PC,           { "PC",           0, 0, 1 } },
	{ Instruction::MEMSIZE,      { "MEMSIZE",      0, 0, 1 } },
	{ Instruction::GAS,          { "GAS",          0, 0, 1 } },
	{ Instruction::PUSH1,        { "PUSH1",        1, 0, 1 } },
	{ Instruction::PUSH2,        { "PUSH2",        2, 0, 1 } },
	{ Instruction::PUSH3,        { "PUSH3",        3, 0, 1 } },
	{ Instruction::PUSH4,        { "PUSH4",        4, 0, 1 } },
	{ Instruction::PUSH5,        { "PUSH5",        5, 0, 1 } },
	{ Instruction::PUSH6,        { "PUSH6",        6, 0, 1 } },
	{ Instruction::PUSH7,        { "PUSH7",        7, 0, 1 } },
	{ Instruction::PUSH8,        { "PUSH8",        8, 0, 1 } },
	{ Instruction::PUSH9,        { "PUSH9",        9, 0, 1 } },
	{ Instruction::PUSH10,       { "PUSH10",       10, 0, 1 } },
	{ Instruction::PUSH11,       { "PUSH11",       11, 0, 1 } },
	{ Instruction::PUSH12,       { "PUSH12",       12, 0, 1 } },
	{ Instruction::PUSH13,       { "PUSH13",       13, 0, 1 } },
	{ Instruction::PUSH14,       { "PUSH14",       14, 0, 1 } },
	{ Instruction::PUSH15,       { "PUSH15",       15, 0, 1 } },
	{ Instruction::PUSH16,       { "PUSH16",       16, 0, 1 } },
	{ Instruction::PUSH17,       { "PUSH17",       17, 0, 1 } },
	{ Instruction::PUSH18,       { "PUSH18",       18, 0, 1 } },
	{ Instruction::PUSH19,       { "PUSH19",       19, 0, 1 } },
	{ Instruction::PUSH20,       { "PUSH20",       20, 0, 1 } },
	{ Instruction::PUSH21,       { "PUSH21",       21, 0, 1 } },
	{ Instruction::PUSH22,       { "PUSH22",       22, 0, 1 } },
	{ Instruction::PUSH23,       { "PUSH23",       23, 0, 1 } },
	{ Instruction::PUSH24,       { "PUSH24",       24, 0, 1 } },
	{ Instruction::PUSH25,       { "PUSH25",       25, 0, 1 } },
	{ Instruction::PUSH26,       { "PUSH26",       26, 0, 1 } },
	{ Instruction::PUSH27,       { "PUSH27",       27, 0, 1 } },
	{ Instruction::PUSH28,       { "PUSH28",       28, 0, 1 } },
	{ Instruction::PUSH29,       { "PUSH29",       29, 0, 1 } },
	{ Instruction::PUSH30,       { "PUSH30",       30, 0, 1 } },
	{ Instruction::PUSH31,       { "PUSH31",       31, 0, 1 } },
	{ Instruction::PUSH32,       { "PUSH32",       32, 0, 1 } },
	{ Instruction::CREATE,       { "CREATE",       0, 3, 1 } },
	{ Instruction::CALL,         { "CALL",         0, 7, 1 } },
	{ Instruction::RETURN,       { "RETURN",       0, 2, 0 } },
	{ Instruction::SUICIDE,      { "SUICIDE",      0, 1, 0} }
};

string eth::disassemble(bytes const& _mem)
{
	stringstream ret;
	uint numerics = 0;
	for (auto it = _mem.begin(); it != _mem.end(); ++it)
	{
		byte n = *it;
		auto iit = c_instructionInfo.find((Instruction)n);
		if (numerics || iit == c_instructionInfo.end() || (byte)iit->first != n)	// not an instruction or expecting an argument...
		{
			if (numerics)
				numerics--;
			ret << "0x" << hex << (int)n << " ";
		}
		else
		{
			auto const& ii = iit->second;
			ret << ii.name << " ";
			numerics = ii.additional;
		}
	}
	return ret.str();
}
