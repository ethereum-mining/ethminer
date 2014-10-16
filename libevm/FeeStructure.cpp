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
/** @file FeeStructure.cpp
 * @author Gav Wood <i@gavwood.com>
 * @author Pawel Bylica <chfast@gmail.com>
 * @date 2014
 */

#include "FeeStructure.h"

#include <libevmface/Instruction.h>

#include "VM.h"

namespace dev
{
namespace eth
{

uint32_t FeeStructure::getInstructionFee(Instruction _inst)
{
	switch (_inst)
	{
	default:
		BOOST_THROW_EXCEPTION(BadInstruction());

	case Instruction::STOP:
	case Instruction::SUICIDE:
		return 0;

	case Instruction::SSTORE:
		return c_sstoreGas;

	case Instruction::SLOAD:
		return c_sloadGas;

	case Instruction::SHA3:
		return c_sha3Gas;

	case Instruction::BALANCE:
		return c_sha3Gas;

	case Instruction::CALL:
	case Instruction::CALLCODE:
		return c_callGas;

	case Instruction::CREATE:
		return c_createGas;

	case Instruction::ADD:
	case Instruction::MUL:
	case Instruction::SUB:
	case Instruction::DIV:
	case Instruction::SDIV:
	case Instruction::MOD:
	case Instruction::SMOD:
	case Instruction::EXP:
	case Instruction::NEG:
	case Instruction::LT:
	case Instruction::GT:
	case Instruction::SLT:
	case Instruction::SGT:
	case Instruction::EQ:
	case Instruction::NOT:
	case Instruction::AND:
	case Instruction::OR:
	case Instruction::XOR:
	case Instruction::BYTE:
	case Instruction::ADDMOD:
	case Instruction::MULMOD:
	case Instruction::ADDRESS:
	case Instruction::ORIGIN:
	case Instruction::CALLER:
	case Instruction::CALLVALUE:
	case Instruction::CALLDATALOAD:
	case Instruction::CALLDATASIZE:
	case Instruction::CODESIZE:
	case Instruction::EXTCODESIZE:
	case Instruction::GASPRICE:
	case Instruction::PREVHASH:
	case Instruction::COINBASE:
	case Instruction::TIMESTAMP:
	case Instruction::NUMBER:
	case Instruction::DIFFICULTY:
	case Instruction::GASLIMIT:
	case Instruction::PUSH1:
	case Instruction::PUSH2:
	case Instruction::PUSH3:
	case Instruction::PUSH4:
	case Instruction::PUSH5:
	case Instruction::PUSH6:
	case Instruction::PUSH7:
	case Instruction::PUSH8:
	case Instruction::PUSH9:
	case Instruction::PUSH10:
	case Instruction::PUSH11:
	case Instruction::PUSH12:
	case Instruction::PUSH13:
	case Instruction::PUSH14:
	case Instruction::PUSH15:
	case Instruction::PUSH16:
	case Instruction::PUSH17:
	case Instruction::PUSH18:
	case Instruction::PUSH19:
	case Instruction::PUSH20:
	case Instruction::PUSH21:
	case Instruction::PUSH22:
	case Instruction::PUSH23:
	case Instruction::PUSH24:
	case Instruction::PUSH25:
	case Instruction::PUSH26:
	case Instruction::PUSH27:
	case Instruction::PUSH28:
	case Instruction::PUSH29:
	case Instruction::PUSH30:
	case Instruction::PUSH31:
	case Instruction::PUSH32:
	case Instruction::POP:
	case Instruction::DUP1:
	case Instruction::DUP2:
	case Instruction::DUP3:
	case Instruction::DUP4:
	case Instruction::DUP5:
	case Instruction::DUP6:
	case Instruction::DUP7:
	case Instruction::DUP8:
	case Instruction::DUP9:
	case Instruction::DUP10:
	case Instruction::DUP11:
	case Instruction::DUP12:
	case Instruction::DUP13:
	case Instruction::DUP14:
	case Instruction::DUP15:
	case Instruction::DUP16:
	case Instruction::SWAP1:
	case Instruction::SWAP2:
	case Instruction::SWAP3:
	case Instruction::SWAP4:
	case Instruction::SWAP5:
	case Instruction::SWAP6:
	case Instruction::SWAP7:
	case Instruction::SWAP8:
	case Instruction::SWAP9:
	case Instruction::SWAP10:
	case Instruction::SWAP11:
	case Instruction::SWAP12:
	case Instruction::SWAP13:
	case Instruction::SWAP14:
	case Instruction::SWAP15:
	case Instruction::SWAP16:
	case Instruction::JUMP:
	case Instruction::JUMPI:
	case Instruction::PC:
	case Instruction::MSIZE:
	case Instruction::GAS:
	case Instruction::JUMPDEST:
	case Instruction::RETURN:
	case Instruction::MSTORE:
	case Instruction::MSTORE8:
	case Instruction::MLOAD:
	case Instruction::CALLDATACOPY:
	case Instruction::CODECOPY:
	case Instruction::EXTCODECOPY:
		return c_stepGas;
	}
}

}
}