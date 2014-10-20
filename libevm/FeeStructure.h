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
/** @file FeeStructure.h
 * @author Gav Wood <i@gavwood.com>
 * @author Pawel Bylica <chfast@gmail.com>
 * @date 2014
 */

#pragma once

#include <cstdint>

#include <boost/config.hpp>

namespace dev
{
namespace eth
{

enum class Instruction: uint8_t;

struct FeeStructure
{
	static uint32_t const c_stepGas		= 1;	///< Once per operation, except for SSTORE, SLOAD, BALANCE, SHA3, CREATE, CALL.
	static uint32_t const c_balanceGas	= 20;	///< Once per BALANCE operation.
	static uint32_t const c_sha3Gas		= 20;	///< Once per SHA3 operation.
	static uint32_t const c_sloadGas	= 20;	///< Once per SLOAD operation.
	static uint32_t const c_sstoreGas	= 100;	///< Once per non-zero storage element in a CREATE call/transaction. Also, once/twice per SSTORE operation depending on whether the zeroness changes (twice iff it changes from zero; nothing at all if to zero) or doesn't (once).
	static uint32_t const c_createGas	= 100;	///< Once per CREATE operation & contract-creation transaction.
	static uint32_t const c_callGas		= 20;	///< Once per CALL operation & message call transaction.
	static uint32_t const c_memoryGas	= 1;	///< Times the address of the (highest referenced byte in memory + 1). NOTE: referencing happens on read, write and in instructions such as RETURN and CALL.
	static uint32_t const c_txDataGas	= 5;	///< Per byte of data attached to a transaction. NOTE: Not payable on data of calls between transactions.
	static uint32_t const c_txGas		= 500;	///< Per transaction. NOTE: Not payable on data of calls between transactions.

	/// Returns step fee of the instruction. 
	/// In case of bad instruction code, throws BadInstruction exception.
	static uint32_t getInstructionFee(Instruction _inst);
};

}
}
