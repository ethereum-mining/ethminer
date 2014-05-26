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
 * @date 2014
 */

#pragma once

#include <libethsupport/Common.h>

namespace eth
{

extern u256 const c_stepGas;			///< Once per operation, except for SSTORE, SLOAD, BALANCE, SHA3, CREATE, CALL.
extern u256 const c_balanceGas;			///< Once per BALANCE operation.
extern u256 const c_sha3Gas;			///< Once per SHA3 operation.
extern u256 const c_sloadGas;			///< Once per SLOAD operation.
extern u256 const c_sstoreGas;			///< Once per non-zero storage element in a CREATE call/transaction. Also, once/twice per SSTORE operation depending on whether the zeroness changes (twice iff it changes from zero; nothing at all if to zero) or doesn't (once).
extern u256 const c_createGas;			///< Once per CREATE operation & contract-creation transaction.
extern u256 const c_callGas;			///< Once per CALL operation & message call transaction.
extern u256 const c_memoryGas;			///< Times the address of the (highest referenced byte in memory + 1). NOTE: referencing happens on read, write and in instructions such as RETURN and CALL.
extern u256 const c_txDataGas;			///< Per byte of data attached to a transaction. NOTE: Not payable on data of calls between transactions.
extern u256 const c_txGas;				///< Per transaction. NOTE: Not payable on data of calls between transactions.

}
