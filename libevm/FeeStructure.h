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

#include <libdevcore/Common.h>

namespace dev
{
namespace eth
{

extern u256 const c_stepGas;			///< Once per operation, except for SSTORE, SLOAD, BALANCE, SHA3, CREATE, CALL.
extern u256 const c_balanceGas;			///< Once per BALANCE operation.
extern u256 const c_sha3Gas;			///< Once per SHA3 operation.
extern u256 const c_sloadGas;			///< Once per SLOAD operation.
extern u256 const c_sstoreSetGas;		///< Once per SSTORE operation if the zeroness changes from zero.
extern u256 const c_sstoreResetGas;		///< Once per SSTORE operation if the zeroness doesn't change.
extern u256 const c_sstoreRefundGas;	///< Refunded gas, once per SSTORE operation if the zeroness changes to zero.
extern u256 const c_createGas;			///< Once per CREATE operation & contract-creation transaction.
extern u256 const c_callGas;			///< Once per CALL operation & message call transaction.
extern u256 const c_memoryGas;			///< Times the address of the (highest referenced byte in memory + 1). NOTE: referencing happens on read, write and in instructions such as RETURN and CALL.
extern u256 const c_txDataZeroGas;		///< Per byte of data attached to a transaction that equals zero. NOTE: Not payable on data of calls between transactions.
extern u256 const c_txDataNonZeroGas;	///< Per byte of data attached to a transaction that is not equal to zero. NOTE: Not payable on data of calls between transactions.
extern u256 const c_txGas;				///< Per transaction. NOTE: Not payable on data of calls between transactions.
extern u256 const c_logGas;				///< Per LOG* operation.
extern u256 const c_logDataGas;			///< Per byte in a LOG* operation's data.
extern u256 const c_logTopicGas;		///< Multiplied by the * of the LOG*, per LOG transaction. e.g. LOG0 incurs 0 * c_txLogTopicGas, LOG4 incurs 4 * c_txLogTopicGas.
extern u256 const c_copyGas;			///< Multiplied by the number of 32-byte words that are copied (round up) for any *COPY operation and added.

}
}
