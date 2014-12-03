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
 * @date 2014
 */

#include "FeeStructure.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

u256 const dev::eth::c_stepGas = 1;
u256 const dev::eth::c_balanceGas = 20;
u256 const dev::eth::c_sha3Gas = 20;
u256 const dev::eth::c_sloadGas = 20;
u256 const dev::eth::c_sstoreSetGas = 300;
u256 const dev::eth::c_sstoreResetGas = 100;
u256 const dev::eth::c_sstoreRefundGas = 100;
u256 const dev::eth::c_createGas = 100;
u256 const dev::eth::c_callGas = 20;
u256 const dev::eth::c_expGas = 1;
u256 const dev::eth::c_expByteGas = 1;
u256 const dev::eth::c_memoryGas = 1;
u256 const dev::eth::c_txDataZeroGas = 1;
u256 const dev::eth::c_txDataNonZeroGas = 5;
u256 const dev::eth::c_txGas = 500;
u256 const dev::eth::c_logGas = 32;
u256 const dev::eth::c_logDataGas = 1;
u256 const dev::eth::c_logTopicGas = 32;
u256 const dev::eth::c_copyGas = 1;
