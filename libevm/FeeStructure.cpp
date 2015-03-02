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

u256 const dev::eth::c_tierStepGas[8] = { 0, 2, 3, 5, 8, 10, 20, 0 };
u256 const dev::eth::c_expGas = 10;
u256 const dev::eth::c_expByteGas = 10;

u256 const dev::eth::c_sha3Gas = 30;
u256 const dev::eth::c_sha3WordGas = 6;

u256 const dev::eth::c_sloadGas = 50;
u256 const dev::eth::c_sstoreSetGas = 20000;
u256 const dev::eth::c_sstoreResetGas = 5000;
u256 const dev::eth::c_sstoreClearGas = 5000;
u256 const dev::eth::c_sstoreRefundGas = 15000;
u256 const dev::eth::c_jumpdestGas = 1;

u256 const dev::eth::c_logGas = 2000;
u256 const dev::eth::c_logDataGas = 8;
u256 const dev::eth::c_logTopicGas = 2000;

u256 const dev::eth::c_createGas = 32000;

u256 const dev::eth::c_callGas = 40;
u256 const dev::eth::c_callValueTransferGas = 6700;
u256 const dev::eth::c_callNewAccountGas = 25000;

u256 const dev::eth::c_suicideRefundGas = 24000;

u256 const dev::eth::c_memoryGas = 3;
u256 const dev::eth::c_quadCoeffDiv = 512;


u256 const dev::eth::c_createDataGas = 200;
u256 const dev::eth::c_txGas = 21000;
u256 const dev::eth::c_txDataZeroGas = 37;
u256 const dev::eth::c_txDataNonZeroGas = 2;

u256 const dev::eth::c_copyGas = 3;


