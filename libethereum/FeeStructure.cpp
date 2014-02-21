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
using namespace eth;

u256 const c_stepFee = 1;
u256 const c_dataFee = 20;
u256 const c_memoryFee = 5;
u256 const c_extroFee = 40;
u256 const c_cryptoFee = 20;
u256 const c_newContractFee = 100;
u256 const c_txFee = 100;

void FeeStructure::setMultiplier(u256 _x)
{
	m_stepFee = c_stepFee * _x;
	m_dataFee = c_dataFee * _x;
	m_memoryFee = c_memoryFee * _x;
	m_extroFee = c_extroFee * _x;
	m_cryptoFee = c_cryptoFee * _x;
	m_newContractFee = c_newContractFee * _x;
	m_txFee = c_txFee * _x;
}

u256 FeeStructure::multiplier() const
{
	return m_stepFee / c_stepFee;
}
