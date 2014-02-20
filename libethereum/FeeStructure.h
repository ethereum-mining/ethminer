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

#include "Common.h"

namespace eth
{

struct FeeStructure
{
	/// The fee structure. Values yet to be agreed on...
	void setMultiplier(u256 _x);				///< The current block multiplier.
	u256 multiplier() const;
	u256 m_stepFee;
	u256 m_dataFee;
	u256 m_memoryFee;
	u256 m_extroFee;
	u256 m_cryptoFee;
	u256 m_newContractFee;
	u256 m_txFee;
};

}
