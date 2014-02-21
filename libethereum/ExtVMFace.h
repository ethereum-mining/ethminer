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
/** @file ExtVMFace.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
#include "FeeStructure.h"
#include "BlockInfo.h"

namespace eth
{

struct Transaction;

class ExtVMFace
{
public:
	ExtVMFace(Address _myAddress, Address _txSender, u256 _txValue, u256s const& _txData, FeeStructure const& _fees, BlockInfo const& _previousBlock, BlockInfo const& _currentBlock, uint _currentNumber):
		myAddress(_myAddress),
		txSender(_txSender),
		txValue(_txValue),
		txData(_txData),
		fees(_fees),
		previousBlock(_previousBlock),
		currentBlock(_currentBlock),
		currentNumber(_currentNumber)
	{}

	u256 store(u256 _n) { return 0; }
	void setStore(u256 _n, u256 _v) {}
	u256 temp(u256 _n) { return 0; }
	void setTemp(u256 _n, u256 _v) {}
	void mktx(Transaction& _t) {}
	u256 balance(Address _a) { return 0; }
	void payFee(bigint _fee) {}
	u256 txCount(Address _a) { return 0; }
	u256 extro(Address _a, u256 _pos) { return 0; }
	u256 extroPrice(Address _a) { return 0; }
	void suicide(Address _a) {}

	Address myAddress;
	Address txSender;
	u256 txValue;
	u256s const& txData;
	FeeStructure fees;
	BlockInfo previousBlock;					///< The current block's information.
	BlockInfo currentBlock;					///< The current block's information.
	uint currentNumber;
};

}
