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
	ExtVMFace() {}

	ExtVMFace(BlockInfo const& _previousBlock, BlockInfo const& _currentBlock, uint _currentNumber):
		previousBlock(_previousBlock),
		currentBlock(_currentBlock),
		currentNumber(_currentNumber)
	{}

	ExtVMFace(Address _myAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, bytesConstRef _code, BlockInfo const& _previousBlock, BlockInfo const& _currentBlock, uint _currentNumber):
		myAddress(_myAddress),
		txSender(_txSender),
		txValue(_txValue),
		gasPrice(_gasPrice),
		txData(_txData),
		code(_code),
		previousBlock(_previousBlock),
		currentBlock(_currentBlock),
		currentNumber(_currentNumber)
	{}

#pragma warning(push)
#pragma warning(disable: 4100)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

	byte getCode(u256 _n) const { return _n < code.size() ? code[(unsigned)_n] : 0; }
	u256 store(u256 _n) { return 0; }
	void setStore(u256 _n, u256 _v) {}
	u256 balance(Address _a) { return 0; }
	void subBalance(u256 _a) {}
	u256 txCount(Address _a) { return 0; }
	void suicide(Address _a) {}
	h160 create(u256 _endowment, u256* _gas, bytesConstRef _code, bytesConstRef _init) { return h160(); }
	bool call(Address _receiveAddress, u256 _txValue, bytesConstRef _txData, u256* _gas, bytesRef _tx) { return false; }

#pragma GCC diagnostic pop
#pragma warning(pop)

	Address myAddress;
	Address txSender;
	Address origin;
	u256 txValue;
	u256 gasPrice;
	bytesConstRef txData;
	bytesConstRef code;
	BlockInfo previousBlock;					///< The current block's information.
	BlockInfo currentBlock;						///< The current block's information.
	uint currentNumber;
};

}
