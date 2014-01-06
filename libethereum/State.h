/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file State.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <array>
#include <map>
#include <unordered_map>
#include "Common.h"
#include "RLP.h"
#include "Exceptions.h"
#include "BlockInfo.h"
#include "AddressState.h"
#include "Transaction.h"

namespace eth
{

class State
{
public:
	explicit State(Address _minerAddress);

	bool verify(bytes const& _block, uint _number = 0);
	bool execute(bytes const& _rlp) { try { Transaction t(_rlp); execute(t, t.sender()); } catch (...) { return false; } }

	bool isNormalAddress(Address _address) const;
	bool isContractAddress(Address _address) const;

	u256 balance(Address _id) const;
	void addBalance(Address _id, u256 _amount);
	// bigint as we don't want any accidental problems with -ve numbers.
	bool subBalance(Address _id, bigint _amount);

	u256 contractMemory(Address _contract, u256 _memory) const;
	u256 transactionsFrom(Address _address) const;

private:
	struct MinerFeeAdder
	{
		~MinerFeeAdder() { state->addBalance(state->m_minerAddress, fee); }
		State* state;
		u256 fee;
	};

	void execute(Transaction const& _t, Address _sender);
	void execute(Address _myAddress, Address _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* o_totalFee);

	std::map<Address, AddressState> m_current;
	BlockInfo m_previousBlock;
	BlockInfo m_currentBlock;

	Address m_minerAddress;

	static const u256 c_stepFee;
	static const u256 c_dataFee;
	static const u256 c_memoryFee;
	static const u256 c_extroFee;
	static const u256 c_cryptoFee;
	static const u256 c_newContractFee;
	static const u256 c_txFee;
};

}


