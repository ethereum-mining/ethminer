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

#include <libethential/Common.h>
#include <libevmface/Instruction.h>
#include <libethcore/CommonEth.h>
#include <libethcore/BlockInfo.h>

namespace eth
{

struct Post
{
	Address from;
	Address to;
	u256 value;
	bytes data;
	u256 gas;
};

using PostList = std::list<Post>;

/**
 * @brief A null implementation of the class for specifying VM externalities.
 */
class ExtVMFace
{
public:
	/// Null constructor.
	ExtVMFace() {}

	/// Full constructor.
	ExtVMFace(Address _myAddress, Address _caller, Address _origin, u256 _value, u256 _gasPrice, bytesConstRef _data, bytesConstRef _code, BlockInfo const& _previousBlock, BlockInfo const& _currentBlock);

	/// Get the code at the given location in code ROM.
	byte getCode(u256 _n) const { return _n < code.size() ? code[(unsigned)_n] : 0; }

	/// Read storage location.
	u256 store(u256) { return 0; }

	/// Write a value in storage.
	void setStore(u256, u256) {}

	/// Read address's balance.
	u256 balance(Address) { return 0; }

	/// Subtract amount from account's balance.
	void subBalance(u256) {}

	/// Determine account's TX count.
	u256 txCount(Address) { return 0; }

	/// Suicide the associated contract and give proceeds to the given address.
	void suicide(Address) { suicides.insert(myAddress); }

	/// Create a new (contract) account.
	h160 create(u256, u256*, bytesConstRef, bytesConstRef) { return h160(); }

	/// Make a new message call.
	bool call(Address, u256, bytesConstRef, u256*, bytesRef) { return false; }

	/// Post a new message call.
	void post(Address _to, u256 _value, bytesConstRef _data, u256 _gas) { posts.push_back(Post({myAddress, _to, _value, _data.toBytes(), _gas})); }

	/// Revert any changes made (by any of the other calls).
	void revert() {}

	/// Execute any posts that may exist, including those that are incurred as a result of earlier posts.
	void doPosts() {}

	Address myAddress;			///< Address associated with executing code (a contract, or contract-to-be).
	Address caller;				///< Address which sent the message (either equal to origin or a contract).
	Address origin;				///< Original transactor.
	u256 value;					///< Value (in Wei) that was passed to this address.
	u256 gasPrice;				///< Price of gas (that we already paid).
	bytesConstRef data;			///< Current input data.
	bytesConstRef code;			///< Current code that is executing.
	BlockInfo previousBlock;	///< The previous block's information.
	BlockInfo currentBlock;		///< The current block's information.
	std::set<Address> suicides;	///< Any accounts that have suicided.
	std::list<Post> posts;		///< Any posts that have been made.
};

typedef std::function<void(uint64_t /*steps*/, Instruction /*instr*/, bigint /*newMemSize*/, bigint /*gasCost*/, void/*VM*/*, void/*ExtVM*/ const*)> OnOpFunc;

}
