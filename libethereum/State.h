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
#include "TransactionQueue.h"
#include "Exceptions.h"
#include "BlockInfo.h"
#include "AddressState.h"
#include "Transaction.h"

namespace eth
{

class BlockChain;

/**
 * @brief Model of the current state of the ledger.
 * Maintains current ledger (m_current) as a fast hash-map. This is hashed only when required (i.e. to create or verify a block).
 * Should maintain ledger as of last N blocks, also, in case we end up on the wrong branch.
 * TODO: Block database I/O class.
 */
class State
{
public:
	/// Construct state object.
	explicit State(Address _minerAddress);

	/// Verify a given block.
	bool verify(bytes const& _block, uint _number = 0);

	/// Attempt to find valid nonce for block that this state represents.
	/// @param _msTimeout Timeout before return in milliseconds.
	/// @returns true if it got lucky.
	bool mine(uint _msTimeout = 1000) const;

	/// Get the complete current block, including valid nonce.
	bytes const& blockData() const { return m_currentBytes; }

	/// Sync our state with the block chain.
	/// This basically involves wiping ourselves if we've been superceded and rebuilding from the transaction queue.
	void sync(BlockChain const& _bc, TransactionQueue const& _tq);

	/// Execute a given transaction.
	bool execute(bytes const& _rlp) { try { Transaction t(_rlp); execute(t, t.sender()); } catch (...) { return false; } }

	/// Check if the address is a valid normal (non-contract) account address.
	bool isNormalAddress(Address _address) const;

	/// Check if the address is a valid contract's address.
	bool isContractAddress(Address _address) const;

	/// Get an account's balance.
	/// @returns 0 if the address has never been used.
	u256 balance(Address _id) const;

	/// Add some amount to balance.
	/// Will initialise the address if it has never been used.
	void addBalance(Address _id, u256 _amount);

	/** Subtract some amount from balance.
	 * @throws NotEnoughCash if balance of @a _id is less than @a _value (or has never been used).
	 * @note We use bigint here as we don't want any accidental problems with negative numbers.
	 */
	void subBalance(Address _id, bigint _value);

	/// Get the value of a memory position of a contract.
	/// @returns 0 if no contract exists at that address.
	u256 contractMemory(Address _contract, u256 _memory) const;

	/// Get the number of transactions a particular address has sent (used for the transaction nonce).
	/// @returns 0 if the address has never been used.
	u256 transactionsFrom(Address _address) const;

private:
	/// Fee-adder on destruction RAII class.
	struct MinerFeeAdder
	{
		~MinerFeeAdder() { state->addBalance(state->m_minerAddress, fee); }
		State* state;
		u256 fee;
	};

	/// Execute a decoded transaction object, given a sender.
	/// This will append @a _t to the transaction list and change the state accordingly.
	void execute(Transaction const& _t, Address _sender);

	/// Execute a contract transaction.
	void execute(Address _myAddress, Address _txSender, u256 _txValue, u256 _txFee, u256s const& _txData, u256* o_totalFee);

	// TODO: std::hash<Address> and then move to unordered_map.
	// Will need to sort on hash construction.
	std::map<Address, AddressState> m_current;	///< The current state. We work with a C++ hash map rather than a Trie.
	std::vector<Transaction> m_transactions;	///< The current list of transactions that we've included in the state.

	BlockInfo m_previousBlock;					///< The previous block's information.
	BlockInfo m_currentBlock;					///< The current block's information.
	bytes m_currentBytes;						///< The current block.

	Address m_minerAddress;						///< Our address (i.e. the address to which fees go).

	/// The fee structure. Values yet to be agreed on...
	static const u256 c_stepFee;
	static const u256 c_dataFee;
	static const u256 c_memoryFee;
	static const u256 c_extroFee;
	static const u256 c_cryptoFee;
	static const u256 c_newContractFee;
	static const u256 c_txFee;
};

}


