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
/** @file State.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <array>
#include <map>
#include <unordered_map>
#include <libethential/Common.h>
#include <libethential/RLP.h>
#include <libethcore/TrieDB.h>
#include <libethcore/Exceptions.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/Dagger.h>
#include <libevm/FeeStructure.h>
#include <libevm/ExtVMFace.h>
#include "TransactionQueue.h"
#include "AddressState.h"
#include "Transaction.h"
#include "Executive.h"
#include "AccountDiff.h"

namespace eth
{

class BlockChain;

struct StateChat: public LogChannel { static const char* name() { return "=S="; } static const int verbosity = 4; };
struct StateTrace: public LogChannel { static const char* name() { return "=S="; } static const int verbosity = 7; };

struct TransactionReceipt
{
	TransactionReceipt(Transaction const& _t, h256 _root, u256 _gasUsed, Manifest const& _ms): transaction(_t), stateRoot(_root), gasUsed(_gasUsed), changes(_ms) {}

//	Manifest const& changes() const { return changes; }

	void fillStream(RLPStream& _s) const
	{
		_s.appendList(3);
		transaction.fillStream(_s);
		_s.append(stateRoot, false, true) << gasUsed;
	}

	Transaction transaction;
	h256 stateRoot;
	u256 gasUsed;
	Manifest changes;
};

/**
 * @brief Model of the current state of the ledger.
 * Maintains current ledger (m_current) as a fast hash-map. This is hashed only when required (i.e. to create or verify a block).
 * Should maintain ledger as of last N blocks, also, in case we end up on the wrong branch.
 */
class State
{
	friend class ExtVM;
	friend class Executive;

public:
	/// Construct state object.
	State(Address _coinbaseAddress = Address(), OverlayDB const& _db = OverlayDB());

	/// Construct state object from arbitrary point in blockchain.
	State(OverlayDB const& _db, BlockChain const& _bc, h256 _hash);

	/// Copy state object.
	State(State const& _s);

	/// Copy state object.
	State& operator=(State const& _s);

	/// Set the coinbase address for any transactions we do.
	/// This causes a complete reset of current block.
	void setAddress(Address _coinbaseAddress) { m_ourAddress = _coinbaseAddress; resetCurrent(); }
	Address address() const { return m_ourAddress; }

	/// Open a DB - useful for passing into the constructor & keeping for other states that are necessary.
	static OverlayDB openDB(std::string _path, bool _killExisting = false);
	static OverlayDB openDB(bool _killExisting = false) { return openDB(std::string(), _killExisting); }
	OverlayDB const& db() const { return m_db; }

	/// @returns the set containing all addresses currently in use in Ethereum.
	std::map<Address, u256> addresses() const;

	BlockInfo const& info() const { return m_currentBlock; }

	/// @brief Checks that mining the current object will result in a valid block.
	/// Effectively attempts to import the serialised block.
	/// @returns true if all is ok. If it's false, worry.
	bool amIJustParanoid(BlockChain const& _bc);

	/// Prepares the current state for mining.
	/// Commits all transactions into the trie, compiles uncles and transactions list, applies all
	/// rewards and populates the current block header with the appropriate hashes.
	/// The only thing left to do after this is to actually mine().
	///
	/// This may be called multiple times and without issue.
	void commitToMine(BlockChain const& _bc);

	/// Attempt to find valid nonce for block that this state represents.
	/// This function is thread-safe. You can safely have other interactions with this object while it is happening.
	/// @param _msTimeout Timeout before return in milliseconds.
	/// @returns Information on the mining.
	MineInfo mine(uint _msTimeout = 1000, bool _turbo = false);

	/** Commit to DB and build the final block if the previous call to mine()'s result is completion.
	 * Typically looks like:
	 * @code
	 * // lock
	 * commitToMine(blockchain);
	 * // unlock
	 * MineInfo info;
	 * for (info.complete = false; !info.complete; info = mine()) {}
	 * // lock
	 * completeMine();
	 * // unlock
	 * @endcode
	 */
	void completeMine();

	/// Get the complete current block, including valid nonce.
	/// Only valid after mine() returns true.
	bytes const& blockData() const { return m_currentBytes; }

	// TODO: Cleaner interface.
	/// Sync our transactions, killing those from the queue that we have and assimilating those that we don't.
	/// @returns a list of bloom filters one for each transaction placed from the queue into the state.
	/// @a o_transactionQueueChanged boolean pointer, the value of which will be set to true if the transaction queue
	/// changed and the pointer is non-null
	h256s sync(TransactionQueue& _tq, bool* o_transactionQueueChanged = nullptr);
	/// Like sync but only operate on _tq, killing the invalid/old ones.
	bool cull(TransactionQueue& _tq) const;

	/// Execute a given transaction.
	/// This will append @a _t to the transaction list and change the state accordingly.
	u256 execute(bytes const& _rlp, bytes* o_output = nullptr, bool _commit = true) { return execute(&_rlp, o_output, _commit); }
	u256 execute(bytesConstRef _rlp, bytes* o_output = nullptr, bool _commit = true);

	/// Get the remaining gas limit in this block.
	u256 gasLimitRemaining() const { return m_currentBlock.gasLimit - gasUsed(); }

	/// Check if the address is in use.
	bool addressInUse(Address _address) const;

	/// Check if the address contains executable code.
	bool addressHasCode(Address _address) const;

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

	/// Get the root of the storage of an account.
	h256 storageRoot(Address _contract) const;

	/// Get the value of a storage position of an account.
	/// @returns 0 if no account exists at that address.
	u256 storage(Address _contract, u256 _memory) const;

	/// Set the value of a storage position of an account.
	void setStorage(Address _contract, u256 _location, u256 _value) { m_cache[_contract].setStorage(_location, _value); }

	/// Get the storage of an account.
	/// @note This is expensive. Don't use it unless you need to.
	/// @returns std::map<u256, u256> if no account exists at that address.
	std::map<u256, u256> storage(Address _contract) const;

	/// Get the code of an account.
	/// @returns bytes() if no account exists at that address.
	bytes const& code(Address _contract) const;

	/// Note that the given address is sending a transaction and thus increment the associated ticker.
	void noteSending(Address _id);

	/// Get the number of transactions a particular address has sent (used for the transaction nonce).
	/// @returns 0 if the address has never been used.
	u256 transactionsFrom(Address _address) const;

	/// The hash of the root of our state tree.
	h256 rootHash() const { return m_state.root(); }

	/// Get the list of pending transactions.
	Transactions pending() const { Transactions ret; for (auto const& t: m_transactions) ret.push_back(t.transaction); return ret; }

	/// Get the list of pending transactions.
	Manifest changesFromPending(unsigned _i) const { return m_transactions[_i].changes; }

	/// Get the bloom filter of all changes happened in the block.
	h256 bloom() const;

	/// Get the bloom filter of a particular transaction that happened in the block.
	h256 bloom(unsigned _i) const { return m_transactions[_i].changes.bloom(); }

	/// Get the State immediately after the given number of pending transactions have been applied.
	/// If (_i == 0) returns the initial state of the block.
	/// If (_i == pending().size()) returns the final state of the block, prior to rewards.
	State fromPending(unsigned _i) const;

	/// @returns the StateDiff caused by the pending transaction of index @a _i.
	StateDiff pendingDiff(unsigned _i) const { return fromPending(_i).diff(fromPending(_i + 1)); }

	/// @return the difference between this state (origin) and @a _c (destination).
	StateDiff diff(State const& _c) const;

	/// Sync our state with the block chain.
	/// This basically involves wiping ourselves if we've been superceded and rebuilding from the transaction queue.
	bool sync(BlockChain const& _bc);

	/// Sync with the block chain, but rather than synching to the latest block, instead sync to the given block.
	bool sync(BlockChain const& _bc, h256 _blockHash, BlockInfo const& _bi = BlockInfo());

	/// Execute all transactions within a given block.
	/// @returns the additional total difficulty.
	u256 enactOn(bytesConstRef _block, BlockInfo const& _bi, BlockChain const& _bc);

	/// Returns back to a pristine state after having done a playback.
	/// @arg _fullCommit if true flush everything out to disk. If false, this effectively only validates
	/// the block since all state changes are ultimately reversed.
	void cleanup(bool _fullCommit);

private:
	/// Undo the changes to the state for committing to mine.
	void uncommitToMine();

	/// Retrieve all information about a given address into the cache.
	/// If _requireMemory is true, grab the full memory should it be a contract item.
	/// If _forceCreate is true, then insert a default item into the cache, in the case it doesn't
	/// exist in the DB.
	void ensureCached(Address _a, bool _requireCode, bool _forceCreate) const;

	/// Retrieve all information about a given address into a cache.
	void ensureCached(std::map<Address, AddressState>& _cache, Address _a, bool _requireCode, bool _forceCreate) const;

	/// Commit all changes waiting in the address cache to the DB.
	void commit();

	/// Execute the given block, assuming it corresponds to m_currentBlock. If _bc is passed, it will be used to check the uncles.
	/// Throws on failure.
	u256 enact(bytesConstRef _block, BlockChain const* _bc = nullptr, bool _checkNonce = true);

	// Two priviledged entry points for the VM (these don't get added to the Transaction lists):
	// We assume all instrinsic fees are paid up before this point.

	/// Execute a contract-creation transaction.
	h160 create(Address _txSender, u256 _endowment, u256 _gasPrice, u256* _gas, bytesConstRef _code, Address _originAddress = Address(), std::set<Address>* o_suicides = nullptr, PostList* o_posts = nullptr, Manifest* o_ms = nullptr, OnOpFunc const& _onOp = OnOpFunc(), unsigned _level = 0);

	/// Execute a call.
	/// @a _gas points to the amount of gas to use for the call, and will lower it accordingly.
	/// @returns false if the call ran out of gas before completion. true otherwise.
	bool call(Address _myAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, u256* _gas, bytesRef _out, Address _originAddress = Address(), std::set<Address>* o_suicides = nullptr, PostList* o_posts = nullptr, Manifest* o_ms = nullptr, OnOpFunc const& _onOp = OnOpFunc(), unsigned _level = 0);

	/// Sets m_currentBlock to a clean state, (i.e. no change from m_previousBlock).
	void resetCurrent();

	/// Finalise the block, applying the earned rewards.
	void applyRewards(Addresses const& _uncleAddresses);

	void refreshManifest(RLPStream* _txs = nullptr);

	/// @returns gas used by transactions thus far executed.
	u256 gasUsed() const { return m_transactions.size() ? m_transactions.back().gasUsed : 0; }

	bool isTrieGood(bool _enforceRefs, bool _requireNoLeftOvers) const;
	void paranoia(std::string const& _when, bool _enforceRefs = false) const;

	OverlayDB m_db;								///< Our overlay for the state tree.
	TrieDB<Address, OverlayDB> m_state;			///< Our state tree, as an OverlayDB DB.
	std::vector<TransactionReceipt> m_transactions;	///< The current list of transactions that we've included in the state.
	std::set<h256> m_transactionSet;			///< The set of transaction hashes that we've included in the state.
	OverlayDB m_lastTx;

	mutable std::map<Address, AddressState> m_cache;	///< Our address cache. This stores the states of each address that has (or at least might have) been changed.

	BlockInfo m_previousBlock;					///< The previous block's information.
	BlockInfo m_currentBlock;					///< The current block's information.
	bytes m_currentBytes;						///< The current block.

	bytes m_currentTxs;
	bytes m_currentUncles;

	Address m_ourAddress;						///< Our address (i.e. the address to which fees go).

	Dagger m_dagger;

	u256 m_blockReward;

	static std::string c_defaultPath;

	friend std::ostream& operator<<(std::ostream& _out, State const& _s);
};

std::ostream& operator<<(std::ostream& _out, State const& _s);

template <class DB>
void commit(std::map<Address, AddressState> const& _cache, DB& _db, TrieDB<Address, DB>& _state)
{
	for (auto const& i: _cache)
		if (!i.second.isAlive())
			_state.remove(i.first);
		else
		{
			RLPStream s(4);
			s << i.second.nonce() << i.second.balance();

			if (i.second.storage().empty())
				s.append(i.second.baseRoot(), false, true);
			else
			{
				TrieDB<h256, DB> storageDB(&_db, i.second.baseRoot());
				for (auto const& j: i.second.storage())
					if (j.second)
						storageDB.insert(j.first, rlp(j.second));
					else
						storageDB.remove(j.first);
				s.append(storageDB.root(), false, true);
			}

			if (i.second.isFreshCode())
			{
				h256 ch = sha3(i.second.code());
				_db.insert(ch, &i.second.code());
				if (i.second.code().size())
					s << ch;
				else
					s << "";
			}
			else
				if (i.second.codeHash() == EmptySHA3)
					s << "";
				else
					s << i.second.codeHash();

			_state.insert(i.first, &s.out());
		}
}

}


