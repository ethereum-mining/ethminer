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
#include <libethsupport/Common.h>
#include <libethsupport/RLP.h>
#include <libethsupport/TrieDB.h>
#include <libethcore/Exceptions.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/Dagger.h>
#include <libevm/FeeStructure.h>
#include <libevm/ExtVMFace.h>
#include "TransactionQueue.h"
#include "AddressState.h"
#include "Transaction.h"
#include "Executive.h"

namespace eth
{

class BlockChain;

struct StateChat: public LogChannel { static const char* name() { return "=S="; } static const int verbosity = 4; };
struct StateTrace: public LogChannel { static const char* name() { return "=S="; } static const int verbosity = 7; };

struct TransactionReceipt
{
	TransactionReceipt(Transaction const& _t, h256 _root, u256 _gasUsed): transaction(_t), stateRoot(_root), gasUsed(_gasUsed) {}

	void fillStream(RLPStream& _s) const
	{
		_s.appendList(3);
		transaction.fillStream(_s);
		_s.append(stateRoot, false, true) << gasUsed;
	}

	Transaction transaction;
	h256 stateRoot;
	u256 gasUsed;
};

enum class ExistDiff { Same, New, Dead };
template <class T>
class Diff
{
public:
	Diff() {}
	Diff(T _from, T _to): m_from(_from), m_to(_to) {}

	T const& from() const { return m_from; }
	T const& to() const { return m_to; }

	explicit operator bool() const { return m_from != m_to; }

private:
	T m_from;
	T m_to;
};

enum class AccountChange { None, Creation, Deletion, Intrinsic, CodeStorage, All };

struct AccountDiff
{
	inline bool changed() const { return storage.size() || code || nonce || balance || exist; }
	char const* lead() const;
	AccountChange changeType() const;

	Diff<bool> exist;
	Diff<u256> balance;
	Diff<u256> nonce;
	std::map<u256, Diff<u256>> storage;
	Diff<bytes> code;
};

struct StateDiff
{
	std::map<Address, AccountDiff> accounts;
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

	/// @returns the set containing all addresses currently in use in Ethereum.
	std::map<Address, u256> addresses() const;

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
	/// @param _msTimeout Timeout before return in milliseconds.
	/// @returns a non-empty byte array containing the block if it got lucky. In this case, call blockData()
	/// to get the block if you need it later.
	MineInfo mine(uint _msTimeout = 1000);

	/// Get the complete current block, including valid nonce.
	/// Only valid after mine() returns true.
	bytes const& blockData() const { return m_currentBytes; }

	/// Sync our state with the block chain.
	/// This basically involves wiping ourselves if we've been superceded and rebuilding from the transaction queue.
	bool sync(BlockChain const& _bc);

	/// Sync with the block chain, but rather than synching to the latest block, instead sync to the given block.
	bool sync(BlockChain const& _bc, h256 _blockHash);

	// TODO: Cleaner interface.
	/// Sync our transactions, killing those from the queue that we have and assimilating those that we don't.
	/// @returns true if we uncommitted from mining during the operation.
	/// @a o_changed boolean pointer, the value of which will be set to true if the state changed and the pointer
	/// is non-null
	bool sync(TransactionQueue& _tq, bool* o_changed = nullptr);
	/// Like sync but only operate on _tq, killing the invalid/old ones.
	bool cull(TransactionQueue& _tq) const;

	/// Execute a given transaction.
	/// This will append @a _t to the transaction list and change the state accordingly.
	u256 execute(bytes const& _rlp) { return execute(&_rlp); }
	u256 execute(bytesConstRef _rlp);

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

	/// Get the State immediately after the given number of pending transactions have been applied.
	/// If (_i == 0) returns the initial state of the block.
	/// If (_i == pending().size()) returns the final state of the block, prior to rewards.
	State fromPending(unsigned _i) const;

	/// Execute all transactions within a given block.
	/// @returns the additional total difficulty.
	/// If the _grandParent is passed, it will check the validity of each of the uncles.
	/// This might throw.
	u256 playback(bytesConstRef _block, BlockInfo const& _bi, BlockInfo const& _parent, BlockInfo const& _grandParent, bool _fullCommit);

	/// Get the fee associated for a transaction with the given data.
	u256 txGas(uint _dataCount, u256 _gas = 0) const { return c_txDataGas * _dataCount + c_txGas + _gas; }

	/// Get the fee associated for a contract created with the given data.
	u256 createGas(uint _dataCount, u256 _gas = 0) const { return txGas(_dataCount, _gas); }

	/// Get the fee associated for a normal transaction.
	u256 callGas(uint _dataCount, u256 _gas = 0) const { return txGas(_dataCount, _gas); }

	/// @return the difference between this state (origin) and @a _c (destination).
	StateDiff diff(State const& _c) const;

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

	/// Execute the given block on our previous block. This will set up m_currentBlock first, then call the other playback().
	/// Any failure will be critical.
	u256 trustedPlayback(bytesConstRef _block, bool _fullCommit);

	/// Execute the given block, assuming it corresponds to m_currentBlock. If _grandParent is passed, it will be used to check the uncles.
	/// Throws on failure.
	u256 playbackRaw(bytesConstRef _block, BlockInfo const& _grandParent, bool _fullCommit);

	// Two priviledged entry points for transaction processing used by the VM (these don't get added to the Transaction lists):
	// We assume all instrinsic fees are paid up before this point.

	/// Execute a contract-creation transaction.
	h160 create(Address _txSender, u256 _endowment, u256 _gasPrice, u256* _gas, bytesConstRef _code, Address _originAddress = Address());

	/// Execute a call.
	/// @a _gas points to the amount of gas to use for the call, and will lower it accordingly.
	/// @returns false if the call ran out of gas before completion. true otherwise.
	bool call(Address _myAddress, Address _txSender, u256 _txValue, u256 _gasPrice, bytesConstRef _txData, u256* _gas, bytesRef _out, Address _originAddress = Address());

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
//	GenericTrieDB<OverlayDB> m_transactionManifest;	///< The transactions trie; saved from the last commitToMine, or invalid/empty if commitToMine was never called.
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
std::ostream& operator<<(std::ostream& _out, StateDiff const& _s);
std::ostream& operator<<(std::ostream& _out, AccountDiff const& _s);

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
				s.append(i.second.oldRoot(), false, true);
			else
			{
				TrieDB<h256, DB> storageDB(&_db, i.second.oldRoot());
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
				s << ch;
			}
			else
				s << i.second.codeHash();

			_state.insert(i.first, &s.out());
		}
}

}


