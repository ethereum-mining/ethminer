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
/** @file TransactionQueue.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <functional>
#include <condition_variable>
#include <thread>
#include <deque>
#include <libdevcore/Common.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Log.h>
#include <libethcore/Common.h>
#include "Transaction.h"

namespace dev
{
namespace eth
{

class BlockChain;

struct TransactionQueueChannel: public LogChannel { static const char* name(); static const int verbosity = 4; };
struct TransactionQueueTraceChannel: public LogChannel { static const char* name(); static const int verbosity = 7; };
#define ctxq dev::LogOutputStream<dev::eth::TransactionQueueTraceChannel, true>()

/// Import transaction policy
enum class IfDropped
{
	Ignore, ///< Don't import transaction that was previously dropped.
	Retry 	///< Import transaction even if it was dropped before.
};

/**
 * @brief A queue of Transactions, each stored as RLP.
 * Maintains a transaction queue sorted by nonce diff and gas price.
 * @threadsafe
 */
class TransactionQueue
{
public:
	/// @brief TransactionQueue
	/// @param _limit Maximum number of pending transactions in the queue.
	/// @param _futureLimit Maximum number of future nonce transactions.
	TransactionQueue(unsigned _limit = 1024, unsigned _futureLimit = 1024);
	~TransactionQueue();
	/// Add transaction to the queue to be verified and imported.
	/// @param _data RLP encoded transaction data.
	/// @param _nodeId Optional network identified of a node transaction comes from.
	void enqueue(RLP const& _data, h512 const& _nodeId);

	/// Verify and add transaction to the queue synchronously.
	/// @param _tx RLP encoded transaction data.
	/// @param _ik Set to Retry to force re-addinga transaction that was previously dropped.
	/// @returns Import result code.
	ImportResult import(bytes const& _tx, IfDropped _ik = IfDropped::Ignore) { return import(&_tx, _ik); }

	/// Verify and add transaction to the queue synchronously.
	/// @param _tx Trasnaction data.
	/// @param _ik Set to Retry to force re-addinga transaction that was previously dropped.
	/// @returns Import result code.
	ImportResult import(Transaction const& _tx, IfDropped _ik = IfDropped::Ignore);

	/// Remove transaction from the queue
	/// @param _txHash Trasnaction hash
	void drop(h256 const& _txHash);

	/// Get number of pending transactions for account.
	/// @returns Pending transaction count.
	unsigned waiting(Address const& _a) const;

	/// Get top transactions from the queue. Returned transactions are not removed from the queue automatically.
	/// @param _limit Max number of transactions to return.
	/// @returns up to _limit transactions ordered by nonce and gas price.
	Transactions topTransactions(unsigned _limit) const;

	/// Get a hash set of transactions in the queue
	/// @returns A hash set of all transactions in the queue
	h256Hash knownTransactions() const;

	/// Get max nonce for an account
	/// @returns Max transaction nonce for account in the queue
	u256 maxNonce(Address const& _a) const;

	/// Mark transaction as future. It wont be retured in topTransactions list until a transaction with a preceeding nonce is imported or marked with dropGood
	/// @param _t Transaction hash
	void setFuture(h256 const& _t);

	/// Drop a trasnaction from the list if exists and move following future trasnactions to current (if any)
	/// @param _t Transaction hash
	void dropGood(Transaction const& _t);

	/// Clear the queue
	void clear();

	/// Register a handler that will be called once there is a new transaction imported
	template <class T> Handler<> onReady(T const& _t) { return m_onReady.add(_t); }

	/// Register a handler that will be called once asynchronous verification is comeplte an transaction has been imported
	template <class T> Handler<ImportResult, h256 const&, h512 const&> onImport(T const& _t) { return m_onImport.add(_t); }

	/// Register a handler that will be called once asynchronous verification is comeplte an transaction has been imported
	template <class T> Handler<h256 const&> onReplaced(T const& _t) { return m_onReplaced.add(_t); }

private:

	/// Verified and imported transaction
	struct VerifiedTransaction
	{
		VerifiedTransaction(Transaction const& _t): transaction(_t) {}
		VerifiedTransaction(VerifiedTransaction&& _t): transaction(std::move(_t.transaction)) {}

		VerifiedTransaction(VerifiedTransaction const&) = delete;
		VerifiedTransaction& operator=(VerifiedTransaction const&) = delete;

		Transaction transaction; ///< Transaction data
	};

	/// Trasnaction pending verification
	struct UnverifiedTransaction
	{
		UnverifiedTransaction() {}
		UnverifiedTransaction(bytesConstRef const& _t, h512 const& _nodeId): transaction(_t.toBytes()), nodeId(_nodeId) {}
		UnverifiedTransaction(UnverifiedTransaction&& _t): transaction(std::move(_t.transaction)) {}
		UnverifiedTransaction& operator=(UnverifiedTransaction&& _other) { transaction = std::move(_other.transaction); nodeId = std::move(_other.nodeId); return *this; }

		UnverifiedTransaction(UnverifiedTransaction const&) = delete;
		UnverifiedTransaction& operator=(UnverifiedTransaction const&) = delete;

		bytes transaction;	///< RLP encoded transaction data
		h512 nodeId;		///< Network Id of the peer transaction comes from
	};

	struct PriorityCompare
	{
		TransactionQueue& queue;
		/// Compare transaction by nonce height and gas price.
		bool operator()(VerifiedTransaction const& _first, VerifiedTransaction const& _second) const
		{
			u256 const& height1 = _first.transaction.nonce() - queue.m_currentByAddressAndNonce[_first.transaction.sender()].begin()->first;
			u256 const& height2 = _second.transaction.nonce() - queue.m_currentByAddressAndNonce[_second.transaction.sender()].begin()->first;
			return height1 < height2 || (height1 == height2 && _first.transaction.gasPrice() > _second.transaction.gasPrice());
		}
	};

	// Use a set with dynamic comparator for minmax priority queue. The comparator takes into account min account nonce. Updating it does not affect the order.
	using PriorityQueue = std::multiset<VerifiedTransaction, PriorityCompare>;

	ImportResult import(bytesConstRef _tx, IfDropped _ik = IfDropped::Ignore);
	ImportResult check_WITH_LOCK(h256 const& _h, IfDropped _ik);
	ImportResult manageImport_WITH_LOCK(h256 const& _h, Transaction const& _transaction);

	void insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p);
	void makeCurrent_WITH_LOCK(Transaction const& _t);
	bool remove_WITH_LOCK(h256 const& _txHash);
	u256 maxNonce_WITH_LOCK(Address const& _a) const;
	void verifierBody();

	mutable SharedMutex m_lock;													///< General lock.
	h256Hash m_known;															///< Hashes of transactions in both sets.

	std::unordered_map<h256, std::function<void(ImportResult)>> m_callbacks;	///< Called once.
	h256Hash m_dropped;															///< Transactions that have previously been dropped

	PriorityQueue m_current;
	std::unordered_map<h256, PriorityQueue::iterator> m_currentByHash;			///< Transaction hash to set ref
	std::unordered_map<Address, std::map<u256, PriorityQueue::iterator>> m_currentByAddressAndNonce; ///< Transactions grouped by account and nonce
	std::unordered_map<Address, std::map<u256, VerifiedTransaction>> m_future;	/// Future transactions

	Signal<> m_onReady;															///< Called when a subsequent call to import transactions will return a non-empty container. Be nice and exit fast.
	Signal<ImportResult, h256 const&, h512 const&> m_onImport;					///< Called for each import attempt. Arguments are result, transaction id an node id. Be nice and exit fast.
	Signal<h256 const&> m_onReplaced;											///< Called whan transction is dropped during a call to import() to make room for another transaction.
	unsigned m_limit;															///< Max number of pending transactions
	unsigned m_futureLimit;														///< Max number of future transactions
	unsigned m_futureSize = 0;													///< Current number of future transactions

	std::condition_variable m_queueReady;										///< Signaled when m_unverified has a new entry.
	std::vector<std::thread> m_verifiers;
	std::deque<UnverifiedTransaction> m_unverified;								///< Pending verification queue
	mutable Mutex x_queue;														///< Verification queue mutex
	bool m_aborting = false;													///< Exit condition for verifier.
};

}
}

