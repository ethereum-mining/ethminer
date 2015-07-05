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

enum class IfDropped { Ignore, Retry };

/**
 * @brief A queue of Transactions, each stored as RLP.
 * Maintains a transaction queue sorted by nonce diff and gas price
 * @threadsafe
 */
class TransactionQueue
{
public:
	using ImportCallback = std::function<void(ImportResult)>;

	/// @brief TransactionQueue
	/// @param _limit Maximum number of pending transactions in the queue
	/// @param _futureLimit Maximum number of future nonce transactions
	TransactionQueue(unsigned _limit = 1024, unsigned _futureLimit = 1024): m_current(PriorityCompare { *this }), m_limit(_limit), m_futureLimit(_futureLimit) {}
	ImportResult import(Transaction const& _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore);
	ImportResult import(bytes const& _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore) { return import(&_tx, _cb, _ik); }
	ImportResult import(bytesConstRef _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore);

	void drop(h256 const& _txHash);

	unsigned waiting(Address const& _a) const;
	Transactions topTransactions(unsigned _limit) const;
	h256Hash knownTransactions() const;
	u256 maxNonce(Address const& _a) const;
	void setFuture(h256 const& _t);

	void clear();
	template <class T> Handler onReady(T const& _t) { return m_onReady.add(_t); }

private:
	struct VerifiedTransaction
	{
		VerifiedTransaction(Transaction const& _t): transaction(_t) {}
		VerifiedTransaction(VerifiedTransaction&& _t): transaction(std::move(_t.transaction)) {}

		VerifiedTransaction(VerifiedTransaction const&) = delete;
		VerifiedTransaction operator=(VerifiedTransaction const&) = delete;

		Transaction transaction;
	};

	struct PriorityCompare
	{
		TransactionQueue& queue;
		bool operator()(VerifiedTransaction const& _first, VerifiedTransaction const& _second) const
		{
			u256 const& height1 = _first.transaction.nonce() - queue.m_currentByAddressAndNonce[_first.transaction.sender()].begin()->first;
			u256 const& height2 = _second.transaction.nonce() - queue.m_currentByAddressAndNonce[_second.transaction.sender()].begin()->first;
			return height1 < height2 || (height1 == height2 && _first.transaction.gasPrice() > _second.transaction.gasPrice());
		}
	};

	// Use a set with dynamic comparator for minmax priority queue. The comparator takes into account min account nonce. Updating it does not affect the order.
	using PriorityQueue = std::multiset<VerifiedTransaction, PriorityCompare>;

	ImportResult check_WITH_LOCK(h256 const& _h, IfDropped _ik);
	ImportResult manageImport_WITH_LOCK(h256 const& _h, Transaction const& _transaction, ImportCallback const& _cb);

	void insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p);
	bool remove_WITH_LOCK(h256 const& _txHash);
	u256 maxNonce_WITH_LOCK(Address const& _a) const;

	mutable SharedMutex m_lock;													///< General lock.
	h256Hash m_known;															///< Hashes of transactions in both sets.

	std::unordered_map<h256, std::function<void(ImportResult)>> m_callbacks;	///< Called once.
	h256Hash m_dropped;															///< Transactions that have previously been dropped

	PriorityQueue m_current;
	std::unordered_map<h256, PriorityQueue::iterator> m_currentByHash;			///< Transaction hash to set ref
	std::unordered_map<Address, std::map<u256, PriorityQueue::iterator>> m_currentByAddressAndNonce; ///< Transactions grouped by account and nonce
	std::unordered_map<Address, std::map<u256, VerifiedTransaction>> m_future;	/// Future transactions

	Signal m_onReady;															///< Called when a subsequent call to import transactions will return a non-empty container. Be nice and exit fast.
	unsigned m_limit;															///< Max number of pending transactions
	unsigned m_futureLimit;														///< Max number of future transactions
	unsigned m_futureSize = 0;													///< Current number of future transactions
};

}
}

