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
#define ctxq dev::LogOutputStream<dev::eth::TransactionQueueChannel, true>()

enum class IfDropped { Ignore, Retry };

/**
 * @brief A queue of Transactions, each stored as RLP.
 * @threadsafe
 */
class TransactionQueue
{
public:
	using ImportCallback = std::function<void(ImportResult)>;

	ImportResult import(Transaction const& _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore);
	ImportResult import(bytes const& _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore) { return import(&_tx, _cb, _ik); }
	ImportResult import(bytesConstRef _tx, ImportCallback const& _cb = ImportCallback(), IfDropped _ik = IfDropped::Ignore);

	void drop(h256 const& _txHash);

	std::map<h256, Transaction> transactions() const { ReadGuard l(m_lock); return m_current; }
	std::pair<unsigned, unsigned> items() const { ReadGuard l(m_lock); return std::make_pair(m_current.size(), m_unknown.size()); }
	u256 maxNonce(Address const& _a) const;

	void setFuture(std::pair<h256, Transaction> const& _t);
	void noteGood(std::pair<h256, Transaction> const& _t);

	void clear() { WriteGuard l(m_lock); m_known.clear(); m_current.clear(); m_unknown.clear(); }
	template <class T> Handler onReady(T const& _t) { return m_onReady.add(_t); }

private:
	ImportResult check_WITH_LOCK(h256 const& _h, IfDropped _ik);
	ImportResult manageImport_WITH_LOCK(h256 const& _h, Transaction const& _transaction, ImportCallback const& _cb);

	void insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p);
	bool removeCurrent_WITH_LOCK(h256 const& _txHash);

	mutable SharedMutex m_lock;										///< General lock.
	std::set<h256> m_known;											///< Hashes of transactions in both sets.
	std::map<h256, Transaction> m_current;							///< Map of SHA3(tx) to tx.
	std::multimap<Address, std::pair<h256, Transaction>> m_unknown;	///< For transactions that have a future nonce; we map their sender address to the tx stuff, and insert once the sender has a valid TX.
	std::map<h256, std::function<void(ImportResult)>> m_callbacks;	///< Called once.
	std::set<h256> m_dropped;										///< Transactions that have previously been dropped.
	std::multimap<Address, h256> m_senders;							///< Mapping from the sender address to the transaction hash; useful for determining the nonce of a given sender.
	Signal m_onReady;												///< Called when a subsequent call to import transactions will return a non-empty container. Be nice and exit fast.
};

}
}

