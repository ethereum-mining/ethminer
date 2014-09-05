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

#include <boost/thread.hpp>
#include <libdevcore/Common.h>
#include "libethcore/CommonEth.h"
#include <libdevcore/Guards.h>

namespace dev
{
namespace eth
{

class BlockChain;

/**
 * @brief A queue of Transactions, each stored as RLP.
 * @threadsafe
 */
class TransactionQueue
{
public:
	bool attemptImport(bytesConstRef _tx) { try { import(_tx); return true; } catch (...) { return false; } }
	bool attemptImport(bytes const& _tx) { return attemptImport(&_tx); }
	bool import(bytesConstRef _tx);

	void drop(h256 _txHash);

	std::map<h256, bytes> transactions() const { ReadGuard l(m_lock); return m_current; }
	std::pair<unsigned, unsigned> items() const { ReadGuard l(m_lock); return std::make_pair(m_current.size(), m_future.size()); }

	void setFuture(std::pair<h256, bytes> const& _t);
	void noteGood(std::pair<h256, bytes> const& _t);

private:
	mutable boost::shared_mutex m_lock;							///< General lock.
	std::set<h256> m_known;										///< Hashes of transactions in both sets.
	std::map<h256, bytes> m_current;							///< Map of SHA3(tx) to tx.
	std::multimap<Address, std::pair<h256, bytes>> m_future;	///< For transactions that have a future nonce; we map their sender address to the tx stuff, and insert once the sender has a valid TX.
};

}
}

