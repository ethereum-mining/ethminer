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

#include "Common.h"
#include "Transaction.h"

namespace eth
{

class BlockChain;

/**
 * @brief A queue of Transactions, each stored as RLP.
 */
class TransactionQueue
{
public:
	bool attemptImport(bytes const& _block) { try { import(_block); return true; } catch (...) { return false; } }
	bool import(bytes const& _block);
	void drop(h256 _txHash) { m_data.erase(_txHash); }
	std::map<h256, bytes> const& transactions() const { return m_data; }

	void setFuture(std::pair<h256, bytes> const& _t);
	void noteGood(std::pair<h256, bytes> const& _t);

	Transactions interestQueue() { Transactions ret; swap(ret, m_interestQueue); return ret; }
	void pushInterest(Address _a) { m_interest[_a]++; }
	void popInterest(Address _a) { if (m_interest[_a] > 1) m_interest[_a]--; else if (m_interest[_a]) m_interest.erase(_a); }

private:
	std::map<h256, bytes> m_data;		///< Map of SHA3(tx) to tx.
	Transactions m_interestQueue;
	std::map<Address, int> m_interest;
	std::multimap<Address, std::pair<h256, bytes>> m_future;		///< For transactions that have a future nonce; we map their sender address to the tx stuff, and insert once the sender has a valid TX.
};

}


