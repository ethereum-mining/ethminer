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
/** @file TransactionQueue.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Log.h"
#include "Transaction.h"
#include "TransactionQueue.h"
using namespace std;
using namespace eth;

bool TransactionQueue::import(bytes const& _block)
{
	// Check if we already know this transaction.
	h256 h = sha3(_block);
	if (m_data.count(h))
		return false;

	try
	{
		// Check validity of _block as a transaction. To do this we just deserialise and attempt to determine the sender. If it doesn't work, the signature is bad.
		// The transaction's nonce may yet be invalid (or, it could be "valid" but we may be missing a marginally older transaction).
		Transaction t(_block);
		auto s = t.sender();
		if (m_interest.count(s))
			m_interestQueue.push_back(t);

		// If valid, append to blocks.
		m_data[h] = _block;
	}
	catch (InvalidTransactionFormat const& _e)
	{
		cwarn << "Ignoring invalid transaction: " << _e.description();
		return false;
	}
	catch (std::exception const& _e)
	{
		cwarn << "Ignoring invalid transaction: " << _e.what();
		return false;
	}

	return true;
}

void TransactionQueue::setFuture(std::pair<h256, bytes> const& _t)
{
	if (m_data.count(_t.first))
	{
		m_data.erase(_t.first);
		m_future.insert(make_pair(Transaction(_t.second).sender(), _t));
	}
}

void TransactionQueue::noteGood(std::pair<h256, bytes> const& _t)
{
	auto r = m_future.equal_range(Transaction(_t.second).sender());
	for (auto it = r.first; it != r.second; ++it)
		m_data.insert(_t);
	m_future.erase(r.first, r.second);
}
