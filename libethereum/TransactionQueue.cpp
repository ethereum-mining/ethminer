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

#include "TransactionQueue.h"

#include <libdevcore/Log.h>
#include <libethcore/Exceptions.h>
#include "Transaction.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

ImportResult TransactionQueue::import(bytesConstRef _transactionRLP)
{
	// Check if we already know this transaction.
	h256 h = sha3(_transactionRLP);

	UpgradableGuard l(m_lock);
	// TODO: keep old transactions around and check in State for nonce validity

	if (m_known.count(h))
		return ImportResult::AlreadyKnown;

	try
	{
		// Check validity of _transactionRLP as a transaction. To do this we just deserialise and attempt to determine the sender.
		// If it doesn't work, the signature is bad.
		// The transaction's nonce may yet be invalid (or, it could be "valid" but we may be missing a marginally older transaction).
		Transaction t(_transactionRLP, CheckTransaction::Everything);

		UpgradeGuard ul(l);
		// If valid, append to blocks.
		m_current[h] = t;
		m_known.insert(h);

		ctxq << "Queued vaguely legit-looking transaction" << h.abridged();
	}
	catch (Exception const& _e)
	{
		ctxq << "Ignoring invalid transaction: " <<  diagnostic_information(_e);
		return ImportResult::Malformed;
	}
	catch (std::exception const& _e)
	{
		ctxq << "Ignoring invalid transaction: " << _e.what();
		return ImportResult::Malformed;
	}

	return ImportResult::Success;
}

void TransactionQueue::setFuture(std::pair<h256, Transaction> const& _t)
{
	WriteGuard l(m_lock);
	if (m_current.count(_t.first))
	{
		m_current.erase(_t.first);
		m_unknown.insert(make_pair(_t.second.sender(), _t));
	}
}

void TransactionQueue::noteGood(std::pair<h256, Transaction> const& _t)
{
	WriteGuard l(m_lock);
	auto r = m_unknown.equal_range(_t.second.sender());
	for (auto it = r.first; it != r.second; ++it)
		m_current.insert(it->second);
	m_unknown.erase(r.first, r.second);
}

void TransactionQueue::drop(h256 _txHash)
{
	UpgradableGuard l(m_lock);

	if (!m_known.count(_txHash))
		return;

	UpgradeGuard ul(l);
	m_known.erase(_txHash);

	if (m_current.count(_txHash))
		m_current.erase(_txHash);
	else
	{
		for (auto i = m_unknown.begin(); i != m_unknown.end(); ++i)
			if (i->second.first == _txHash)
			{
				m_unknown.erase(i);
				break;
			}
	}
}
