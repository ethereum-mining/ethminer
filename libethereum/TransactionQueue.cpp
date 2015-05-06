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

const char* TransactionQueueChannel::name() { return EthCyan "┉┅▶"; }

ImportResult TransactionQueue::import(bytesConstRef _transactionRLP, ImportCallback const& _cb, IfDropped _ik)
{
	// Check if we already know this transaction.
	h256 h = sha3(_transactionRLP);

	UpgradableGuard l(m_lock);

	auto ir = check_WITH_LOCK(h, _ik);
	if (ir != ImportResult::Success)
		return ir;

	Transaction t(_transactionRLP, CheckTransaction::Everything);
	UpgradeGuard ul(l);
	return manageImport_WITH_LOCK(h, t, _cb);
}

ImportResult TransactionQueue::check_WITH_LOCK(h256 const& _h, IfDropped _ik)
{
	if (m_known.count(_h))
		return ImportResult::AlreadyKnown;

	if (m_dropped.count(_h) && _ik == IfDropped::Ignore)
		return ImportResult::AlreadyInChain;

	return ImportResult::Success;
}

ImportResult TransactionQueue::import(Transaction const& _transaction, ImportCallback const& _cb, IfDropped _ik)
{
	// Check if we already know this transaction.
	h256 h = _transaction.sha3(WithSignature);

	UpgradableGuard l(m_lock);
	// TODO: keep old transactions around and check in State for nonce validity

	auto ir = check_WITH_LOCK(h, _ik);
	if (ir != ImportResult::Success)
		return ir;

	UpgradeGuard ul(l);
	return manageImport_WITH_LOCK(h, _transaction, _cb);
}

ImportResult TransactionQueue::manageImport_WITH_LOCK(h256 const& _h, Transaction const& _transaction, ImportCallback const& _cb)
{
	try
	{
		// Check validity of _transactionRLP as a transaction. To do this we just deserialise and attempt to determine the sender.
		// If it doesn't work, the signature is bad.
		// The transaction's nonce may yet be invalid (or, it could be "valid" but we may be missing a marginally older transaction).

		// If valid, append to blocks.
		insertCurrent_WITH_LOCK(make_pair(_h, _transaction));
		m_known.insert(_h);
		if (_cb)
			m_callbacks[_h] = _cb;
		ctxq << "Queued vaguely legit-looking transaction" << _h;
		m_onReady();
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

u256 TransactionQueue::maxNonce(Address const& _a) const
{
	cdebug << "txQ::maxNonce" << _a;
	ReadGuard l(m_lock);
	u256 ret = 0;
	auto r = m_senders.equal_range(_a);
	for (auto it = r.first; it != r.second; ++it)
	{
		cdebug << it->first << "1+" << m_current.at(it->second).nonce();
		DEV_IGNORE_EXCEPTIONS(ret = max(ret, m_current.at(it->second).nonce() + 1));
	}
	return ret;
}

void TransactionQueue::insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p)
{
	cdebug << "txQ::insertCurrent" << _p.first << _p.second.sender() << _p.second.nonce();
	m_senders.insert(make_pair(_p.second.sender(), _p.first));
	m_current.insert(_p);
}

bool TransactionQueue::removeCurrent_WITH_LOCK(h256 const& _txHash)
{
	cdebug << "txQ::removeCurrent" << _txHash;
	if (m_current.count(_txHash))
	{
		auto r = m_senders.equal_range(m_current[_txHash].sender());
		for (auto it = r.first; it != r.second; ++it)
			if (it->second == _txHash)
			{
				cdebug << "=> sender" << it->first;
				m_senders.erase(it);
				break;
			}
		cdebug << "=> nonce" << m_current[_txHash].nonce();
		m_current.erase(_txHash);
		return true;
	}
	return false;
}

void TransactionQueue::setFuture(std::pair<h256, Transaction> const& _t)
{
	WriteGuard l(m_lock);
	if (m_current.count(_t.first))
	{
		m_unknown.insert(make_pair(_t.second.sender(), _t));
		m_current.erase(_t.first);
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

void TransactionQueue::drop(h256 const& _txHash)
{
	UpgradableGuard l(m_lock);

	if (!m_known.count(_txHash))
		return;

	UpgradeGuard ul(l);
	m_dropped.insert(_txHash);
	m_known.erase(_txHash);

	if (!removeCurrent_WITH_LOCK(_txHash))
	{
		for (auto i = m_unknown.begin(); i != m_unknown.end(); ++i)
			if (i->second.first == _txHash)
			{
				m_unknown.erase(i);
				break;
			}
	}
}
