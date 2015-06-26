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
const char* TransactionQueueTraceChannel::name() { return EthCyan " ┅▶"; }

ImportResult TransactionQueue::import(bytesConstRef _transactionRLP, ImportCallback const& _cb, IfDropped _ik)
{
	// Check if we already know this transaction.
	h256 h = sha3(_transactionRLP);

	Transaction t;
	ImportResult ir;
	{
		UpgradableGuard l(m_lock);

		ir = check_WITH_LOCK(h, _ik);
		if (ir != ImportResult::Success)
			return ir;

		try
		{
			t = Transaction(_transactionRLP, CheckTransaction::Everything);
			UpgradeGuard ul(l);
			ir = manageImport_WITH_LOCK(h, t, _cb);
		}
		catch (...)
		{
			return ImportResult::Malformed;
		}
	}
//	cdebug << "import-END: Nonce of" << t.sender() << "now" << maxNonce(t.sender());
	return ir;
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

//	cdebug << "import-BEGIN: Nonce of sender" << maxNonce(_transaction.sender());
	ImportResult ret;
	{
		UpgradableGuard l(m_lock);
		// TODO: keep old transactions around and check in State for nonce validity

		auto ir = check_WITH_LOCK(h, _ik);
		if (ir != ImportResult::Success)
			return ir;

		{
			UpgradeGuard ul(l);
			ret = manageImport_WITH_LOCK(h, _transaction, _cb);
		}
	}
//	cdebug << "import-END: Nonce of" << _transaction.sender() << "now" << maxNonce(_transaction.sender());
	return ret;
}

std::unordered_map<h256, Transaction> TransactionQueue::transactions() const
{
	ReadGuard l(m_lock);
	auto ret = m_current;
	for (auto const& i: m_future)
		if (i.second.nonce() < maxNonce_WITH_LOCK(i.second.sender()))
			ret.insert(i);
	return ret;
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
		clog(TransactionQueueTraceChannel) << "Queued vaguely legit-looking transaction" << _h;
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
//	cdebug << "txQ::maxNonce" << _a;
	ReadGuard l(m_lock);
	return maxNonce_WITH_LOCK(_a);
}

u256 TransactionQueue::maxNonce_WITH_LOCK(Address const& _a) const
{
	u256 ret = 0;
	auto r = m_senders.equal_range(_a);
	for (auto it = r.first; it != r.second; ++it)
		if (m_current.count(it->second))
		{
//			cdebug << it->first << "1+" << m_current.at(it->second).nonce();
			ret = max(ret, m_current.at(it->second).nonce() + 1);
		}
		else if (m_future.count(it->second))
		{
//			cdebug << it->first << "1+" << m_future.at(it->second).nonce();
			ret = max(ret, m_future.at(it->second).nonce() + 1);
		}
		else
		{
			cwarn << "ERRROR!!!!! m_senders references non-current transaction";
			cwarn << "Sender" << it->first << "has transaction" << it->second;
			cwarn << "Count of m_current for" << it->second << "is" << m_current.count(it->second);
		}
	return ret;
}

void TransactionQueue::insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p)
{
//	cdebug << "txQ::insertCurrent" << _p.first << _p.second.sender() << _p.second.nonce();
	m_senders.insert(make_pair(_p.second.sender(), _p.first));
	if (m_current.count(_p.first))
		cwarn << "Transaction hash" << _p.first << "already in current?!";
	m_current.insert(_p);
}

bool TransactionQueue::remove_WITH_LOCK(h256 const& _txHash)
{
//	cdebug << "txQ::remove" << _txHash;
	for (std::unordered_map<h256, Transaction>* pool: { &m_current, &m_future })
	{
		auto pit = pool->find(_txHash);
		if (pit != pool->end())
		{
			auto r = m_senders.equal_range(pit->second.sender());
			for (auto i = r.first; i != r.second; ++i)
				if (i->second == _txHash)
				{
					m_senders.erase(i);
					break;
				}
//			cdebug << "=> nonce" << pit->second.nonce();
			pool->erase(pit);
			return true;
		}
	}
	return false;
}

unsigned TransactionQueue::waiting(Address const& _a) const
{
	auto it = m_senders.equal_range(_a);
	unsigned ret = 0;
	for (auto i = it.first; i != it.second; ++i, ++ret) {}
	return ret;
}

void TransactionQueue::setFuture(std::pair<h256, Transaction> const& _t)
{
//	cdebug << "txQ::setFuture" << _t.first;
	WriteGuard l(m_lock);
	if (m_current.count(_t.first))
	{
		m_future.insert(_t);
		m_current.erase(_t.first);
	}
}

void TransactionQueue::noteGood(std::pair<h256, Transaction> const& _t)
{
//	cdebug << "txQ::noteGood" << _t.first;
	WriteGuard l(m_lock);
	auto r = m_senders.equal_range(_t.second.sender());
	for (auto it = r.first; it != r.second; ++it)
	{
		auto fit = m_future.find(it->second);
		if (fit != m_future.end())
		{
			m_current.insert(*fit);
			m_future.erase(fit);
		}
	}
}

void TransactionQueue::drop(h256 const& _txHash)
{
	UpgradableGuard l(m_lock);

	if (!m_known.count(_txHash))
		return;

	UpgradeGuard ul(l);
	m_dropped.insert(_txHash);
	m_known.erase(_txHash);

	remove_WITH_LOCK(_txHash);
}
