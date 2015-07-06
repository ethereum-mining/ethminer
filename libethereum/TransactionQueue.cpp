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

Transactions TransactionQueue::topTransactions(unsigned _limit) const
{
	ReadGuard l(m_lock);
	Transactions res;
	unsigned n = _limit;
	for (auto t = m_current.begin(); n != 0 && t != m_current.end(); ++t, --n)
		res.push_back(t->transaction);
	return res;
}

h256Hash TransactionQueue::knownTransactions() const
{
	ReadGuard l(m_lock);
	return m_known;
}

ImportResult TransactionQueue::manageImport_WITH_LOCK(h256 const& _h, Transaction const& _transaction, ImportCallback const& _cb)
{
	try
	{
		// Check validity of _transactionRLP as a transaction. To do this we just deserialise and attempt to determine the sender.
		// If it doesn't work, the signature is bad.
		// The transaction's nonce may yet be invalid (or, it could be "valid" but we may be missing a marginally older transaction).
		assert(_h == _transaction.sha3());

		// Remove any prior transaction with the same nonce but a lower gas price.
		// Bomb out if there's a prior transaction with higher gas price.
		auto cs = m_currentByAddressAndNonce.find(_transaction.from());
		if (cs != m_currentByAddressAndNonce.end())
		{
			auto t = cs->second.find(_transaction.nonce());
			if (t != cs->second.end())
			{
				if (_transaction.gasPrice() < (*t->second).transaction.gasPrice())
					return ImportResult::OverbidGasPrice;
				else
					remove_WITH_LOCK((*t->second).transaction.sha3());
			}
		}
		auto fs = m_future.find(_transaction.from());
		if (fs != m_future.end())
		{
			auto t = fs->second.find(_transaction.nonce());
			if (t != fs->second.end())
			{
				if (_transaction.gasPrice() < t->second.transaction.gasPrice())
					return ImportResult::OverbidGasPrice;
				else
				{
					fs->second.erase(t);
					--m_futureSize;
				}
			}
		}
		// If valid, append to blocks.
		insertCurrent_WITH_LOCK(make_pair(_h, _transaction));
		if (_cb)
			m_callbacks[_h] = _cb;
		clog(TransactionQueueTraceChannel) << "Queued vaguely legit-looking transaction" << _h;

		while (m_current.size() > m_limit)
		{
			clog(TransactionQueueTraceChannel) << "Dropping out of bounds transaction" << _h;
			remove_WITH_LOCK(m_current.rbegin()->transaction.sha3());
		}

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
	auto cs = m_currentByAddressAndNonce.find(_a);
	if (cs != m_currentByAddressAndNonce.end() && !cs->second.empty())
		ret = cs->second.rbegin()->first;
	auto fs = m_future.find(_a);
	if (fs != m_future.end() && !fs->second.empty())
		ret = std::max(ret, fs->second.rbegin()->first);
	return ret + 1;
}

void TransactionQueue::insertCurrent_WITH_LOCK(std::pair<h256, Transaction> const& _p)
{
	if (m_currentByHash.count(_p.first))
	{
		cwarn << "Transaction hash" << _p.first << "already in current?!";
		return;
	}

	Transaction const& t = _p.second;
	// Insert into current
	auto inserted = m_currentByAddressAndNonce[t.from()].insert(std::make_pair(t.nonce(), PriorityQueue::iterator()));
	PriorityQueue::iterator handle = m_current.emplace(VerifiedTransaction(t));
	inserted.first->second = handle;
	m_currentByHash[_p.first] = handle;

	// Move following transactions from future to current
	auto fs = m_future.find(t.from());
	if (fs != m_future.end())
	{
		u256 nonce = t.nonce() + 1;
		auto fb = fs->second.find(nonce);
		if (fb != fs->second.end())
		{
			auto ft = fb;
			while (ft != fs->second.end() && ft->second.transaction.nonce() == nonce)
			{
				inserted = m_currentByAddressAndNonce[t.from()].insert(std::make_pair(ft->second.transaction.nonce(), PriorityQueue::iterator()));
				PriorityQueue::iterator handle = m_current.emplace(move(ft->second));
				inserted.first->second = handle;
				m_currentByHash[(*handle).transaction.sha3()] = handle;
				--m_futureSize;
				++ft;
				++nonce;
			}
			fs->second.erase(fb, ft);
			if (fs->second.empty())
				m_future.erase(t.from());
		}
	}
	m_known.insert(_p.first);
}

bool TransactionQueue::remove_WITH_LOCK(h256 const& _txHash)
{
	auto t = m_currentByHash.find(_txHash);
	if (t == m_currentByHash.end())
		return false;

	Address from = (*t->second).transaction.from();
	auto it = m_currentByAddressAndNonce.find(from);
	assert (it != m_currentByAddressAndNonce.end());
	it->second.erase((*t->second).transaction.nonce());
	m_current.erase(t->second);
	m_currentByHash.erase(t);
	if (it->second.empty())
		m_currentByAddressAndNonce.erase(it);
	m_known.erase(_txHash);
	return true;
}

unsigned TransactionQueue::waiting(Address const& _a) const
{
	ReadGuard l(m_lock);
	unsigned ret = 0;
	auto cs = m_currentByAddressAndNonce.find(_a);
	if (cs != m_currentByAddressAndNonce.end())
		ret = cs->second.size();
	auto fs = m_future.find(_a);
	if (fs != m_future.end())
		ret += fs->second.size();
	return ret;
}

void TransactionQueue::setFuture(h256 const& _txHash)
{
//	cdebug << "txQ::setFuture" << _t.first;
	WriteGuard l(m_lock);
	auto it = m_currentByHash.find(_txHash);
	if (it == m_currentByHash.end())
		return;

	VerifiedTransaction const& st = *(it->second);

	Address from = st.transaction.from();
	auto& queue = m_currentByAddressAndNonce[from];
	auto& target = m_future[from];
	auto cutoff = queue.lower_bound(st.transaction.nonce());
	for (auto m = cutoff; m != queue.end(); ++m)
	{
		VerifiedTransaction& t = const_cast<VerifiedTransaction&>(*(m->second)); // set has only const iterators. Since we are moving out of container that's fine
		m_currentByHash.erase(t.transaction.sha3());
		target.emplace(t.transaction.nonce(), move(t));
		m_current.erase(m->second);
		++m_futureSize;
	}
	queue.erase(cutoff, queue.end());
	if (queue.empty())
		m_currentByAddressAndNonce.erase(from);
}

void TransactionQueue::drop(h256 const& _txHash)
{
	UpgradableGuard l(m_lock);

	if (!m_known.count(_txHash))
		return;

	UpgradeGuard ul(l);
	m_dropped.insert(_txHash);

	remove_WITH_LOCK(_txHash);

}

void TransactionQueue::clear()
{
	WriteGuard l(m_lock);
	m_known.clear();
	m_current.clear();
	m_currentByAddressAndNonce.clear();
	m_currentByHash.clear();
	m_future.clear();
	m_futureSize = 0;
}
