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
/** @file AccountHolder.cpp
 * @authors:
 *   Christian R <c@ethdev.com>
 *   Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */

#include "AccountHolder.h"
#include <libethereum/Client.h>


using namespace std;
using namespace dev;
using namespace dev::eth;

vector<TransactionSkeleton> emptyQueue;

void AccountHolder::setAccounts(vector<KeyPair> const& _accounts)
{
	m_accounts.clear();
	for (auto const& keyPair: _accounts)
	{
		m_accounts.push_back(keyPair.address());
		m_keyPairs[keyPair.address()] = keyPair;
	}
}

vector<Address> AccountHolder::getAllAccounts() const
{
	vector<Address> accounts = m_accounts;
	for (auto const& pair: m_proxyAccounts)
		if (!isRealAccount(pair.first))
			accounts.push_back(pair.first);
	return accounts;
}

Address const& AccountHolder::getDefaultCallAccount() const
{
	if (m_accounts.empty())
		return ZeroAddress;
	Address const* bestMatch = &m_accounts.front();
	for (auto const& account: m_accounts)
		if (m_client()->balanceAt(account) > m_client()->balanceAt(*bestMatch))
			bestMatch = &account;
	return *bestMatch;
}

int AccountHolder::addProxyAccount(const Address& _account)
{
	int const c_id = m_transactionQueues.empty() ? 1 : m_transactionQueues.rbegin()->first + 1;
	if (isProxyAccount(_account))
		return 0;
	m_proxyAccounts.insert(make_pair(_account, c_id));
	m_transactionQueues[c_id].first = _account;
	return c_id;
}

bool AccountHolder::removeProxyAccount(unsigned _id)
{
	if (!m_transactionQueues.count(_id))
		return false;
	m_proxyAccounts.erase(m_transactionQueues[_id].first);
	m_transactionQueues.erase(_id);
	return true;
}

void AccountHolder::queueTransaction(TransactionSkeleton const& _transaction)
{
	if (!m_proxyAccounts.count(_transaction.from))
		return;
	int id = m_proxyAccounts[_transaction.from];
	m_transactionQueues[id].second.push_back(_transaction);
}

vector<TransactionSkeleton> const& AccountHolder::getQueuedTransactions(int _id) const
{
	if (!m_transactionQueues.count(_id))
		return emptyQueue;
	return m_transactionQueues.at(_id).second;
}

void AccountHolder::clearQueue(int _id)
{
	if (m_transactionQueues.count(_id))
		m_transactionQueues.at(_id).second.clear();
}
