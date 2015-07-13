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
#include <random>
#include <ctime>
#include <libdevcore/Guards.h>
#include <libethereum/Client.h>
#include <libethcore/KeyManager.h>


using namespace std;
using namespace dev;
using namespace dev::eth;

vector<TransactionSkeleton> g_emptyQueue;
static std::mt19937 g_randomNumberGenerator(time(0));
static Mutex x_rngMutex;

vector<Address> AccountHolder::allAccounts() const
{
	vector<Address> accounts;
	accounts += realAccounts();
	for (auto const& pair: m_proxyAccounts)
		if (!isRealAccount(pair.first))
			accounts.push_back(pair.first);
	return accounts;
}

Address const& AccountHolder::defaultTransactAccount() const
{
	auto accounts = realAccounts();
	if (accounts.empty())
		return ZeroAddress;
	Address const* bestMatch = &*accounts.begin();
	for (auto const& account: accounts)
		if (m_client()->balanceAt(account) > m_client()->balanceAt(*bestMatch))
			bestMatch = &account;
	return *bestMatch;
}

int AccountHolder::addProxyAccount(const Address& _account)
{
	Guard g(x_rngMutex);
	int id = std::uniform_int_distribution<int>(1)(g_randomNumberGenerator);
	id = int(u256(FixedHash<32>(sha3(bytesConstRef((byte*)(&id), sizeof(int) / sizeof(byte))))));
	if (isProxyAccount(_account) || id == 0 || m_transactionQueues.count(id))
		return 0;
	m_proxyAccounts.insert(make_pair(_account, id));
	m_transactionQueues[id].first = _account;
	return id;
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

vector<TransactionSkeleton> const& AccountHolder::queuedTransactions(int _id) const
{
	if (!m_transactionQueues.count(_id))
		return g_emptyQueue;
	return m_transactionQueues.at(_id).second;
}

void AccountHolder::clearQueue(int _id)
{
	if (m_transactionQueues.count(_id))
		m_transactionQueues.at(_id).second.clear();
}

AddressHash SimpleAccountHolder::realAccounts() const
{
	return m_keyManager.accountsHash();
}

h256 SimpleAccountHolder::authenticate(dev::eth::TransactionSkeleton const& _t)
{
	if (isRealAccount(_t.from))
		return m_client()->submitTransaction(_t, m_keyManager.secret(_t.from, [&](){ return m_getPassword(_t.from); })).first;
	else if (isProxyAccount(_t.from))
		queueTransaction(_t);
	return h256();
}

h256 FixedAccountHolder::authenticate(dev::eth::TransactionSkeleton const& _t)
{
	if (isRealAccount(_t.from))
		return m_client()->submitTransaction(_t, m_accounts[_t.from]).first;
	else if (isProxyAccount(_t.from))
		queueTransaction(_t);
	return h256();
}



