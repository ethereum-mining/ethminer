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
/** @file AccountHolder.h
 * @authors:
 *   Christian R <c@ethdev.com>
 *   Lefteris Karapetsas <lefteris@ethdev.com>
 * @date 2015
 */

#pragma once

#include <functional>
#include <vector>
#include <map>
#include <libdevcrypto/Common.h>
#include <libethcore/CommonJS.h>

namespace dev
{
namespace eth
{
class Interface;
}

/**
 * Manages real accounts (where we know the secret key) and proxy accounts (where transactions
 * to be sent from these accounts are forwarded to a proxy on the other side).
 */
class AccountHolder
{
public:
	explicit AccountHolder(std::function<eth::Interface*()> const& _client): m_client(_client) {}

	/// Sets or resets the list of real accounts.
	void setAccounts(std::vector<KeyPair> const& _accounts);
	std::vector<Address> const& getRealAccounts() const { return m_accounts; }
	bool isRealAccount(Address const& _account) const { return m_keyPairs.count(_account) > 0; }
	bool isProxyAccount(Address const& _account) const { return m_proxyAccounts.count(_account) > 0; }
	Secret const& secretKey(Address const& _account) const { return m_keyPairs.at(_account).secret(); }
	std::vector<Address> getAllAccounts() const;
	Address const& getDefaultTransactAccount() const;

	int addProxyAccount(Address const& _account);
	bool removeProxyAccount(unsigned _id);
	void queueTransaction(eth::TransactionSkeleton const& _transaction);

	std::vector<eth::TransactionSkeleton> const& getQueuedTransactions(int _id) const;
	void clearQueue(int _id);

private:
	using TransactionQueue = std::vector<eth::TransactionSkeleton>;

	std::map<Address, KeyPair> m_keyPairs;
	std::vector<Address> m_accounts;
	std::map<Address, int> m_proxyAccounts;
	std::map<int, std::pair<Address, TransactionQueue>> m_transactionQueues;
	std::function<eth::Interface*()> m_client;
};

}
