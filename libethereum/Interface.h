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
/** @file Interface.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Guards.h>
#include <libdevcrypto/Common.h>
#include <libevm/FeeStructure.h>
#include "MessageFilter.h"
#include "Transaction.h"
#include "AccountDiff.h"

namespace dev
{
namespace eth
{

/**
 * @brief Main API hub for interfacing with Ethereum.
 * @TODO Mining hooks.
 */
class Interface
{
public:
	/// Constructor.
	Interface() {}

	/// Destructor.
	virtual ~Interface() {}

	// [TRANSACTION API]

	/// Submits the given message-call transaction.
	virtual void transact(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo) = 0;

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	virtual Address transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas = 10000, u256 _gasPrice = 10 * szabo) = 0;

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	virtual void inject(bytesConstRef _rlp) = 0;

	/// Blocks until all pending transactions have been processed.
	virtual void flushTransactions() = 0;

	/// Makes the given call. Nothing is recorded into the state.
	virtual bytes call(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo) = 0;

	// Informational stuff

	// [NEW API]

	int getDefault() const { return m_default; }
	void setDefault(int _block) { m_default = _block; }

	u256 balanceAt(Address _a) const { return balanceAt(_a, m_default); }
	u256 countAt(Address _a) const { return countAt(_a, m_default); }
	u256 stateAt(Address _a, u256 _l) const { return stateAt(_a, _l, m_default); }
	bytes codeAt(Address _a) const { return codeAt(_a, m_default); }
	std::map<u256, u256> storageAt(Address _a) const { return storageAt(_a, m_default); }

	virtual u256 balanceAt(Address _a, int _block) const = 0;
	virtual u256 countAt(Address _a, int _block) const = 0;
	virtual u256 stateAt(Address _a, u256 _l, int _block) const = 0;
	virtual bytes codeAt(Address _a, int _block) const = 0;
	virtual std::map<u256, u256> storageAt(Address _a, int _block) const = 0;

	virtual unsigned installWatch(MessageFilter const& _filter) = 0;
	virtual unsigned installWatch(h256 _filterId) = 0;
	virtual void uninstallWatch(unsigned _watchId) = 0;
	virtual bool peekWatch(unsigned _watchId) const = 0;
	virtual bool checkWatch(unsigned _watchId) = 0;

	virtual PastMessages messages(unsigned _watchId) const = 0;
	virtual PastMessages messages(MessageFilter const& _filter) const = 0;

	// [EXTRA API]:

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	virtual Transactions pending() const = 0;

	/// Differences between transactions.
	StateDiff diff(unsigned _txi) const { return diff(_txi, m_default); }
	virtual StateDiff diff(unsigned _txi, h256 _block) const = 0;
	virtual StateDiff diff(unsigned _txi, int _block) const = 0;

	/// Get a list of all active addresses.
	virtual Addresses addresses() const { return addresses(m_default); }
	virtual Addresses addresses(int _block) const = 0;

	/// Get the fee associated for a transaction with the given data.
	static u256 txGas(unsigned _dataCount, u256 _gas = 0) { return c_txDataGas * _dataCount + c_txGas + _gas; }

	/// Get the remaining gas limit in this block.
	virtual u256 gasLimitRemaining() const = 0;

protected:
	int m_default = -1;
};

}
}
