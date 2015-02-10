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
#include "LogFilter.h"
#include "Transaction.h"
#include "AccountDiff.h"
#include "BlockDetails.h"
#include "Miner.h"

namespace dev
{
namespace eth
{

/**
 * @brief Main API hub for interfacing with Ethereum.
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

	// [STATE-QUERY API]

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

	// [LOGS API]
	
	virtual LocalisedLogEntries logs(unsigned _watchId) const = 0;
	virtual LocalisedLogEntries logs(LogFilter const& _filter) const = 0;

	/// Install, uninstall and query watches.
	virtual unsigned installWatch(LogFilter const& _filter) = 0;
	virtual unsigned installWatch(h256 _filterId) = 0;
	virtual void uninstallWatch(unsigned _watchId) = 0;
	virtual LocalisedLogEntries peekWatch(unsigned _watchId) const = 0;
	virtual LocalisedLogEntries checkWatch(unsigned _watchId) = 0;

	// [BLOCK QUERY API]

	virtual h256 hashFromNumber(unsigned _number) const = 0;
	virtual BlockInfo blockInfo(h256 _hash) const = 0;
	virtual BlockDetails blockDetails(h256 _hash) const = 0;
	virtual Transaction transaction(h256 _blockHash, unsigned _i) const = 0;
	virtual BlockInfo uncle(h256 _blockHash, unsigned _i) const = 0;

	// [EXTRA API]:

	/// @returns The height of the chain.
	virtual unsigned number() const = 0;

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
	template <class T> static bigint txGas(T const& _data, u256 _gas = 0) { bigint ret = c_txGas + _gas; for (auto i: _data) ret += i ? c_txDataNonZeroGas : c_txDataZeroGas; return ret; }

	/// Get the remaining gas limit in this block.
	virtual u256 gasLimitRemaining() const = 0;

	// [MINING API]:

	/// Set the coinbase address.
	virtual void setAddress(Address _us) = 0;
	/// Get the coinbase address.
	virtual Address address() const = 0;

	/// Stops mining and sets the number of mining threads (0 for automatic).
	virtual void setMiningThreads(unsigned _threads = 0) = 0;
	/// Get the effective number of mining threads.
	virtual unsigned miningThreads() const = 0;

	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	virtual void startMining() = 0;
	/// Stop mining.
	/// NOT thread-safe
	virtual void stopMining() = 0;
	/// Are we mining now?
	virtual bool isMining() = 0;

	/// Get hash of the current block to be mined minus the nonce (the 'work hash').
	virtual std::pair<h256, u256> getWork() = 0;
	/// Submit the nonce for the proof-of-work.
	virtual bool submitNonce(h256 const&) = 0;

	/// Check the progress of the mining.
	virtual MineProgress miningProgress() const = 0;

protected:
	int m_default = -1;
};

class Watch;

}
}

namespace std { void swap(dev::eth::Watch& _a, dev::eth::Watch& _b); }

namespace dev
{
namespace eth
{

class Watch: public boost::noncopyable
{
	friend void std::swap(Watch& _a, Watch& _b);

public:
	Watch() {}
	Watch(Interface& _c, h256 _f): m_c(&_c), m_id(_c.installWatch(_f)) {}
	Watch(Interface& _c, LogFilter const& _tf): m_c(&_c), m_id(_c.installWatch(_tf)) {}
	~Watch() { if (m_c) m_c->uninstallWatch(m_id); }

	LocalisedLogEntries check() { return m_c ? m_c->checkWatch(m_id) : LocalisedLogEntries(); }
	LocalisedLogEntries peek() { return m_c ? m_c->peekWatch(m_id) : LocalisedLogEntries(); }
	LocalisedLogEntries logs() const { return m_c->logs(m_id); }

private:
	Interface* m_c = nullptr;
	unsigned m_id = 0;
};

}
}

namespace std
{

inline void swap(dev::eth::Watch& _a, dev::eth::Watch& _b)
{
	swap(_a.m_c, _b.m_c);
	swap(_a.m_id, _b.m_id);
}

}
