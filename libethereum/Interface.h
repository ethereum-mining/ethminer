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
#include <libethcore/Ethash.h>
#include <libethereum/GasPricer.h>
#include "LogFilter.h"
#include "Transaction.h"
#include "AccountDiff.h"
#include "BlockDetails.h"

namespace dev
{
namespace eth
{

using TransactionHashes = h256s;
using UncleHashes = h256s;

enum class Reaping
{
	Automatic,
	Manual
};

enum class FudgeFactor
{
	Strict,
	Lenient
};

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

	/// Submits a new transaction.
	/// @returns the transaction's hash.
	virtual std::pair<h256, Address> submitTransaction(TransactionSkeleton const& _t, Secret const& _secret) = 0;

	/// Submits the given message-call transaction.
	void submitTransaction(Secret const& _secret, u256 const& _value, Address const& _dest, bytes const& _data = bytes(), u256 const& _gas = 10000, u256 const& _gasPrice = c_defaultGasPrice, u256 const& _nonce = UndefinedU256);

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	Address submitTransaction(Secret const& _secret, u256 const& _endowment, bytes const& _init, u256 const& _gas = 10000, u256 const& _gasPrice = c_defaultGasPrice, u256 const& _nonce = UndefinedU256);

	/// Blocks until all pending transactions have been processed.
	virtual void flushTransactions() = 0;

	/// Makes the given call. Nothing is recorded into the state.
	virtual ExecutionResult call(Address const& _from, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) = 0;
	ExecutionResult call(Address const& _from, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = c_defaultGasPrice, FudgeFactor _ff = FudgeFactor::Strict) { return call(_from, _value, _dest, _data, _gas, _gasPrice, m_default, _ff); }
	ExecutionResult call(Secret const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) { return call(toAddress(_secret), _value, _dest, _data, _gas, _gasPrice, _blockNumber, _ff); }
	ExecutionResult call(Secret const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, FudgeFactor _ff = FudgeFactor::Strict) { return call(toAddress(_secret), _value, _dest, _data, _gas, _gasPrice, _ff); }

	/// Does the given creation. Nothing is recorded into the state.
	/// @returns the pair of the Address of the created contract together with its code.
	virtual ExecutionResult create(Address const& _from, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) = 0;
	ExecutionResult create(Address const& _from, u256 _value, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = c_defaultGasPrice, FudgeFactor _ff = FudgeFactor::Strict) { return create(_from, _value, _data, _gas, _gasPrice, m_default, _ff); }
	ExecutionResult create(Secret const& _secret, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) { return create(toAddress(_secret), _value, _data, _gas, _gasPrice, _blockNumber, _ff); }
	ExecutionResult create(Secret const& _secret, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, FudgeFactor _ff = FudgeFactor::Strict) { return create(toAddress(_secret), _value, _data, _gas, _gasPrice, _ff); }

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	virtual ImportResult injectTransaction(bytes const& _rlp) = 0;

	/// Injects the RLP-encoded block given by the _rlp into the block queue directly.
	virtual ImportResult injectBlock(bytes const& _block) = 0;

	// [STATE-QUERY API]

	int getDefault() const { return m_default; }
	void setDefault(BlockNumber _block) { m_default = _block; }

	u256 balanceAt(Address _a) const { return balanceAt(_a, m_default); }
	u256 countAt(Address _a) const { return countAt(_a, m_default); }
	u256 stateAt(Address _a, u256 _l) const { return stateAt(_a, _l, m_default); }
	bytes codeAt(Address _a) const { return codeAt(_a, m_default); }
	h256 codeHashAt(Address _a) const { return codeHashAt(_a, m_default); }
	std::unordered_map<u256, u256> storageAt(Address _a) const { return storageAt(_a, m_default); }

	virtual u256 balanceAt(Address _a, BlockNumber _block) const = 0;
	virtual u256 countAt(Address _a, BlockNumber _block) const = 0;
	virtual u256 stateAt(Address _a, u256 _l, BlockNumber _block) const = 0;
	virtual bytes codeAt(Address _a, BlockNumber _block) const = 0;
	virtual h256 codeHashAt(Address _a, BlockNumber _block) const = 0;
	virtual std::unordered_map<u256, u256> storageAt(Address _a, BlockNumber _block) const = 0;

	// [LOGS API]
	
	virtual LocalisedLogEntries logs(unsigned _watchId) const = 0;
	virtual LocalisedLogEntries logs(LogFilter const& _filter) const = 0;

	/// Install, uninstall and query watches.
	virtual unsigned installWatch(LogFilter const& _filter, Reaping _r = Reaping::Automatic) = 0;
	virtual unsigned installWatch(h256 _filterId, Reaping _r = Reaping::Automatic) = 0;
	virtual bool uninstallWatch(unsigned _watchId) = 0;
	LocalisedLogEntries peekWatchSafe(unsigned _watchId) const { try { return peekWatch(_watchId); } catch (...) { return LocalisedLogEntries(); } }
	LocalisedLogEntries checkWatchSafe(unsigned _watchId) { try { return checkWatch(_watchId); } catch (...) { return LocalisedLogEntries(); } }
	virtual LocalisedLogEntries peekWatch(unsigned _watchId) const = 0;
	virtual LocalisedLogEntries checkWatch(unsigned _watchId) = 0;

	// [BLOCK QUERY API]

	virtual bool isKnownTransaction(h256 const& _transactionHash) const = 0;
	virtual bool isKnownTransaction(h256 const& _blockHash, unsigned _i) const = 0;
	virtual Transaction transaction(h256 _transactionHash) const = 0;
	virtual LocalisedTransaction localisedTransaction(h256 const& _transactionHash) const = 0;
	virtual TransactionReceipt transactionReceipt(h256 const& _transactionHash) const = 0;
	virtual LocalisedTransactionReceipt localisedTransactionReceipt(h256 const& _transactionHash) const = 0;
	virtual std::pair<h256, unsigned> transactionLocation(h256 const& _transactionHash) const = 0;
	virtual h256 hashFromNumber(BlockNumber _number) const = 0;
	virtual BlockNumber numberFromHash(h256 _blockHash) const = 0;
	virtual int compareBlockHashes(h256 _h1, h256 _h2) const = 0;

	virtual bool isKnown(BlockNumber _block) const = 0;
	virtual bool isKnown(h256 const& _hash) const = 0;
	virtual BlockInfo blockInfo(h256 _hash) const = 0;
	virtual BlockDetails blockDetails(h256 _hash) const = 0;
	virtual Transaction transaction(h256 _blockHash, unsigned _i) const = 0;
	virtual LocalisedTransaction localisedTransaction(h256 const& _blockHash, unsigned _i) const = 0;
	virtual BlockInfo uncle(h256 _blockHash, unsigned _i) const = 0;
	virtual UncleHashes uncleHashes(h256 _blockHash) const = 0;
	virtual unsigned transactionCount(h256 _blockHash) const = 0;
	virtual unsigned uncleCount(h256 _blockHash) const = 0;
	virtual Transactions transactions(h256 _blockHash) const = 0;
	virtual TransactionHashes transactionHashes(h256 _blockHash) const = 0;

	BlockInfo blockInfo(BlockNumber _block) const { return blockInfo(hashFromNumber(_block)); }
	BlockDetails blockDetails(BlockNumber _block) const { return blockDetails(hashFromNumber(_block)); }
	Transaction transaction(BlockNumber _block, unsigned _i) const { auto p = transactions(_block); return _i < p.size() ? p[_i] : Transaction(); }
	unsigned transactionCount(BlockNumber _block) const { if (_block == PendingBlock) { auto p = pending(); return p.size(); } return transactionCount(hashFromNumber(_block)); }
	Transactions transactions(BlockNumber _block) const { if (_block == PendingBlock) return pending(); return transactions(hashFromNumber(_block)); }
	TransactionHashes transactionHashes(BlockNumber _block) const { if (_block == PendingBlock) return pendingHashes(); return transactionHashes(hashFromNumber(_block)); }
	BlockInfo uncle(BlockNumber _block, unsigned _i) const { return uncle(hashFromNumber(_block), _i); }
	UncleHashes uncleHashes(BlockNumber _block) const { return uncleHashes(hashFromNumber(_block)); }
	unsigned uncleCount(BlockNumber _block) const { return uncleCount(hashFromNumber(_block)); }

	// [EXTRA API]:

	/// @returns The height of the chain.
	virtual unsigned number() const = 0;

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	virtual Transactions pending() const = 0;
	virtual h256s pendingHashes() const = 0;

	/// Differences between transactions.
	StateDiff diff(unsigned _txi) const { return diff(_txi, m_default); }
	virtual StateDiff diff(unsigned _txi, h256 _block) const = 0;
	virtual StateDiff diff(unsigned _txi, BlockNumber _block) const = 0;

	/// Get a list of all active addresses.
	/// NOTE: This only works when compiled with ETH_FATDB; otherwise will throw InterfaceNotSupported.
	virtual Addresses addresses() const { return addresses(m_default); }
	virtual Addresses addresses(BlockNumber _block) const = 0;

	/// Get the remaining gas limit in this block.
	virtual u256 gasLimitRemaining() const = 0;

	// [MINING API]:

	/// Set the coinbase address.
	virtual void setBeneficiary(Address _us) = 0;
	/// Get the coinbase address.
	virtual Address address() const = 0;

	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	virtual void startMining() = 0;
	/// Stop mining.
	/// NOT thread-safe
	virtual void stopMining() = 0;
	/// Are we mining now?
	virtual bool isMining() const = 0;
	/// Would we like to mine now?
	virtual bool wouldMine() const = 0;
	/// Current hash rate.
	virtual uint64_t hashrate() const = 0;

	/// Get hash of the current block to be mined minus the nonce (the 'work hash').
	virtual std::tuple<h256, h256, h256> getEthashWork() { BOOST_THROW_EXCEPTION(InterfaceNotSupported("Interface::getEthashWork")); }
	/// Submit the nonce for the proof-of-work.
	virtual bool submitEthashWork(h256 const&, h64 const&) { BOOST_THROW_EXCEPTION(InterfaceNotSupported("Interface::submitEthashWork")); }

	/// Check the progress of the mining.
	virtual WorkingProgress miningProgress() const = 0;

protected:
	int m_default = PendingBlock;
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
