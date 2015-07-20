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
/** @file ClientBase.h
 * @author Gav Wood <i@gavwood.com>
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#pragma once

#include <chrono>
#include "Interface.h"
#include "LogFilter.h"
#include "TransactionQueue.h"

namespace dev {

namespace eth {

struct InstalledFilter
{
	InstalledFilter(LogFilter const& _f): filter(_f) {}

	LogFilter filter;
	unsigned refCount = 1;
	LocalisedLogEntries changes;
};

static const h256 PendingChangedFilter = u256(0);
static const h256 ChainChangedFilter = u256(1);

static const LogEntry SpecialLogEntry = LogEntry(Address(), h256s(), bytes());
static const LocalisedLogEntry InitialChange(SpecialLogEntry);

struct ClientWatch
{
	ClientWatch(): lastPoll(std::chrono::system_clock::now()) {}
	explicit ClientWatch(h256 _id, Reaping _r): id(_id), lastPoll(_r == Reaping::Automatic ? std::chrono::system_clock::now() : std::chrono::system_clock::time_point::max()) {}

	h256 id;
#if INITIAL_STATE_AS_CHANGES
	LocalisedLogEntries changes = LocalisedLogEntries{ InitialChange };
#else
	LocalisedLogEntries changes;
#endif
	mutable std::chrono::system_clock::time_point lastPoll = std::chrono::system_clock::now();
};

struct WatchChannel: public LogChannel { static const char* name(); static const int verbosity = 7; };
#define cwatch LogOutputStream<WatchChannel, true>()
struct WorkInChannel: public LogChannel { static const char* name(); static const int verbosity = 16; };
struct WorkOutChannel: public LogChannel { static const char* name(); static const int verbosity = 16; };
struct WorkChannel: public LogChannel { static const char* name(); static const int verbosity = 21; };
#define cwork LogOutputStream<WorkChannel, true>()
#define cworkin LogOutputStream<WorkInChannel, true>()
#define cworkout LogOutputStream<WorkOutChannel, true>()

class ClientBase: public Interface
{
public:
	ClientBase() {}
	virtual ~ClientBase() {}

	/// Submits the given transaction.
	/// @returns the new transaction's hash.
	virtual std::pair<h256, Address> submitTransaction(TransactionSkeleton const& _t, Secret const& _secret) override;
	using Interface::submitTransaction;

	/// Makes the given call. Nothing is recorded into the state.
	virtual ExecutionResult call(Address const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) override;
	using Interface::call;

	/// Makes the given create. Nothing is recorded into the state.
	virtual ExecutionResult create(Address const& _secret, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, FudgeFactor _ff = FudgeFactor::Strict) override;
	using Interface::create;

	using Interface::balanceAt;
	using Interface::countAt;
	using Interface::stateAt;
	using Interface::codeAt;
	using Interface::codeHashAt;
	using Interface::storageAt;

	virtual u256 balanceAt(Address _a, BlockNumber _block) const override;
	virtual u256 countAt(Address _a, BlockNumber _block) const override;
	virtual u256 stateAt(Address _a, u256 _l, BlockNumber _block) const override;
	virtual bytes codeAt(Address _a, BlockNumber _block) const override;
	virtual h256 codeHashAt(Address _a, BlockNumber _block) const override;
	virtual std::unordered_map<u256, u256> storageAt(Address _a, BlockNumber _block) const override;

	virtual LocalisedLogEntries logs(unsigned _watchId) const override;
	virtual LocalisedLogEntries logs(LogFilter const& _filter) const override;
	virtual void prependLogsFromBlock(LogFilter const& _filter, h256 const& _blockHash, BlockPolarity _polarity, LocalisedLogEntries& io_logs) const;

	/// Install, uninstall and query watches.
	virtual unsigned installWatch(LogFilter const& _filter, Reaping _r = Reaping::Automatic) override;
	virtual unsigned installWatch(h256 _filterId, Reaping _r = Reaping::Automatic) override;
	virtual bool uninstallWatch(unsigned _watchId) override;
	virtual LocalisedLogEntries peekWatch(unsigned _watchId) const override;
	virtual LocalisedLogEntries checkWatch(unsigned _watchId) override;

	virtual h256 hashFromNumber(BlockNumber _number) const override;
	virtual BlockNumber numberFromHash(h256 _blockHash) const override;
	virtual int compareBlockHashes(h256 _h1, h256 _h2) const override;
	virtual BlockInfo blockInfo(h256 _hash) const override;
	virtual BlockDetails blockDetails(h256 _hash) const override;
	virtual Transaction transaction(h256 _transactionHash) const override;
	virtual LocalisedTransaction localisedTransaction(h256 const& _transactionHash) const override;
	virtual Transaction transaction(h256 _blockHash, unsigned _i) const override;
	virtual LocalisedTransaction localisedTransaction(h256 const& _blockHash, unsigned _i) const override;
	virtual TransactionReceipt transactionReceipt(h256 const& _transactionHash) const override;
	virtual LocalisedTransactionReceipt localisedTransactionReceipt(h256 const& _transactionHash) const override;
	virtual std::pair<h256, unsigned> transactionLocation(h256 const& _transactionHash) const override;
	virtual Transactions transactions(h256 _blockHash) const override;
	virtual TransactionHashes transactionHashes(h256 _blockHash) const override;
	virtual BlockInfo uncle(h256 _blockHash, unsigned _i) const override;
	virtual UncleHashes uncleHashes(h256 _blockHash) const override;
	virtual unsigned transactionCount(h256 _blockHash) const override;
	virtual unsigned uncleCount(h256 _blockHash) const override;
	virtual unsigned number() const override;
	virtual Transactions pending() const override;
	virtual h256s pendingHashes() const override;

	virtual ImportResult injectTransaction(bytes const& _rlp) override { prepareForTransaction(); return m_tq.import(_rlp); }
	virtual ImportResult injectBlock(bytes const& _block) override;

	using Interface::diff;
	virtual StateDiff diff(unsigned _txi, h256 _block) const override;
	virtual StateDiff diff(unsigned _txi, BlockNumber _block) const override;

	using Interface::addresses;
	virtual Addresses addresses(BlockNumber _block) const override;
	virtual u256 gasLimitRemaining() const override;

	/// Get the coinbase address
	virtual Address address() const override;

	virtual bool isKnown(h256 const& _hash) const override;
	virtual bool isKnown(BlockNumber _block) const override;
	virtual bool isKnownTransaction(h256 const& _transactionHash) const override;
	virtual bool isKnownTransaction(h256 const& _blockHash, unsigned _i) const override;

	/// TODO: consider moving it to a separate interface

	virtual void startMining() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::startMining")); }
	virtual void stopMining() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::stopMining")); }
	virtual bool isMining() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::isMining")); }
	virtual bool wouldMine() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::wouldMine")); }
	virtual uint64_t hashrate() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::hashrate")); }
	virtual WorkingProgress miningProgress() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("ClientBase::miningProgress")); }

	State asOf(BlockNumber _h) const;

protected:
	/// The interface that must be implemented in any class deriving this.
	/// {
	virtual BlockChain& bc() = 0;
	virtual BlockChain const& bc() const = 0;
	virtual State asOf(h256 const& _h) const = 0;
	virtual State preMine() const = 0;
	virtual State postMine() const = 0;
	virtual void prepareForTransaction() = 0;
	/// }

	TransactionQueue m_tq;							///< Maintains a list of incoming transactions not yet in a block on the blockchain.

	// filters
	mutable Mutex x_filtersWatches;							///< Our lock.
	std::unordered_map<h256, InstalledFilter> m_filters;	///< The dictionary of filters that are active.
	std::unordered_map<h256, h256s> m_specialFilters = std::unordered_map<h256, std::vector<h256>>{{PendingChangedFilter, {}}, {ChainChangedFilter, {}}};
															///< The dictionary of special filters and their additional data
	std::map<unsigned, ClientWatch> m_watches;				///< Each and every watch - these reference a filter.
};

}}
