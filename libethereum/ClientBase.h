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
static const LocalisedLogEntry InitialChange(SpecialLogEntry, 0);

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

struct WatchChannel: public LogChannel { static const char* name() { return "(o)"; } static const int verbosity = 7; };
#define cwatch dev::LogOutputStream<dev::eth::WatchChannel, true>()
struct WorkInChannel: public LogChannel { static const char* name() { return ">W>"; } static const int verbosity = 16; };
struct WorkOutChannel: public LogChannel { static const char* name() { return "<W<"; } static const int verbosity = 16; };
struct WorkChannel: public LogChannel { static const char* name() { return "-W-"; } static const int verbosity = 16; };
#define cwork dev::LogOutputStream<dev::eth::WorkChannel, true>()
#define cworkin dev::LogOutputStream<dev::eth::WorkInChannel, true>()
#define cworkout dev::LogOutputStream<dev::eth::WorkOutChannel, true>()

class ClientBase: public dev::eth::Interface
{
public:
	ClientBase() {}
	virtual ~ClientBase() {}

	/// Submits the given message-call transaction.
	virtual void submitTransaction(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo) override;

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	virtual Address submitTransaction(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas = 10000, u256 _gasPrice = 10 * szabo) override;

	/// Makes the given call. Nothing is recorded into the state.
	virtual ExecutionResult call(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo, BlockNumber _blockNumber = PendingBlock) override;

	virtual ExecutionResult create(Secret _secret, u256 _value, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo, BlockNumber _blockNumber = PendingBlock) override;
	
	using Interface::balanceAt;
	using Interface::countAt;
	using Interface::stateAt;
	using Interface::codeAt;
	using Interface::storageAt;

	virtual u256 balanceAt(Address _a, BlockNumber _block) const override;
	virtual u256 countAt(Address _a, BlockNumber _block) const override;
	virtual u256 stateAt(Address _a, u256 _l, BlockNumber _block) const override;
	virtual bytes codeAt(Address _a, BlockNumber _block) const override;
	virtual std::map<u256, u256> storageAt(Address _a, BlockNumber _block) const override;

	virtual LocalisedLogEntries logs(unsigned _watchId) const override;
	virtual LocalisedLogEntries logs(LogFilter const& _filter) const override;

	/// Install, uninstall and query watches.
	virtual unsigned installWatch(LogFilter const& _filter, Reaping _r = Reaping::Automatic) override;
	virtual unsigned installWatch(h256 _filterId, Reaping _r = Reaping::Automatic) override;
	virtual bool uninstallWatch(unsigned _watchId) override;
	virtual LocalisedLogEntries peekWatch(unsigned _watchId) const override;
	virtual LocalisedLogEntries checkWatch(unsigned _watchId) override;

	virtual h256 hashFromNumber(unsigned _number) const override;
	virtual eth::BlockInfo blockInfo(h256 _hash) const override;
	virtual eth::BlockDetails blockDetails(h256 _hash) const override;
	virtual eth::Transaction transaction(h256 _transactionHash) const override;
	virtual eth::Transaction transaction(h256 _blockHash, unsigned _i) const override;
	virtual eth::Transactions transactions(h256 _blockHash) const override;
	virtual eth::TransactionHashes transactionHashes(h256 _blockHash) const override;
	virtual eth::BlockInfo uncle(h256 _blockHash, unsigned _i) const override;
	virtual eth::UncleHashes uncleHashes(h256 _blockHash) const override;
	virtual unsigned transactionCount(h256 _blockHash) const override;
	virtual unsigned uncleCount(h256 _blockHash) const override;
	virtual unsigned number() const override;
	virtual eth::Transactions pending() const override;

	using Interface::diff;
	virtual StateDiff diff(unsigned _txi, h256 _block) const override;
	virtual StateDiff diff(unsigned _txi, BlockNumber _block) const override;

	using Interface::addresses;
	virtual Addresses addresses(BlockNumber _block) const override;
	virtual u256 gasLimitRemaining() const override;

	/// Set the coinbase address
	virtual void setAddress(Address _us) override; 

	/// Get the coinbase address
	virtual Address address() const override;

	/// TODO: consider moving it to a separate interface

	virtual void setMiningThreads(unsigned _threads) override { (void)_threads; BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::setMiningThreads")); }
	virtual unsigned miningThreads() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::miningThreads")); }
	virtual void startMining() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::startMining")); }
	virtual void stopMining() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::stopMining")); }
	virtual bool isMining() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::isMining")); }
	virtual eth::MineProgress miningProgress() const override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::miningProgress")); }
	virtual std::pair<h256, u256> getWork() override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::getWork")); }
	virtual bool submitWork(eth::ProofOfWork::Proof const&) override { BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::ClientBase::submitWork")); }

	State asOf(BlockNumber _h) const;

protected:
	/// The interface that must be implemented in any class deriving this.
	/// {
	virtual BlockChain const& bc() const = 0;
	virtual State asOf(h256 const& _h) const = 0;
	virtual State preMine() const = 0;
	virtual State postMine() const = 0;
	virtual void prepareForTransaction() = 0;
	/// }

	TransactionQueue m_tq;							///< Maintains a list of incoming transactions not yet in a block on the blockchain.

	// filters
	mutable Mutex x_filtersWatches;					///< Our lock.
	std::map<h256, InstalledFilter> m_filters;		///< The dictionary of filters that are active.
	std::map<unsigned, ClientWatch> m_watches;		///< Each and every watch - these reference a filter.
};

}}
