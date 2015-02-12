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
/** @file Client.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <atomic>
#include <string>
#include <array>
#include <boost/utility.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libevm/FeeStructure.h>
#include <libp2p/Common.h>
#include "CanonBlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "CommonNet.h"
#include "LogFilter.h"
#include "Miner.h"
#include "Interface.h"

namespace dev
{
namespace eth
{

class Client;
class DownloadMan;

enum ClientWorkState
{
	Active = 0,
	Deleting,
	Deleted
};

class VersionChecker
{
public:
	VersionChecker(std::string const& _dbPath);

	void setOk();
	bool ok() const { return m_ok; }

private:
	bool m_ok;
	std::string m_path;
};

static const int GenesisBlock = INT_MIN;

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
	explicit ClientWatch(h256 _id): id(_id), lastPoll(std::chrono::system_clock::now()) {}

	h256 id;
	LocalisedLogEntries changes = LocalisedLogEntries{ InitialChange };
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

template <class T> struct ABISerialiser {};
template <unsigned N> struct ABISerialiser<FixedHash<N>> { static bytes serialise(FixedHash<N> const& _t) { static_assert(N <= 32, "Cannot serialise hash > 32 bytes."); static_assert(N > 0, "Cannot serialise zero-length hash."); return bytes(32 - N, 0) + _t.asBytes(); } };
template <> struct ABISerialiser<u256> { static bytes serialise(u256 const& _t) { return h256(_t).asBytes(); } };
template <> struct ABISerialiser<u160> { static bytes serialise(u160 const& _t) { return bytes(12, 0) + h160(_t).asBytes(); } };
template <> struct ABISerialiser<string32> { static bytes serialise(string32 const& _t) { return bytesConstRef((byte const*)_t.data(), 32).toBytes(); } };

inline bytes abiInAux() { return {}; }
template <class T, class ... U> bytes abiInAux(T const& _t, U const& ... _u)
{
	return ABISerialiser<T>::serialise(_t) + abiInAux(_u ...);
}

template <class ... T> bytes abiIn(std::string _id, T const& ... _t)
{
	return sha3(_id).ref().cropped(0, 4).toBytes() + abiInAux(_t ...);
}

template <class T> struct ABIDeserialiser {};
template <unsigned N> struct ABIDeserialiser<FixedHash<N>> { static FixedHash<N> deserialise(bytesConstRef& io_t) { static_assert(N <= 32, "Parameter sizes must be at most 32 bytes."); FixedHash<N> ret; io_t.cropped(32 - N, N).populate(ret.ref()); io_t = io_t.cropped(32); return ret; } };
template <> struct ABIDeserialiser<u256> { static u256 deserialise(bytesConstRef& io_t) { u256 ret = fromBigEndian<u256>(io_t.cropped(0, 32)); io_t = io_t.cropped(32); return ret; } };
template <> struct ABIDeserialiser<u160> { static u160 deserialise(bytesConstRef& io_t) { u160 ret = fromBigEndian<u160>(io_t.cropped(12, 20)); io_t = io_t.cropped(32); return ret; } };
template <> struct ABIDeserialiser<string32> { static string32 deserialise(bytesConstRef& io_t) { string32 ret; io_t.cropped(0, 32).populate(vector_ref<char>(ret.data(), 32)); io_t = io_t.cropped(32); return ret; } };

template <class T> T abiOut(bytes const& _data)
{
	bytesConstRef o(&_data);
	return ABIDeserialiser<T>::deserialise(o);
}

class RemoteMiner: public Miner
{
public:
	RemoteMiner() {}

	void update(State const& _provisional, BlockChain const& _bc) { m_state = _provisional; m_state.commitToMine(_bc); }

	h256 workHash() const { return m_state.info().headerHash(IncludeNonce::WithoutNonce); }
	u256 const& difficulty() const { return m_state.info().difficulty; }

	bool submitWork(h256 const& _nonce) { return (m_isComplete = m_state.completeMine(_nonce)); }

	virtual bool isComplete() const override { return m_isComplete; }
	virtual bytes const& blockData() const { return m_state.blockData(); }

	virtual void noteStateChange() override {}

private:
	bool m_isComplete = false;
	State m_state;
};

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client: public MinerHost, public Interface, Worker
{
	friend class Miner;

public:
	/// New-style Constructor.
	explicit Client(p2p::Host* _host, std::string const& _dbPath = std::string(), bool _forceClean = false, u256 _networkId = 0, int miners = -1);

	/// Destructor.
	virtual ~Client();

	/// Submits the given message-call transaction.
	virtual void transact(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	virtual Address transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	virtual void inject(bytesConstRef _rlp);

	/// Blocks until all pending transactions have been processed.
	virtual void flushTransactions();

	/// Makes the given call. Nothing is recorded into the state.
	virtual bytes call(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	/// Makes the given call. Nothing is recorded into the state. This cheats by creating a null address and endowing it with a lot of ETH.
	virtual bytes call(Address _dest, bytes const& _data = bytes(), u256 _gas = 125000, u256 _value = 0, u256 _gasPrice = 1 * ether);

	// Informational stuff

	// [NEW API]

	using Interface::balanceAt;
	using Interface::countAt;
	using Interface::stateAt;
	using Interface::codeAt;
	using Interface::storageAt;

	virtual u256 balanceAt(Address _a, int _block) const;
	virtual u256 countAt(Address _a, int _block) const;
	virtual u256 stateAt(Address _a, u256 _l, int _block) const;
	virtual bytes codeAt(Address _a, int _block) const;
	virtual std::map<u256, u256> storageAt(Address _a, int _block) const;

	virtual unsigned installWatch(LogFilter const& _filter);
	virtual unsigned installWatch(h256 _filterId);
	virtual void uninstallWatch(unsigned _watchId);
	virtual LocalisedLogEntries peekWatch(unsigned _watchId) const;
	virtual LocalisedLogEntries checkWatch(unsigned _watchId);

	virtual LocalisedLogEntries logs(unsigned _watchId) const { try { Guard l(m_filterLock); return logs(m_filters.at(m_watches.at(_watchId).id).filter); } catch (...) { return LocalisedLogEntries(); } }
	virtual LocalisedLogEntries logs(LogFilter const& _filter) const;

	// [EXTRA API]:

	/// @returns the length of the chain.
	virtual unsigned number() const { return m_bc.number(); }

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	virtual Transactions pending() const { return m_postMine.pending(); }

	virtual h256 hashFromNumber(unsigned _number) const { return m_bc.numberHash(_number); }
	virtual BlockInfo blockInfo(h256 _hash) const { return BlockInfo(m_bc.block(_hash)); }
	virtual BlockDetails blockDetails(h256 _hash) const { return m_bc.details(_hash); }
	virtual Transaction transaction(h256 _blockHash, unsigned _i) const;
	virtual BlockInfo uncle(h256 _blockHash, unsigned _i) const;

	/// Differences between transactions.
	using Interface::diff;
	virtual StateDiff diff(unsigned _txi, h256 _block) const;
	virtual StateDiff diff(unsigned _txi, int _block) const;

	/// Get a list of all active addresses.
	using Interface::addresses;
	virtual std::vector<Address> addresses(int _block) const;

	/// Get the remaining gas limit in this block.
	virtual u256 gasLimitRemaining() const { return m_postMine.gasLimitRemaining(); }

	// [PRIVATE API - only relevant for base clients, not available in general]

	dev::eth::State state(unsigned _txi, h256 _block) const;
	dev::eth::State state(h256 _block) const;
	dev::eth::State state(unsigned _txi) const;

	/// Get the object representing the current state of Ethereum.
	dev::eth::State postState() const { ReadGuard l(x_stateDB); return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	CanonBlockChain const& blockChain() const { return m_bc; }

	// Mining stuff:

	/// Check block validity prior to mining.
	bool miningParanoia() const { return m_paranoia; }
	/// Change whether we check block validity prior to mining.
	void setParanoia(bool _p) { m_paranoia = _p; }
	/// Should we force mining to happen, even without transactions?
	bool forceMining() const { return m_forceMining; }
	/// Enable/disable forcing of mining to happen, even without transactions.
	void setForceMining(bool _enable);
	/// Are we mining as fast as we can?
	bool turboMining() const { return m_turboMining; }
	/// Enable/disable fast mining.
	void setTurboMining(bool _enable = true) { m_turboMining = _enable; }

	/// Set the coinbase address.
	virtual void setAddress(Address _us) { m_preMine.setAddress(_us); }
	/// Get the coinbase address.
	virtual Address address() const { return m_preMine.address(); }
	/// Stops mining and sets the number of mining threads (0 for automatic).
	virtual void setMiningThreads(unsigned _threads = 0);
	/// Get the effective number of mining threads.
	virtual unsigned miningThreads() const { ReadGuard l(x_localMiners); return m_localMiners.size(); }
	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	virtual void startMining() { startWorking(); ReadGuard l(x_localMiners); for (auto& m: m_localMiners) m.start(); }
	/// Stop mining.
	/// NOT thread-safe
	virtual void stopMining() { ReadGuard l(x_localMiners); for (auto& m: m_localMiners) m.stop(); }
	/// Are we mining now?
	virtual bool isMining() { ReadGuard l(x_localMiners); return m_localMiners.size() && m_localMiners[0].isRunning(); }
	/// Check the progress of the mining.
	virtual MineProgress miningProgress() const;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	/// Update to the latest transactions and get hash of the current block to be mined minus the
	/// nonce (the 'work hash') and the difficulty to be met.
	virtual std::pair<h256, u256> getWork() override;
	/// Submit the nonce for the proof-of-work.
	virtual bool submitNonce(h256  const&_nonce) override;

	// Debug stuff:

	DownloadMan const* downloadMan() const;
	bool isSyncing() const;
	/// Sets the network id.
	void setNetworkId(u256 _n);
	/// Clears pending transactions. Just for debug use.
	void clearPending();
	/// Kills the blockchain. Just for debug use.
	void killChain();

protected:
	/// Collate the changed filters for the bloom filter of the given pending transaction.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewPending(TransactionReceipt const& _receipt, h256Set& io_changed);

	/// Collate the changed filters for the hash of the given block.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewBlock(h256 const& _blockHash, h256Set& io_changed);

	/// Record that the set of filters @a _filters have changed.
	/// This doesn't actually make any callbacks, but incrememnts some counters in m_watches.
	void noteChanged(h256Set const& _filters);

private:
	/// Do some work. Handles blockchain maintenance and mining.
	virtual void doWork();

	/// Called when Worker is exiting.
	virtual void doneWorking();

	/// Overrides for being a mining host.
	virtual void setupState(State& _s);
	virtual bool turbo() const { return m_turboMining; }
	virtual bool force() const { return m_forceMining; }

	/// Return the actual block number of the block with the given int-number (positive is the same, INT_MIN is genesis block, < 0 is negative age, thus -1 is most recently mined, 0 is pending.
	unsigned numberOf(int _b) const;

	State asOf(int _h) const;
	State asOf(unsigned _h) const;

	VersionChecker m_vc;					///< Dummy object to check & update the protocol version.
	CanonBlockChain m_bc;					///< Maintains block database.
	TransactionQueue m_tq;					///< Maintains a list of incoming transactions not yet in a block on the blockchain.
	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).

	mutable SharedMutex x_stateDB;			///< Lock on the state DB, effectively a lock on m_postMine.
	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;						///< The present state of the client.
	State m_postMine;						///< The state of the client which we're mining (i.e. it'll have all the rewards added).

	std::weak_ptr<EthereumHost> m_host;		///< Our Ethereum Host. Don't do anything if we can't lock.

	mutable Mutex x_remoteMiner;			///< The remote miner lock.
	RemoteMiner m_remoteMiner;				///< The remote miner.

	std::vector<LocalMiner> m_localMiners;	///< The in-process miners.
	mutable SharedMutex x_localMiners;		///< The in-process miners lock.
	bool m_paranoia = false;				///< Should we be paranoid about our state?
	bool m_turboMining = false;				///< Don't squander all of our time mining actually just sleeping.
	bool m_forceMining = false;				///< Mine even when there are no transactions pending?

	mutable Mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;

	mutable std::chrono::system_clock::time_point m_lastGarbageCollection;
};

}
}
