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
#include <boost/utility.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libevm/FeeStructure.h>
#include <libethcore/Dagger.h>
#include <libp2p/Common.h>
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "CommonNet.h"
#include "PastMessage.h"
#include "MessageFilter.h"
#include "Miner.h"
#include "Interface.h"

namespace dev
{
namespace eth
{

class Client;

enum ClientWorkState
{
	Active = 0,
	Deleting,
	Deleted
};

enum class NodeMode
{
	PeerServer,
	Full
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
	InstalledFilter(MessageFilter const& _f): filter(_f) {}

	MessageFilter filter;
	unsigned refCount = 1;
};

static const h256 PendingChangedFilter = u256(0);
static const h256 ChainChangedFilter = u256(1);

struct ClientWatch
{
	ClientWatch() {}
	explicit ClientWatch(h256 _id): id(_id) {}

	h256 id;
	unsigned changes = 1;
};

struct WatchChannel: public LogChannel { static const char* name() { return "(o)"; } static const int verbosity = 7; };
#define cwatch dev::LogOutputStream<dev::eth::WatchChannel, true>()
struct WorkInChannel: public LogChannel { static const char* name() { return ">W>"; } static const int verbosity = 16; };
struct WorkOutChannel: public LogChannel { static const char* name() { return "<W<"; } static const int verbosity = 16; };
struct WorkChannel: public LogChannel { static const char* name() { return "-W-"; } static const int verbosity = 16; };
#define cwork dev::LogOutputStream<dev::eth::WorkChannel, true>()
#define cworkin dev::LogOutputStream<dev::eth::WorkInChannel, true>()
#define cworkout dev::LogOutputStream<dev::eth::WorkOutChannel, true>()

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client: public MinerHost, public Interface, Worker
{
	friend class Miner;

public:
	/// New-style Constructor.
	explicit Client(p2p::Host* _host, std::string const& _dbPath = std::string(), bool _forceClean = false, u256 _networkId = 0);

	/// Destructor.
	~Client();

	/// Submits the given message-call transaction.
	void transact(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	/// Submits a new contract-creation transaction.
	/// @returns the new contract's address (assuming it all goes through).
	Address transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	void inject(bytesConstRef _rlp);

	/// Blocks until all pending transactions have been processed.
	void flushTransactions();

	/// Makes the given call. Nothing is recorded into the state.
	bytes call(Secret _secret, u256 _value, Address _dest, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * szabo);

	// Informational stuff

	// [NEW API]

	using Interface::balanceAt;
	using Interface::countAt;
	using Interface::stateAt;
	using Interface::codeAt;
	using Interface::storageAt;

	u256 balanceAt(Address _a, int _block) const;
	u256 countAt(Address _a, int _block) const;
	u256 stateAt(Address _a, u256 _l, int _block) const;
	bytes codeAt(Address _a, int _block) const;
	std::map<u256, u256> storageAt(Address _a, int _block) const;

	unsigned installWatch(MessageFilter const& _filter);
	unsigned installWatch(h256 _filterId);
	void uninstallWatch(unsigned _watchId);
	bool peekWatch(unsigned _watchId) const { std::lock_guard<std::mutex> l(m_filterLock); try { return m_watches.at(_watchId).changes != 0; } catch (...) { return false; } }
	bool checkWatch(unsigned _watchId) { std::lock_guard<std::mutex> l(m_filterLock); bool ret = false; try { ret = m_watches.at(_watchId).changes != 0; m_watches.at(_watchId).changes = 0; } catch (...) {} return ret; }

	PastMessages messages(unsigned _watchId) const { try { std::lock_guard<std::mutex> l(m_filterLock); return messages(m_filters.at(m_watches.at(_watchId).id).filter); } catch (...) { return PastMessages(); } }
	PastMessages messages(MessageFilter const& _filter) const;

	// [EXTRA API]:

	/// @returns the length of the chain.
	virtual unsigned number() const { return m_bc.number(); }

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	Transactions pending() const { return m_postMine.pending(); }

	/// Differences between transactions.
	using Interface::diff;
	StateDiff diff(unsigned _txi, h256 _block) const;
	StateDiff diff(unsigned _txi, int _block) const;

	/// Get a list of all active addresses.
	using Interface::addresses;
	std::vector<Address> addresses(int _block) const;

	/// Get the remaining gas limit in this block.
	u256 gasLimitRemaining() const { return m_postMine.gasLimitRemaining(); }

	// [PRIVATE API - only relevant for base clients, not available in general]

	dev::eth::State state(unsigned _txi, h256 _block) const;
	dev::eth::State state(h256 _block) const;
	dev::eth::State state(unsigned _txi) const;

	/// Get the object representing the current state of Ethereum.
	dev::eth::State postState() const { ReadGuard l(x_stateDB); return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	BlockChain const& blockChain() const { return m_bc; }

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
	void setAddress(Address _us) { m_preMine.setAddress(_us); }
	/// Get the coinbase address.
	Address address() const { return m_preMine.address(); }
	/// Stops mining and sets the number of mining threads (0 for automatic).
	void setMiningThreads(unsigned _threads = 0);
	/// Get the effective number of mining threads.
	unsigned miningThreads() const { ReadGuard l(x_miners); return m_miners.size(); }
	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	void startMining() { startWorking(); ReadGuard l(x_miners); for (auto& m: m_miners) m.start(); }
	/// Stop mining.
	/// NOT thread-safe
	void stopMining() { ReadGuard l(x_miners); for (auto& m: m_miners) m.stop(); }
	/// Are we mining now?
	bool isMining() { ReadGuard l(x_miners); return m_miners.size() && m_miners[0].isRunning(); }
	/// Check the progress of the mining.
	MineProgress miningProgress() const;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	// Debug stuff:

	/// Sets the network id.
	void setNetworkId(u256 _n);
	/// Clears pending transactions. Just for debug use.
	void clearPending();
	/// Kills the blockchain. Just for debug use.
	void killChain();

private:
	/// Do some work. Handles blockchain maintenance and mining.
	virtual void doWork();

	virtual void doneWorking();

	/// Overrides for being a mining host.
	virtual void setupState(State& _s);
	virtual bool turbo() const { return m_turboMining; }
	virtual bool force() const { return m_forceMining; }

	/// Collate the changed filters for the bloom filter of the given pending transaction.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewPending(h256 _pendingTransactionBloom, h256Set& o_changed) const;

	/// Collate the changed filters for the hash of the given block.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewBlock(h256 _blockHash, h256Set& o_changed) const;

	/// Record that the set of filters @a _filters have changed.
	/// This doesn't actually make any callbacks, but incrememnts some counters in m_watches.
	void noteChanged(h256Set const& _filters);

	/// Return the actual block number of the block with the given int-number (positive is the same, INT_MIN is genesis block, < 0 is negative age, thus -1 is most recently mined, 0 is pending.
	unsigned numberOf(int _b) const;

	State asOf(int _h) const;
	State asOf(unsigned _h) const;

	VersionChecker m_vc;					///< Dummy object to check & update the protocol version.
	BlockChain m_bc;						///< Maintains block database.
	TransactionQueue m_tq;					///< Maintains a list of incoming transactions not yet in a block on the blockchain.
	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	// TODO: remove in favour of copying m_stateDB as required and thread-safing/copying State. Have a good think about what state objects there should be. Probably want 4 (pre, post, mining, user-visible).
	mutable boost::shared_mutex x_stateDB;	///< Lock on the state DB, effectively a lock on m_postMine.
	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;						///< The present state of the client.
	State m_postMine;						///< The state of the client which we're mining (i.e. it'll have all the rewards added).

	std::weak_ptr<EthereumHost> m_host;	///< Our Ethereum Host. Don't do anything if we can't lock.

	std::vector<Miner> m_miners;
	mutable boost::shared_mutex x_miners;
	bool m_paranoia = false;				///< Should we be paranoid about our state?
	bool m_turboMining = false;				///< Don't squander all of our time mining actually just sleeping.
	bool m_forceMining = false;				///< Mine even when there are no transactions pending?

	mutable std::mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;
};

}
}
