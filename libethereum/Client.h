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
#include <libethential/Common.h>
#include <libethential/CommonIO.h>
#include <libethential/Guards.h>
#include <libevm/FeeStructure.h>
#include <libethcore/Dagger.h>
#include <libethnet/Common.h>
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "CommonNet.h"
#include "PastMessage.h"
#include "MessageFilter.h"
#include "Miner.h"

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
#define cwatch eth::LogOutputStream<eth::WatchChannel, true>()
struct WorkInChannel: public LogChannel { static const char* name() { return ">W>"; } static const int verbosity = 16; };
struct WorkOutChannel: public LogChannel { static const char* name() { return "<W<"; } static const int verbosity = 16; };
struct WorkChannel: public LogChannel { static const char* name() { return "-W-"; } static const int verbosity = 16; };
#define cwork eth::LogOutputStream<eth::WorkChannel, true>()
#define cworkin eth::LogOutputStream<eth::WorkInChannel, true>()
#define cworkout eth::LogOutputStream<eth::WorkOutChannel, true>()

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client: public MinerHost
{
	friend class Miner;

public:
	/// Constructor.
	explicit Client(std::string const& _clientVersion, Address _us = Address(), std::string const& _dbPath = std::string(), bool _forceClean = false);

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

	int getDefault() const { return m_default; }
	void setDefault(int _block) { m_default = _block; }

	u256 balanceAt(Address _a) const { return balanceAt(_a, m_default); }
	u256 countAt(Address _a) const { return countAt(_a, m_default); }
	u256 stateAt(Address _a, u256 _l) const { return stateAt(_a, _l, m_default); }
	bytes codeAt(Address _a) const { return codeAt(_a, m_default); }
	std::map<u256, u256> storageAt(Address _a) const { return storageAt(_a, m_default); }

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

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	Transactions pending() const { return m_postMine.pending(); }

	/// Differences between transactions.
	StateDiff diff(unsigned _txi) const { return diff(_txi, m_default); }
	StateDiff diff(unsigned _txi, h256 _block) const;
	StateDiff diff(unsigned _txi, int _block) const;

	/// Get a list of all active addresses.
	std::vector<Address> addresses() const { return addresses(m_default); }
	std::vector<Address> addresses(int _block) const;

	/// Get the fee associated for a transaction with the given data.
	static u256 txGas(uint _dataCount, u256 _gas = 0) { return c_txDataGas * _dataCount + c_txGas + _gas; }

	/// Get the remaining gas limit in this block.
	u256 gasLimitRemaining() const { return m_postMine.gasLimitRemaining(); }

	// [PRIVATE API - only relevant for base clients, not available in general]

	eth::State state(unsigned _txi, h256 _block) const;
	eth::State state(h256 _block) const;
	eth::State state(unsigned _txi) const;

	/// Get the object representing the current state of Ethereum.
	eth::State postState() const { ReadGuard l(x_stateDB); return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	BlockChain const& blockChain() const { return m_bc; }

	// Misc stuff:

	void setClientVersion(std::string const& _name) { m_clientVersion = _name; }

	// Network stuff:

	/// Get information on the current peer set.
	std::vector<PeerInfo> peers();
	/// Same as peers().size(), but more efficient.
	size_t peerCount() const;
	/// Same as peers().size(), but more efficient.
	void setIdealPeerCount(size_t _n) const;

	/// Start the network subsystem.
	void startNetwork(unsigned short _listenPort = 30303, std::string const& _remoteHost = std::string(), unsigned short _remotePort = 30303, NodeMode _mode = NodeMode::Full, unsigned _peers = 5, std::string const& _publicIP = std::string(), bool _upnp = true, u256 _networkId = 0);
	/// Connect to a particular peer.
	void connect(std::string const& _seedHost, unsigned short _port = 30303);
	/// Stop the network subsystem.
	void stopNetwork();
	/// Is the network subsystem up?
	bool haveNetwork() { ReadGuard l(x_net); return !!m_net; }
	/// Save peers
	bytes savePeers();
	/// Restore peers
	void restorePeers(bytesConstRef _saved);

	// Mining stuff:

	/// Check block validity prior to mining.
	bool miningParanoia() const { return m_paranoia; }
	/// Change whether we check block validity prior to mining.
	void setParanoia(bool _p) { m_paranoia = _p; }
	/// Should we force mining to happen, even without transactions?
	bool forceMining() const { return m_forceMining; }
	/// Enable/disable forcing of mining to happen, even without transactions.
	void setForceMining(bool _enable) { m_forceMining = _enable; }
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
	void startMining() { ensureWorking(); ReadGuard l(x_miners); for (auto& m: m_miners) m.start(); }
	/// Stop mining.
	/// NOT thread-safe
	void stopMining() { ReadGuard l(x_miners); for (auto& m: m_miners) m.stop(); }
	/// Are we mining now?
	bool isMining() { ReadGuard l(x_miners); return m_miners.size() && m_miners[0].isRunning(); }
	/// Check the progress of the mining.
	MineProgress miningProgress() const;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	/// Clears pending transactions. Just for debug use.
	void clearPending();

private:
	/// Ensure the worker thread is running. Needed for blockchain maintenance & mining.
	void ensureWorking();

	/// Do some work. Handles blockchain maintenance and mining.
	/// @param _justQueue If true will only processing the transaction queues.
	void work();

	/// Do some work on the network.
	void workNet();

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

	std::string m_clientVersion;			///< Our end-application client's name/version.
	VersionChecker m_vc;					///< Dummy object to check & update the protocol version.
	BlockChain m_bc;						///< Maintains block database.
	TransactionQueue m_tq;					///< Maintains a list of incoming transactions not yet in a block on the blockchain.
	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	// TODO: remove in favour of copying m_stateDB as required and thread-safing/copying State. Have a good think about what state objects there should be. Probably want 4 (pre, post, mining, user-visible).
	mutable boost::shared_mutex x_stateDB;	///< Lock on the state DB, effectively a lock on m_postMine.
	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;						///< The present state of the client.
	State m_postMine;						///< The state of the client which we're mining (i.e. it'll have all the rewards added).

	std::unique_ptr<std::thread> m_workNet;	///< The network thread.
	std::atomic<ClientWorkState> m_workNetState;
	mutable boost::shared_mutex x_net;		///< Lock for the network existance.
	std::unique_ptr<PeerHost> m_net;		///< Should run in background and send us events when blocks found and allow us to send blocks as required.

	std::unique_ptr<std::thread> m_work;	///< The work thread.
	std::atomic<ClientWorkState> m_workState;

	std::vector<Miner> m_miners;
	mutable boost::shared_mutex x_miners;
	bool m_paranoia = false;				///< Should we be paranoid about our state?
	bool m_turboMining = false;				///< Don't squander all of our time mining actually just sleeping.
	bool m_forceMining = false;				///< Mine even when there are no transactions pending?

	mutable std::mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;

	int m_default = -1;
};

class Watch;

}

namespace std { void swap(eth::Watch& _a, eth::Watch& _b); }

namespace eth
{

class Watch: public boost::noncopyable
{
	friend void std::swap(Watch& _a, Watch& _b);

public:
	Watch() {}
	Watch(Client& _c, h256 _f): m_c(&_c), m_id(_c.installWatch(_f)) {}
	Watch(Client& _c, MessageFilter const& _tf): m_c(&_c), m_id(_c.installWatch(_tf)) {}
	~Watch() { if (m_c) m_c->uninstallWatch(m_id); }

	bool check() { return m_c ? m_c->checkWatch(m_id) : false; }
	bool peek() { return m_c ? m_c->peekWatch(m_id) : false; }
	PastMessages messages() const { return m_c->messages(m_id); }

private:
	Client* m_c;
	unsigned m_id;
};

}

namespace std
{

inline void swap(eth::Watch& _a, eth::Watch& _b)
{
	swap(_a.m_c, _b.m_c);
	swap(_a.m_id, _b.m_id);
}

}
