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
#include <libethential/Common.h>
#include <libethcore/Dagger.h>
#include "BlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "PeerNetwork.h"

namespace eth
{

struct MineProgress
{
	double requirement;
	double best;
	double current;
	uint hashes;
	uint ms;
};

class Client;

class ClientGuard
{
public:
	inline ClientGuard(Client const* _c);
	inline ~ClientGuard();

private:
	Client const* m_client;
};

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

struct PastMessage
{
	PastMessage(Manifest const& _m, std::vector<unsigned> _path, Address _o): to(_m.to), from(_m.from), value(_m.value), input(_m.input), output(_m.output), path(_path), origin(_o) {}

	PastMessage& polish(h256 _b, u256 _ts, unsigned _n) { block = _b; timestamp = _ts; number = _n; return *this; }

	Address to;					///< The receiving address of the transaction. Address() in the case of a creation.
	Address from;				///< The receiving address of the transaction. Address() in the case of a creation.
	u256 value;					///< The value associated with the call.
	bytes input;				///< The data associated with the message, or the initialiser if it's a creation transaction.
	bytes output;				///< The data returned by the message, or the body code if it's a creation transaction.

	std::vector<unsigned> path;	///< Call path into the block transaction. size() is always > 0. First item is the transaction index in the block.
	Address origin;				///< Originating sender of the transaction.
	h256 block;					///< Block hash.
	u256 timestamp;				///< Block timestamp.
	unsigned number;			///< Block number.
};

typedef std::vector<PastMessage> PastMessages;

class TransactionFilter
{
public:
	TransactionFilter(int _earliest = 0, int _latest = -1, unsigned _max = 10, unsigned _skip = 0): m_earliest(_earliest), m_latest(_latest), m_max(_max), m_skip(_skip) {}

	void fillStream(RLPStream& _s) const;
	h256 sha3() const;

	int earliest() const { return m_earliest; }
	int latest() const { return m_latest; }
	unsigned max() const { return m_max; }
	unsigned skip() const { return m_skip; }
	bool matches(h256 _bloom) const;
	bool matches(State const& _s, unsigned _i) const;
	PastMessages matches(Manifest const& _m, unsigned _i) const;

	TransactionFilter from(Address _a) { m_from.insert(_a); return *this; }
	TransactionFilter to(Address _a) { m_to.insert(_a); return *this; }
	TransactionFilter altered(Address _a, u256 _l) { m_stateAltered.insert(std::make_pair(_a, _l)); return *this; }
	TransactionFilter altered(Address _a) { m_altered.insert(_a); return *this; }
	TransactionFilter withMax(unsigned _m) { m_max = _m; return *this; }
	TransactionFilter withSkip(unsigned _m) { m_skip = _m; return *this; }
	TransactionFilter withEarliest(int _e) { m_earliest = _e; return *this; }
	TransactionFilter withLatest(int _e) { m_latest = _e; return *this; }

private:
	bool matches(Manifest const& _m, std::vector<unsigned> _p, Address _o, PastMessages _limbo, PastMessages& o_ret) const;

	std::set<Address> m_from;
	std::set<Address> m_to;
	std::set<std::pair<Address, u256>> m_stateAltered;
	std::set<Address> m_altered;
	int m_earliest = 0;
	int m_latest = -1;
	unsigned m_max;
	unsigned m_skip;
};

struct InstalledFilter
{
	InstalledFilter(TransactionFilter const& _f): filter(_f) {}

	TransactionFilter filter;
	unsigned refCount = 1;
};

static const h256 NewPendingFilter = u256(0);
static const h256 NewBlockFilter = u256(1);

struct Watch
{
	Watch() {}
	explicit Watch(h256 _id): id(_id) {}

	h256 id;
	unsigned changes = 1;
};

struct WatchChannel: public LogChannel { static const char* name() { return "(o)"; } static const int verbosity = 6; };
#define cwatch eth::LogOutputStream<eth::WatchChannel, true>()
struct WorkInChannel: public LogChannel { static const char* name() { return ">W>"; } static const int verbosity = 5; };
struct WorkOutChannel: public LogChannel { static const char* name() { return "<W<"; } static const int verbosity = 5; };
struct WorkChannel: public LogChannel { static const char* name() { return "-W-"; } static const int verbosity = 5; };
#define cwork eth::LogOutputStream<eth::WorkChannel, true>()
#define cworkin eth::LogOutputStream<eth::WorkInChannel, true>()
#define cworkout eth::LogOutputStream<eth::WorkOutChannel, true>()

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client
{
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

	/// Blocks until all pending transactions have been processed.
	void flushTransactions();

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	void inject(bytesConstRef _rlp);

	/// Makes the given call. Nothing is recorded into the state. TODO
//	bytes call(Secret _secret, u256 _amount, u256 _gasPrice, Address _dest, u256 _gas, bytes _data = bytes());

	// Informational stuff

	// [OLD API]:

	/// Locks/unlocks the state/blockChain/transactionQueue for access.
	void lock() const;
	void unlock() const;

	/// Get the object representing the current state of Ethereum.
	State const& state() const { return m_preMine; }
	/// Get the object representing the current state of Ethereum.
	State const& postState() const { return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	BlockChain const& blockChain() const { return m_bc; }

	// [NEW API]

	void setDefault(int _block) { m_default = _block; }

	u256 balanceAt(Address _a) const { return balanceAt(_a, m_default); }
	u256 countAt(Address _a) const { return countAt(_a, m_default); }
	u256 stateAt(Address _a, u256 _l) const { return stateAt(_a, _l, m_default); }
	bytes codeAt(Address _a) const { return codeAt(_a, m_default); }

	u256 balanceAt(Address _a, int _block) const;
	u256 countAt(Address _a, int _block) const;
	u256 stateAt(Address _a, u256 _l, int _block) const;
	bytes codeAt(Address _a, int _block) const;
	PastMessages transactions(TransactionFilter const& _filter) const;
	PastMessages transactions(unsigned _watchId) const { try { std::lock_guard<std::mutex> l(m_filterLock); return transactions(m_filters.at(m_watches.at(_watchId).id).filter); } catch (...) { return PastMessages(); } }
	unsigned installWatch(TransactionFilter const& _filter);
	unsigned installWatch(h256 _filterId);
	void uninstallWatch(unsigned _watchId);
	bool peekWatch(unsigned _watchId) const { std::lock_guard<std::mutex> l(m_filterLock); try { return m_watches.at(_watchId).changes != 0; } catch (...) { return false; } }
	bool checkWatch(unsigned _watchId) { std::lock_guard<std::mutex> l(m_filterLock); bool ret = false; try { ret = m_watches.at(_watchId).changes != 0; m_watches.at(_watchId).changes = 0; } catch (...) {} return ret; }

	// [EXTRA API]:

	/// Get a map containing each of the pending transactions.
	/// @TODO: Remove in favour of transactions().
	Transactions pending() const { return m_postMine.pending(); }

	/// Get a list of all active addresses.
	std::vector<Address> addresses() const { return addresses(m_default); }
	std::vector<Address> addresses(int _block) const;

	// Misc stuff:

	void setClientVersion(std::string const& _name) { m_clientVersion = _name; }

	// Network stuff:

	/// Get information on the current peer set.
	std::vector<PeerInfo> peers();
	/// Same as peers().size(), but more efficient.
	size_t peerCount() const;

	/// Start the network subsystem.
	void startNetwork(unsigned short _listenPort = 30303, std::string const& _remoteHost = std::string(), unsigned short _remotePort = 30303, NodeMode _mode = NodeMode::Full, unsigned _peers = 5, std::string const& _publicIP = std::string(), bool _upnp = true);
	/// Connect to a particular peer.
	void connect(std::string const& _seedHost, unsigned short _port = 30303);
	/// Stop the network subsystem.
	void stopNetwork();
	/// Is the network subsystem up?
	bool haveNetwork() { Guard l(x_net); return !!m_net; }
	/// Get access to the peer server object. This will be null if the network isn't online. DANGEROUS! DO NOT USE!
	PeerServer* peerServer() const { Guard l(x_net); return m_net.get(); }

	// Mining stuff:

	/// Check block validity prior to mining.
	bool paranoia() const { return m_paranoia; }
	/// Change whether we check block validity prior to mining.
	void setParanoia(bool _p) { m_paranoia = _p; }
	/// Set the coinbase address.
	void setAddress(Address _us) { m_preMine.setAddress(_us); }
	/// Get the coinbase address.
	Address address() const { return m_preMine.address(); }
	/// Start mining.
	void startMining();
	/// Stop mining.
	void stopMining();
	/// Are we mining now?
	bool isMining() { return m_doMine; }
	/// Register a callback for information concerning mining.
	/// This callback will be in an arbitrary thread, blocking progress. JUST COPY THE DATA AND GET OUT.
	/// Check the progress of the mining.
	MineProgress miningProgress() const { return m_mineProgress; }
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory() { auto ret = m_mineHistory; m_mineHistory.clear(); return ret; }

	/// Clears pending transactions. Just for debug use.
	void clearPending();

private:
	/// Ensure the worker thread is running. Needed for networking & mining.
	void ensureWorking();

	/// Do some work. Handles networking and mining.
	/// @param _justQueue If true will only processing the transaction queues.
	void work(bool _justQueue = false);

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

	std::string m_clientVersion;		///< Our end-application client's name/version.
	VersionChecker m_vc;				///< Dummy object to check & update the protocol version.
	BlockChain m_bc;					///< Maintains block database.
	TransactionQueue m_tq;				///< Maintains a list of incoming transactions not yet in a block on the blockchain.
	BlockQueue m_bq;					///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	OverlayDB m_stateDB;				///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;					///< The present state of the client.
	State m_postMine;					///< The state of the client which we're mining (i.e. it'll have all the rewards added).

	mutable std::mutex x_net;			///< Lock for the network.
	std::unique_ptr<PeerServer> m_net;	///< Should run in background and send us events when blocks found and allow us to send blocks as required.
	
	std::unique_ptr<std::thread> m_work;///< The work thread.
	
	mutable std::recursive_mutex m_lock;
	std::atomic<ClientWorkState> m_workState;
	bool m_paranoia = false;
	bool m_doMine = false;				///< Are we supposed to be mining?
	MineProgress m_mineProgress;
	std::list<MineInfo> m_mineHistory;
	mutable bool m_restartMining = false;

	mutable std::mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, Watch> m_watches;

	int m_default = -1;
};

inline ClientGuard::ClientGuard(Client const* _c): m_client(_c)
{
	m_client->lock();
}

inline ClientGuard::~ClientGuard()
{
	m_client->unlock();
}

}
