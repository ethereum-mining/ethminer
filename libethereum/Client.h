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
	inline ClientGuard(Client* _c);
	inline ~ClientGuard();

private:
	Client* m_client;
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
	VersionChecker(std::string const& _dbPath, unsigned _protocolVersion);

	void setOk();
	bool ok() const { return m_ok; }

private:
	bool m_ok;
	std::string m_path;
	unsigned m_protocolVersion;
};

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

	void inject(bytesConstRef _rlp);

	/// Makes the given call. Nothing is recorded into the state. TODO
//	bytes call(Secret _secret, u256 _amount, u256 _gasPrice, Address _dest, u256 _gas, bytes _data = bytes());

	/// Requires transactions involving this address be queued for inspection.
	void setInterest(Address _dest);

	/// @returns incoming minable transactions that we wanted to be notified of. Clears the queue.
	Transactions pendingQueue() { ClientGuard g(this); return m_tq.interestQueue(); }

	/// @returns alterations in state of a mined block that we wanted to be notified of. Clears the queue.
	std::vector<std::pair<Address, AddressState>> minedQueue() { ClientGuard g(this); return m_bc.interestQueue(); }

	// Not yet - probably best as using some sort of signals implementation.
	/// Calls @a _f when a valid transaction is received that involves @a _dest and once per such transaction.
//	void onPending(Address _dest, function<void(Transaction)> const& _f);

	/// Calls @a _f when a transaction is mined that involves @a _dest and once per change.
//	void onConfirmed(Address _dest, function<void(Transaction, AddressState)> const& _f);

	// Informational stuff

	/// Determines whether at least one of the state/blockChain/transactionQueue has changed since the last call to changed().
	bool changed() const { auto ret = m_changed; m_changed = false; return ret; }
	bool peekChanged() const { return m_changed; }

	/// Get a map containing each of the pending transactions.
	Transactions pending() const { return m_postMine.pending(); }

	// [OLD API]:

	/// Locks/unlocks the state/blockChain/transactionQueue for access.
	void lock();
	void unlock();

	/// Get the object representing the current state of Ethereum.
	State const& state() const { return m_preMine; }
	/// Get the object representing the current state of Ethereum.
	State const& postState() const { return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	BlockChain const& blockChain() const { return m_bc; }

	// [NEW API]

	u256 balanceAt(Address _a, int _block = -1) const;
	u256 countAt(Address _a, int _block = -1) const;
	u256 stateAt(Address _a, u256 _l, int _block = -1) const;
	bytes codeAt(Address _a, int _block = -1) const;
	Transactions transactions(Addresses const& _from, Addresses const& _to, std::vector<std::pair<u256, u256>> const& _stateAlterations, Addresses const& _altered, int _blockFrom = 0, int _blockTo = -1, unsigned _max = 10) const;

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
	bool haveNetwork() { return !!m_net; }
	/// Get access to the peer server object. This will be null if the network isn't online.
	PeerServer* peerServer() const { return m_net.get(); }

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

private:
	void work();

	std::string m_clientVersion;		///< Our end-application client's name/version.
	VersionChecker m_vc;				///< Dummy object to check & update the protocol version.
	BlockChain m_bc;					///< Maintains block database.
	TransactionQueue m_tq;				///< Maintains list of incoming transactions not yet on the block chain.
	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;					///< The present state of the client.
	State m_postMine;					///< The state of the client which we're mining (i.e. it'll have all the rewards added).
	std::unique_ptr<PeerServer> m_net;	///< Should run in background and send us events when blocks found and allow us to send blocks as required.
	
	std::unique_ptr<std::thread> m_work;///< The work thread.
	
	std::recursive_mutex m_lock;
	std::atomic<ClientWorkState> m_workState;
	bool m_paranoia = false;
	bool m_doMine = false;				///< Are we supposed to be mining?
	MineProgress m_mineProgress;
	std::list<MineInfo> m_mineHistory;
	mutable bool m_restartMining = false;

	mutable bool m_changed;
};

inline ClientGuard::ClientGuard(Client* _c): m_client(_c)
{
	m_client->lock();
}

inline ClientGuard::~ClientGuard()
{
	m_client->unlock();
}

}
