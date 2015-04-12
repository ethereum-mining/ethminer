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
#include <libethcore/Params.h>
#include <libp2p/Common.h>
#include "CanonBlockChain.h"
#include "TransactionQueue.h"
#include "State.h"
#include "CommonNet.h"
#include "Miner.h"
#include "ABI.h"
#include "Farm.h"
#include "ClientBase.h"

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
	WithExisting action() const { return m_action; }

private:
	WithExisting m_action;
	std::string m_path;
};

class BasicGasPricer: public GasPricer
{
public:
	explicit BasicGasPricer(u256 _weiPerRef, u256 _refsPerBlock): m_weiPerRef(_weiPerRef), m_refsPerBlock(_refsPerBlock) {}

	void setRefPrice(u256 _weiPerRef) { m_weiPerRef = _weiPerRef; }
	void setRefBlockFees(u256 _refsPerBlock) { m_refsPerBlock = _refsPerBlock; }

	u256 ask(State const&) const override { return m_weiPerRef * m_refsPerBlock / m_gasPerBlock; }
	u256 bid(TransactionPriority _p = TransactionPriority::Medium) const override { return m_octiles[(int)_p] > 0 ? m_octiles[(int)_p] : (m_weiPerRef * m_refsPerBlock / m_gasPerBlock); }

	void update(BlockChain const& _bc) override;

private:
	u256 m_weiPerRef;
	u256 m_refsPerBlock;
	u256 m_gasPerBlock = 3141592;
	std::array<u256, 9> m_octiles;
};

struct ClientNote: public LogChannel { static const char* name() { return "*C*"; } static const int verbosity = 2; };
struct ClientChat: public LogChannel { static const char* name() { return "=C="; } static const int verbosity = 4; };
struct ClientTrace: public LogChannel { static const char* name() { return "-C-"; } static const int verbosity = 7; };
struct ClientDetail: public LogChannel { static const char* name() { return " C "; } static const int verbosity = 14; };

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client: public ClientBase, Worker
{
	friend class OldMiner;

public:
	/// New-style Constructor.
	explicit Client(
		p2p::Host* _host,
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0
	);

	explicit Client(
		p2p::Host* _host,
		std::shared_ptr<GasPricer> _gpForAdoption,		// pass it in with new.
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0
	);

	/// Destructor.
	virtual ~Client();

	/// Resets the gas pricer to some other object.
	void setGasPricer(std::shared_ptr<GasPricer> _gp) { m_gp = _gp; }

	/// Injects the RLP-encoded transaction given by the _rlp into the transaction queue directly.
	virtual void inject(bytesConstRef _rlp);

	/// Blocks until all pending transactions have been processed.
	virtual void flushTransactions() override;

	using Interface::call; // to remove warning about hiding virtual function
	/// Makes the given call. Nothing is recorded into the state. This cheats by creating a null address and endowing it with a lot of ETH.
	ExecutionResult call(Address _dest, bytes const& _data = bytes(), u256 _gas = 125000, u256 _value = 0, u256 _gasPrice = 1 * ether, Address const& _from = Address());

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
	/// Get some information on the block queue.
	BlockQueueStatus blockQueueStatus() const { return m_bq.status(); }

	// Mining stuff:

	void setAddress(Address _us) { WriteGuard l(x_stateDB); m_preMine.setAddress(_us); }

	/// Check block validity prior to mining.
	bool miningParanoia() const { return m_paranoia; }
	/// Change whether we check block validity prior to mining.
	void setParanoia(bool _p) { m_paranoia = _p; }
	/// Should we force mining to happen, even without transactions?
	bool forceMining() const { return m_forceMining; }
	/// Enable/disable forcing of mining to happen, even without transactions.
	void setForceMining(bool _enable);
	/// Are we allowed to GPU mine?
	bool turboMining() const { return m_turboMining; }
	/// Enable/disable GPU mining.
	void setTurboMining(bool _enable = true) { m_turboMining = _enable; if (isMining()) startMining(); }

	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	void startMining() override { if (m_turboMining) m_farm.startGPU(); else m_farm.startCPU(); }
	/// Stop mining.
	/// NOT thread-safe
	void stopMining() override { m_farm.stop(); }
	/// Are we mining now?
	bool isMining() const override { return m_farm.isMining(); }
	/// The hashrate...
	uint64_t hashrate() const override;
	/// Check the progress of the mining.
	MiningProgress miningProgress() const override;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	/// Update to the latest transactions and get hash of the current block to be mined minus the
	/// nonce (the 'work hash') and the difficulty to be met.
	virtual ProofOfWork::WorkPackage getWork() override;

	/** @brief Submit the proof for the proof-of-work.
	 * @param _s A valid solution.
	 * @return true if the solution was indeed valid and accepted.
	 */
	virtual bool submitWork(ProofOfWork::Solution const& _proof) override;

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
	/// InterfaceStub methods
	virtual BlockChain& bc() override { return m_bc; }
	virtual BlockChain const& bc() const override { return m_bc; }

	/// Returns the state object for the full block (i.e. the terminal state) for index _h.
	/// Works properly with LatestBlock and PendingBlock.
	using ClientBase::asOf;
	virtual State asOf(h256 const& _block) const override;
	virtual State preMine() const override { ReadGuard l(x_stateDB); return m_preMine; }
	virtual State postMine() const override { ReadGuard l(x_stateDB); return m_postMine; }
	virtual void prepareForTransaction() override;

	/// Collate the changed filters for the bloom filter of the given pending transaction.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewPending(TransactionReceipt const& _receipt, h256Set& io_changed, h256 _sha3);

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

	/// Magically called when the chain has changed. An import route is provided.
	/// Called by either submitWork() or in our main thread through syncBlockQueue().
	void onChainChanged(ImportRoute const& _ir);

	/// Signal handler for when the block queue needs processing.
	void syncBlockQueue();

	/// Signal handler for when the block queue needs processing.
	void syncTransactionQueue();

	/// Magically called when m_tq needs syncing. Be nice and don't block.
	void onTransactionQueueReady() { Guard l(x_fakeSignalSystemState); m_syncTransactionQueue = true; }

	/// Magically called when m_tq needs syncing. Be nice and don't block.
	void onBlockQueueReady() { Guard l(x_fakeSignalSystemState); m_syncBlockQueue = true; }

	void checkWatchGarbage();

	VersionChecker m_vc;					///< Dummy object to check & update the protocol version.
	CanonBlockChain m_bc;					///< Maintains block database.
	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	std::shared_ptr<GasPricer> m_gp;		///< The gas pricer.

	mutable SharedMutex x_stateDB;			///< Lock on the state DB, effectively a lock on m_postMine.
	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	State m_preMine;						///< The present state of the client.
	State m_postMine;						///< The state of the client which we're mining (i.e. it'll have all the rewards added).

	std::weak_ptr<EthereumHost> m_host;		///< Our Ethereum Host. Don't do anything if we can't lock.

	GenericFarm<ProofOfWork> m_farm;		///< Our mining farm.
	mutable Mutex x_remoteMiner;			///< The remote miner lock.
	RemoteMiner m_remoteMiner;				///< The remote miner.

	Handler m_tqReady;
	Handler m_bqReady;

	bool m_turboMining = false;				///< Don't squander all of our time mining actually just sleeping.
	bool m_forceMining = false;				///< Mine even when there are no transactions pending?
	bool m_paranoia = false;				///< Should we be paranoid about our state?

	mutable std::chrono::system_clock::time_point m_lastGarbageCollection;
											///< When did we last both doing GC on the watches?

	// TODO!!!!!! REPLACE WITH A PROPER X-THREAD ASIO SIGNAL SYSTEM (could just be condition variables)
	mutable Mutex x_fakeSignalSystemState;
	bool m_syncTransactionQueue = false;
	bool m_syncBlockQueue = false;
};

}
}
