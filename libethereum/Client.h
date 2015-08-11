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
#include <condition_variable>
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
#include <libethcore/Sealer.h>
#include <libethcore/ABI.h>
#include <libp2p/Common.h>
#include "CanonBlockChain.h"
#include "Block.h"
#include "CommonNet.h"
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

struct ClientNote: public LogChannel { static const char* name(); static const int verbosity = 2; };
struct ClientChat: public LogChannel { static const char* name(); static const int verbosity = 4; };
struct ClientTrace: public LogChannel { static const char* name(); static const int verbosity = 7; };
struct ClientDetail: public LogChannel { static const char* name(); static const int verbosity = 14; };

struct ActivityReport
{
	unsigned ticks = 0;
	std::chrono::system_clock::time_point since = std::chrono::system_clock::now();
};

std::ostream& operator<<(std::ostream& _out, ActivityReport const& _r);

/**
 * @brief Main API hub for interfacing with Ethereum.
 * Not to be used directly - subclass.
 */
class Client: public ClientBase, protected Worker
{
public:
	/// Destructor.
	virtual ~Client();

	/// Resets the gas pricer to some other object.
	void setGasPricer(std::shared_ptr<GasPricer> _gp) { m_gp = _gp; }
	std::shared_ptr<GasPricer> gasPricer() const { return m_gp; }

	/// Blocks until all pending transactions have been processed.
	virtual void flushTransactions() override;

	/// Queues a block for import.
	ImportResult queueBlock(bytes const& _block, bool _isSafe = false);

	using Interface::call; // to remove warning about hiding virtual function
	/// Makes the given call. Nothing is recorded into the state. This cheats by creating a null address and endowing it with a lot of ETH.
	ExecutionResult call(Address _dest, bytes const& _data = bytes(), u256 _gas = 125000, u256 _value = 0, u256 _gasPrice = 1 * ether, Address const& _from = Address());

	/// Get the remaining gas limit in this block.
	virtual u256 gasLimitRemaining() const override { return m_postMine.gasLimitRemaining(); }

	// [PRIVATE API - only relevant for base clients, not available in general]
	/// Get the block.
	dev::eth::Block block(h256 const& _blockHash, PopulationStatistics* o_stats = nullptr) const;
	/// Get the state of the given block part way through execution, immediately before transaction
	/// index @a _txi.
	dev::eth::State state(unsigned _txi, h256 const& _block) const;
	/// Get the state of the currently pending block part way through execution, immediately before
	/// transaction index @a _txi.
	dev::eth::State state(unsigned _txi) const;

	/// Get the object representing the current state of Ethereum.
	dev::eth::Block postState() const { ReadGuard l(x_postMine); return m_postMine; }
	/// Get the object representing the current canonical blockchain.
	BlockChain const& blockChain() const { return bc(); }
	/// Get some information on the block queue.
	BlockQueueStatus blockQueueStatus() const { return m_bq.status(); }
	/// Get some information on the block queue.
	SyncStatus syncStatus() const;
	/// Get the block queue.
	BlockQueue const& blockQueue() const { return m_bq; }
	/// Get the block queue.
	OverlayDB const& stateDB() const { return m_stateDB; }

	/// Freeze worker thread and sync some of the block queue.
	std::tuple<ImportRoute, bool, unsigned> syncQueue(unsigned _max = 1);

	// Mining stuff:

	virtual void setBeneficiary(Address _us) override { WriteGuard l(x_preMine); m_preMine.setBeneficiary(_us); }

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
	void setTurboMining(bool _enable = true);
	/// Enable/disable precomputing of the DAG for next epoch
	void setShouldPrecomputeDAG(bool _precompute);

	/// Check to see if we'd mine on an apparently bad chain.
	bool mineOnBadChain() const { return m_mineOnBadChain; }
	/// Set true if you want to mine even when the canary says you're on the wrong chain.
	void setMineOnBadChain(bool _v) { m_mineOnBadChain = _v; }

	/// @returns true if the canary says that the chain is bad.
	bool isChainBad() const;
	/// @returns true if the canary says that the client should be upgraded.
	bool isUpgradeNeeded() const;

	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	void startMining() override;
	/// Stop mining.
	/// NOT thread-safe
	void stopMining() override { m_wouldMine = false; rejigMining(); }
	/// Are we mining now?
	bool isMining() const override;
	/// Are we mining now?
	bool wouldMine() const override { return m_wouldMine; }
	/// The hashrate...
	uint64_t hashrate() const override;
	/// Check the progress of the mining.
	WorkingProgress miningProgress() const override;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	// Debug stuff:

	DownloadMan const* downloadMan() const;
	bool isSyncing() const;
	bool isMajorSyncing() const;
	/// Sets the network id.
	void setNetworkId(u256 _n);
	/// Clears pending transactions. Just for debug use.
	void clearPending();
	/// Kills the blockchain. Just for debug use.
	void killChain() { reopenChain(WithExisting::Kill); }
	/// Reloads the blockchain. Just for debug use.
	void reopenChain(WithExisting _we = WithExisting::Trust);
	/// Retries all blocks with unknown parents.
	void retryUnknown() { m_bq.retryAllUnknown(); }
	/// Get a report of activity.
	ActivityReport activityReport() { ActivityReport ret; std::swap(m_report, ret); return ret; }
	/// Set a JSONRPC server to which we can report bad blocks.
	void setSentinel(std::string const& _server) { m_sentinel = _server; }
	/// Get the JSONRPC server to which we report bad blocks.
	std::string const& sentinel() const { return m_sentinel; }
	/// Set the extra data that goes into mined blocks.
	void setExtraData(bytes const& _extraData) { m_extraData = _extraData; }
	/// Rewind to a prior head.
	void rewind(unsigned _n) { bc().rewind(_n); }
	/// Rescue the chain.
	void rescue() { bc().rescue(m_stateDB); }
	/// Get the seal engine.
	SealEngineFace* sealEngine() const { return m_sealEngine.get(); }

protected:
	/// New-style Constructor.
	/// Any final derived class's constructor should make sure they call init().
	explicit Client(std::shared_ptr<GasPricer> _gpForAdoption);

	/// Perform critical setup functions.
	/// Must be called in the constructor of the finally derived class.
	void init(p2p::Host* _extNet, std::string const& _dbPath, WithExisting _forceAction, u256 _networkId);

	/// InterfaceStub methods
	virtual BlockChain& bc() override = 0;
	virtual BlockChain const& bc() const override = 0;

	/// Returns the state object for the full block (i.e. the terminal state) for index _h.
	/// Works properly with LatestBlock and PendingBlock.
	using ClientBase::asOf;
	virtual Block asOf(h256 const& _block) const override;
	virtual Block preMine() const override { ReadGuard l(x_preMine); return m_preMine; }
	virtual Block postMine() const override { ReadGuard l(x_postMine); return m_postMine; }
	virtual void prepareForTransaction() override;

	/// Collate the changed filters for the bloom filter of the given pending transaction.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromNewPending(TransactionReceipt const& _receipt, h256Hash& io_changed, h256 _sha3);

	/// Collate the changed filters for the hash of the given block.
	/// Insert any filters that are activated into @a o_changed.
	void appendFromBlock(h256 const& _blockHash, BlockPolarity _polarity, h256Hash& io_changed);

	/// Record that the set of filters @a _filters have changed.
	/// This doesn't actually make any callbacks, but incrememnts some counters in m_watches.
	void noteChanged(h256Hash const& _filters);

	/// Submit
	bool submitSealed(bytes const& _s);

protected:
	/// Called when Worker is starting.
	void startedWorking() override;

	/// Do some work. Handles blockchain maintenance and mining.
	void doWork() override;

	/// Called when Worker is exiting.
	void doneWorking() override;

	/// Called when wouldMine(), turboMining(), isChainBad(), forceMining(), pendingTransactions() have changed.
	void rejigMining();

	/// Called on chain changes
	void onDeadBlocks(h256s const& _blocks, h256Hash& io_changed);

	/// Called on chain changes
	void onNewBlocks(h256s const& _blocks, h256Hash& io_changed);

	/// Called after processing blocks by onChainChanged(_ir)
	void resyncStateFromChain();

	/// Clear working state of transactions
	void resetState();

	/// Magically called when the chain has changed. An import route is provided.
	/// Called by either submitWork() or in our main thread through syncBlockQueue().
	void onChainChanged(ImportRoute const& _ir);

	/// Signal handler for when the block queue needs processing.
	void syncBlockQueue();

	/// Signal handler for when the block queue needs processing.
	void syncTransactionQueue();

	/// Magically called when m_tq needs syncing. Be nice and don't block.
	void onTransactionQueueReady() { m_syncTransactionQueue = true; m_signalled.notify_all(); }

	/// Magically called when m_tq needs syncing. Be nice and don't block.
	void onBlockQueueReady() { m_syncBlockQueue = true; m_signalled.notify_all(); }

	/// Called when the post state has changed (i.e. when more transactions are in it or we're mining on a new block).
	/// This updates m_miningInfo.
	void onPostStateChanged();

	/// Does garbage collection on watches.
	void checkWatchGarbage();

	/// Ticks various system-level objects.
	void tick();

	/// @returns true only if it's worth bothering to prep the mining block.
	bool shouldServeWork() const { return m_bq.items().first == 0 && (isMining() || remoteActive()); }

	/// Called when we have attempted to import a bad block.
	/// @warning May be called from any thread.
	void onBadBlock(Exception& _ex) const;

	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	std::shared_ptr<GasPricer> m_gp;		///< The gas pricer.

	OverlayDB m_stateDB;					///< Acts as the central point for the state database, so multiple States can share it.
	mutable SharedMutex x_preMine;			///< Lock on m_preMine.
	Block m_preMine;						///< The present state of the client.
	mutable SharedMutex x_postMine;			///< Lock on m_postMine.
	Block m_postMine;						///< The state of the client which we're mining (i.e. it'll have all the rewards added).
	mutable SharedMutex x_working;			///< Lock on m_working.
	Block m_working;						///< The state of the client which we're mining (i.e. it'll have all the rewards added), while we're actually working on it.
	BlockInfo m_miningInfo;					///< The header we're attempting to mine on (derived from m_postMine).
	bool remoteActive() const;				///< Is there an active and valid remote worker?
	bool m_remoteWorking = false;			///< Has the remote worker recently been reset?
	std::atomic<bool> m_needStateReset = { false };			///< Need reset working state to premin on next sync
	std::chrono::system_clock::time_point m_lastGetWork;	///< Is there an active and valid remote worker?

	std::weak_ptr<EthereumHost> m_host;		///< Our Ethereum Host. Don't do anything if we can't lock.

	std::shared_ptr<SealEngineFace> m_sealEngine;	///< Our block-sealing engine.

	Handler<> m_tqReady;
	Handler<h256 const&> m_tqReplaced;
	Handler<> m_bqReady;

	bool m_wouldMine = false;				///< True if we /should/ be mining.
	bool m_turboMining = false;				///< Don't squander all of our time mining actually just sleeping.
	bool m_forceMining = false;				///< Mine even when there are no transactions pending?
	bool m_mineOnBadChain = false;			///< Mine even when the canary says it's a bad chain.
	bool m_paranoia = false;				///< Should we be paranoid about our state?

	mutable std::chrono::system_clock::time_point m_lastGarbageCollection;
											///< When did we last both doing GC on the watches?
	mutable std::chrono::system_clock::time_point m_lastTick = std::chrono::system_clock::now();
											///< When did we last tick()?

	unsigned m_syncAmount = 50;				///< Number of blocks to sync in each go.

	ActivityReport m_report;

	std::condition_variable m_signalled;
	Mutex x_signalled;
	std::atomic<bool> m_syncTransactionQueue = {false};
	std::atomic<bool> m_syncBlockQueue = {false};

	std::string m_sentinel;
	bytes m_extraData;
};

template <class Sealer>
class SpecialisedClient: public Client
{
public:
	explicit SpecialisedClient(
		p2p::Host* _host,
		std::shared_ptr<GasPricer> _gpForAdoption,
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0
	):
		SpecialisedClient(_gpForAdoption, _dbPath, _forceAction)
	{
		init(_host, _dbPath, _forceAction, _networkId);
	}

	virtual ~SpecialisedClient() { stopWorking(); }

	/// Get the object representing the current canonical blockchain.
	CanonBlockChain<Sealer> const& blockChain() const { return m_bc; }

protected:
	explicit SpecialisedClient(
		std::shared_ptr<GasPricer> _gpForAdoption,
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust
	):
		Client(_gpForAdoption),
		m_bc(_dbPath, _forceAction, [](unsigned d, unsigned t){ std::cerr << "REVISING BLOCKCHAIN: Processed " << d << " of " << t << "...\r"; })
	{
		m_sealEngine = std::shared_ptr<SealEngineFace>(Ethash::createSealEngine());
		m_sealEngine->onSealGenerated([=](bytes const& header){
			this->submitSealed(header);
		});
	}

	virtual BlockChain& bc() override { return m_bc; }
	virtual BlockChain const& bc() const override { return m_bc; }

private:
	CanonBlockChain<Sealer> m_bc;			///< Maintains block database.
};

class EthashClient: public SpecialisedClient<Ethash>
{
public:
	/// Trivial forwarding constructor.
	explicit EthashClient(
		p2p::Host* _host,
		std::shared_ptr<GasPricer> _gpForAdoption,
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0
	):
		SpecialisedClient<Ethash>(_gpForAdoption, _dbPath, _forceAction)
	{
		init(_host, _dbPath, _forceAction, _networkId);
	}

	/// Update to the latest transactions and get hash of the current block to be mined minus the
	/// nonce (the 'work hash') and the difficulty to be met.
	/// @returns Tuple of hash without seal, seed hash, target boundary.
	virtual std::tuple<h256, h256, h256> getEthashWork() override;

	/** @brief Submit the proof for the proof-of-work.
	 * @param _s A valid solution.
	 * @return true if the solution was indeed valid and accepted.
	 */
	virtual bool submitEthashWork(h256 const& _mixHash, h64 const& _nonce) override;
};

}
}
