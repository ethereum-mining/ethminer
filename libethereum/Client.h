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

class RemoteMiner: public Miner
{
public:
	RemoteMiner() {}

	void update(State const& _provisional, BlockChain const& _bc) { m_state = _provisional; m_state.commitToMine(_bc); }

	h256 workHash() const { return m_state.info().headerHash(IncludeNonce::WithoutNonce); }
	u256 const& difficulty() const { return m_state.info().difficulty; }

	bool submitWork(ProofOfWork::Proof const& _result) { return (m_isComplete = m_state.completeMine(_result)); }

	virtual bool isComplete() const override { return m_isComplete; }
	virtual bytes const& blockData() const { return m_state.blockData(); }

	virtual void noteStateChange() override {}

private:
	bool m_isComplete = false;
	State m_state;
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

/**
 * @brief Main API hub for interfacing with Ethereum.
 */
class Client: public MinerHost, public ClientBase, Worker
{
	friend class Miner;

public:
	/// New-style Constructor.
	explicit Client(
		p2p::Host* _host,
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0,
		int _miners = -1
	);

	explicit Client(
		p2p::Host* _host,
		std::shared_ptr<GasPricer> _gpForAdoption,		// pass it in with new.
		std::string const& _dbPath = std::string(),
		WithExisting _forceAction = WithExisting::Trust,
		u256 _networkId = 0,
		int _miners = -1
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
	ExecutionResult call(Address _dest, bytes const& _data = bytes(), u256 _gas = 125000, u256 _value = 0, u256 _gasPrice = 1 * ether);

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

	void setAddress(Address _us) { WriteGuard l(x_stateDB); m_preMine.setAddress(_us); }

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

	/// Stops mining and sets the number of mining threads (0 for automatic).
	virtual void setMiningThreads(unsigned _threads = 0);
	/// Get the effective number of mining threads.
	virtual unsigned miningThreads() const { ReadGuard l(x_localMiners); return m_localMiners.size(); }
	/// Start mining.
	/// NOT thread-safe - call it & stopMining only from a single thread
	virtual void startMining() { startWorking(); { ReadGuard l(x_localMiners); for (auto& m: m_localMiners) m.start(); } }
	/// Stop mining.
	/// NOT thread-safe
	virtual void stopMining() { { ReadGuard l(x_localMiners); for (auto& m: m_localMiners) m.stop(); } }
	/// Are we mining now?
	virtual bool isMining() { { ReadGuard l(x_localMiners); if (!m_localMiners.empty() && m_localMiners[0].isRunning()) return true; } return false; }
	/// Check the progress of the mining.
	virtual MineProgress miningProgress() const;
	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory();

	/// Update to the latest transactions and get hash of the current block to be mined minus the
	/// nonce (the 'work hash') and the difficulty to be met.
	virtual std::pair<h256, u256> getWork() override;
	/// Submit the proof for the proof-of-work.
	virtual bool submitWork(ProofOfWork::Proof const& _proof) override;

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

	/// Overrides for being a mining host.
	virtual void setupState(State& _s);
	virtual bool turbo() const { return m_turboMining; }
	virtual bool force() const { return m_forceMining; }

	VersionChecker m_vc;					///< Dummy object to check & update the protocol version.
	CanonBlockChain m_bc;					///< Maintains block database.
	BlockQueue m_bq;						///< Maintains a list of incoming blocks not yet on the blockchain (to be imported).
	std::shared_ptr<GasPricer> m_gp;		///< The gas pricer.

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
	bool m_verifyOwnBlocks = true;			///< Should be verify blocks that we mined?

	mutable std::chrono::system_clock::time_point m_lastGarbageCollection;
};

}
}
