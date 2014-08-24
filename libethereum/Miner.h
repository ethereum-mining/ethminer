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
/** @file Miner.h
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <list>
#include <atomic>
#include <libethential/Common.h>
#include <libethcore/CommonEth.h>
#include "State.h"

namespace eth
{

/**
 * @brief Describes the progress of a mining operation.
 */
struct MineProgress
{
	void combine(MineProgress const& _m) { requirement = std::max(requirement, _m.requirement); best = std::min(best, _m.best); current = std::max(current, _m.current); hashes += _m.hashes; ms = std::max(ms, _m.ms); }
	double requirement = 0;	///< The PoW requirement - as the second logarithm of the minimum acceptable hash.
	double best = 1e99;		///< The PoW achievement - as the second logarithm of the minimum found hash.
	double current = 0;		///< The most recent PoW achievement - as the second logarithm of the presently found hash.
	uint hashes = 0;		///< Total number of hashes computed.
	uint ms = 0;			///< Total number of milliseconds of mining thus far.
};

/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
class MinerHost
{
public:
	virtual void setupState(State& _s) = 0;		///< Reset the given State object to the one that should be being mined.
	virtual void onProgressed() {}				///< Called once some progress has been made.
	virtual void onComplete() {}				///< Called once a block is found.
	virtual bool turbo() const = 0;				///< @returns true iff the Miner should mine as fast as possible.
	virtual bool force() const = 0;				///< @returns true iff the Miner should mine regardless of the number of transactions.
};

/**
 * @brief Implements Miner.
 * To begin mining, use start() & stop(). noteStateChange() can be used to reset the mining and set up the
 * State object according to the host. Use isRunning() to determine if the miner has been start()ed.
 * Use isComplete() to determine if the miner has finished mining.
 *
 * blockData() can be used to retrieve the complete block, ready for insertion into the BlockChain.
 *
 * Information on the mining can be queried through miningProgress() and miningHistory().
 * @threadsafe
 * @todo Signal Miner to restart once with condition variables.
 */
class Miner
{
public:
	/// Null constructor.
	Miner(): m_host(nullptr), m_id(0) {}

	/// Constructor.
	Miner(MinerHost* _host, unsigned _id = 0);

	/// Move-constructor.
	Miner(Miner&& _m) { std::swap(m_host, _m.m_host); std::swap(m_id, _m.m_id); }

	/// Move-assignment.
	Miner& operator=(Miner&& _m) { std::swap(m_host, _m.m_host); std::swap(m_id, _m.m_id); return *this; }

	/// Destructor. Stops miner.
	~Miner() { stop(); }

	/// Setup its basics.
	void setup(MinerHost* _host, unsigned _id = 0) { m_host = _host; m_id = _id; }

	/// Start mining.
	void start();

	/// Stop mining.
	void stop();

	/// Call to notify Miner of a state change.
	void noteStateChange() { m_miningStatus = Preparing; }

	/// @returns true iff the mining has been start()ed. It may still not be actually mining, depending on the host's turbo() & force().
	bool isRunning() { return !!m_work; }

	/// @returns true if mining is complete.
	bool isComplete() const { return m_miningStatus == Mined; }

	/// @returns the internal State object.
	bytes const& blockData() { return m_mineState.blockData(); }

	/// Check the progress of the mining.
	MineProgress miningProgress() const { Guard l(x_mineInfo); return m_mineProgress; }

	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory() { Guard l(x_mineInfo); auto ret = m_mineHistory; m_mineHistory.clear(); return ret; }

private:
	/// Do some work on the mining.
	void work();

	MinerHost* m_host = nullptr;			///< Our host.
	unsigned m_id = 0;						///< Our identity.

	std::mutex x_work;						///< Mutex protecting the creation of the work thread.
	std::unique_ptr<std::thread> m_work;	///< The work thread.
	bool m_stop = false;					///< Stop working?

	enum MiningStatus { Preparing, Mining, Mined, Stopping, Stopped };
	MiningStatus m_miningStatus = Preparing;///< TODO: consider mutex/atomic variable.
	State m_mineState;						///< The state on which we are mining, generally equivalent to m_postMine.
	mutable unsigned m_pendingCount = 0;	///< How many pending transactions are there in m_mineState?

	mutable std::mutex x_mineInfo;			///< Lock for the mining progress & history.
	MineProgress m_mineProgress;			///< What's our progress?
	std::list<MineInfo> m_mineHistory;		///< What the history of our mining?
};

}
