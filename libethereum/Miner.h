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
#include <libethcore/Dagger.h>
#include <libethcore/OverlayDB.h>
#include "State.h"

namespace eth
{

class BlockChain;
class Client;

struct MineProgress
{
	double requirement;
	double best;
	double current;
	uint hashes;
	uint ms;
};

class MinerHost
{
public:
	virtual void setupState(State& _s) = 0;		///< Reset the given State object to the one that should be being mined.
	virtual void onComplete(State& _s) = 0;		///< Completed the mine!
	virtual bool turbo() const = 0;
	virtual bool force() const = 0;
};

/**
 * @brief Implements Miner.
 * The miner will start a thread when there is work provided by @fn restart().
 * The _progressCb callback is called every ~100ms or when a block is found.
 * @fn completeMine() is to be called once a block is found.
 * If miner is not restarted from _progressCb the thread will terminate.
 * @threadsafe
 * @todo signal from child->parent thread to wait on exit; refactor redundant dagger/miner stats
 */
class Miner
{
public:
	/// Constructor. Starts miner.
	Miner(MinerHost* _host, unsigned _id = 0);

	/// Destructor. Stops miner.
	~Miner() { stop(); }

	/// Start mining.
	void start();

	/// Stop mining.
	void stop();

	/// Restart mining.
	void restart() { m_miningStatus = Preparing; }

	/// @returns if mining
	bool isRunning() { return !!m_work; }

	/// @returns true if mining is complete.
	bool isComplete() const { return m_miningStatus == Mined; }

	/// @returns the internal State object.
	State& state() { return m_mineState; }

	/// Check the progress of the mining.
	MineProgress miningProgress() const { Guard l(x_mineInfo); return m_mineProgress; }

	/// Get and clear the mining history.
	std::list<MineInfo> miningHistory() { Guard l(x_mineInfo); auto ret = m_mineHistory; m_mineHistory.clear(); return ret; }

private:
	/// Do some work on the mining.
	void work();

	MinerHost* m_host;						///< Our host.
	unsigned m_id;							///< Our identity;

	std::mutex x_work;						///< Mutex protecting the creation of the work thread.
	std::unique_ptr<std::thread> m_work;	///< The work thread.
	bool m_stop = false;					///< Stop working?

	enum MiningStatus { Preparing, Mining, Mined };
	MiningStatus m_miningStatus = Preparing;///< TODO: consider mutex/atomic variable.
	State m_mineState;						///< The state on which we are mining, generally equivalent to m_postMine.
	mutable unsigned m_pendingCount = 0;	///< How many pending transactions are there in m_mineState?

	mutable std::mutex x_mineInfo;		///< Lock for the mining progress & history.
	MineProgress m_mineProgress;			///< What's our progress?
	std::list<MineInfo> m_mineHistory;		///< What the history of our mining?
};

}
