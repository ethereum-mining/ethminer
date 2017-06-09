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
/** @file Farm.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#pragma once

#include <thread>
#include <list>
#include <atomic>
#include <libdevcore/Common.h>
#include <libdevcore/Worker.h>
#include <libethcore/Miner.h>
#include <libethcore/BlockHeader.h>

namespace dev
{

namespace eth
{

/**
 * @brief A collective of Miners.
 * Miners ask for work, then submit proofs
 * @threadsafe
 */
class Farm: public FarmFace
{
public:
	struct SealerDescriptor
	{
		std::function<unsigned()> instances;
		std::function<Miner*(Miner::ConstructionInfo ci)> create;
	};

	~Farm()
	{
		stop();
	}

	/**
	 * @brief Sets the current mining mission.
	 * @param _wp The work package we wish to be mining.
	 */
	void setWork(WorkPackage const& _wp)
	{
		Guard l(x_minerWork);
		if (_wp.headerHash == m_work.headerHash && _wp.startNonce == m_work.startNonce)
			return;
		m_work = _wp;
		for (auto const& m: m_miners)
			m->setWork(m_work);
		resetTimer();
	}

	void setSealers(std::map<std::string, SealerDescriptor> const& _sealers) { m_sealers = _sealers; }

	/**
	 * @brief Start a number of miners.
	 */
	bool start(std::string const& _sealer, bool mixed)
	{
		Guard l(x_minerWork);
		if (!m_miners.empty() && m_lastSealer == _sealer)
			return true;
		if (!m_sealers.count(_sealer))
			return false;

		if (!mixed)
		{
			m_miners.clear();
		}
		auto ins = m_sealers[_sealer].instances();
		unsigned start = 0;
		if (!mixed)
		{
			m_miners.reserve(ins);
		}
		else
		{
			start = m_miners.size();
			ins += start;
			m_miners.reserve(ins);
		}
		for (unsigned i = start; i < ins; ++i)
		{
			m_miners.push_back(std::shared_ptr<Miner>(m_sealers[_sealer].create(std::make_pair(this, i))));
			m_miners.back()->setWork(m_work);
		}
		m_isMining = true;
		m_lastSealer = _sealer;
		resetTimer();
		return true;
	}
	/**
	 * @brief Stop all mining activities.
	 */
	void stop()
	{
		Guard l(x_minerWork);
		m_miners.clear();
		m_work.reset();
		m_isMining = false;
	}
	
	bool isMining() const
	{
		return m_isMining;
	}

	/**
	 * @brief Get information on the progress of mining this work package.
	 * @return The progress with mining so far.
	 */
	WorkingProgress const& miningProgress() const
	{
		WorkingProgress p;
		p.ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_lastStart).count();
		{
			Guard l2(x_minerWork);
			for (auto const& i: m_miners)
				p.hashes += i->hashCount();
		}
		Guard l(x_progress);
		m_progress = p;
		return m_progress;
	}

	/**
	 * @brief Reset the mining progess counter.
	 */
	void resetMiningProgress()
	{
		DEV_GUARDED(x_minerWork)
			for (auto const& i: m_miners)
				i->resetHashCount();
		resetTimer();
	}

	SolutionStats getSolutionStats() {
		return m_solutionStats;
	}

	void failedSolution() {
		m_solutionStats.failed();
	}

	void acceptedSolution(bool _stale) {
		if (!_stale)
		{
			m_solutionStats.accepted();
		}
		else
		{
			m_solutionStats.acceptedStale();
		}
	}

	void rejectedSolution(bool _stale) {
		if (!_stale)
		{
			m_solutionStats.rejected();
		}
		else
		{
			m_solutionStats.rejectedStale();
		}
	}

	using SolutionFound = std::function<bool(Solution const&)>;

	/**
	 * @brief Provides a valid header based upon that received previously with setWork().
	 * @param _bi The now-valid header.
	 * @return true if the header was good and that the Farm should pause until more work is submitted.
	 */
	void onSolutionFound(SolutionFound const& _handler) { m_onSolutionFound = _handler; }

	WorkPackage work() const { Guard l(x_minerWork); return m_work; }

private:
	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	bool submitProof(Solution const& _s, Miner* _m) override
	{
		if (m_onSolutionFound && m_onSolutionFound(_s))
		{
			if (x_minerWork.try_lock())
			{
				for (std::shared_ptr<Miner> const& m: m_miners)
					if (m.get() != _m)
						m->setWork();
				m_work.reset();
				x_minerWork.unlock();
				return true;
			}
		}
		return false;
	}

	void resetTimer()
	{
		m_lastStart = std::chrono::steady_clock::now();
	}

	mutable Mutex x_minerWork;
	std::vector<std::shared_ptr<Miner>> m_miners;
	WorkPackage m_work;

	std::atomic<bool> m_isMining = {false};

	mutable Mutex x_progress;
	mutable WorkingProgress m_progress;
	std::chrono::steady_clock::time_point m_lastStart;

	SolutionFound m_onSolutionFound;

	std::map<std::string, SealerDescriptor> m_sealers;
	std::string m_lastSealer;

	mutable SolutionStats m_solutionStats;

}; 

}
}
