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
#include <libethcore/Common.h>
#include <libethcore/Miner.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/ProofOfWork.h>

namespace dev
{

namespace eth
{

/**
 * @brief A collective of Miners.
 * Miners ask for work, then submit proofs
 * @threadsafe
 */
template <class PoW>
class GenericFarm: public GenericFarmFace<PoW>
{
public:
	using WorkPackage = typename PoW::WorkPackage;
	using Solution = typename PoW::Solution;
	using Miner = GenericMiner<PoW>;

	/**
	 * @brief Sets the current mining mission.
	 * @param _bi The block (header) we wish to be mining.
	 */
	void setWork(BlockInfo const& _bi)
	{
		WorkPackage w;
		{
			WriteGuard l(x_work);
			m_header = _bi;
			w = m_work = PoW::package(m_header);
		}

		ReadGuard l2(x_miners);
		for (auto const& m: m_miners)
			m->setWork(m_work);
	}

	/**
	 * @brief (Re)start miners for CPU only.
	 * @returns true if started properly.
	 */
	bool startCPU() { return start<typename PoW::CPUMiner>(); }

	/**
	 * @brief (Re)start miners for GPU only.
	 * @returns true if started properly.
	 */
	bool startGPU() { return start<typename PoW::GPUMiner>(); }

	/**
	 * @brief Stop all mining activities.
	 */
	void stop()
	{
		WriteGuard l(x_miners);
		m_miners.clear();
	}

	bool isMining() const
	{
		ReadGuard l(x_miners);
		return !m_miners.empty();
	}

	/**
	 * @brief Get information on the progress of mining this work package.
	 * @return The progress with mining so far.
	 */
	MiningProgress const& miningProgress() const { ReadGuard l(x_progress); return m_progress; }

	using SolutionFound = std::function<bool(Solution const&)>;

	/**
	 * @brief Provides a valid header based upon that received previously with setWork().
	 * @param _bi The now-valid header.
	 * @return true if the header was good and that the Farm should pause until more work is submitted.
	 */
	void onSolutionFound(SolutionFound const& _handler) { m_onSolutionFound = _handler; }

private:
	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	bool submitProof(Solution const& _s, WorkPackage const& _wp, Miner* _m) override
	{
		if (_wp.headerHash != m_work.headerHash)
			return false;

		if (m_onSolutionFound && m_onSolutionFound(_s))
		{
			ReadGuard l(x_miners);
			for (std::shared_ptr<Miner> const& m: m_miners)
				if (m.get() != _m)
					m->setWork();
			m_work.headerHash = h256();
			return true;
		}
		return false;
	}

	/**
	 * @brief Start a number of miners.
	 */
	template <class MinerType>
	bool start()
	{
		WriteGuard l(x_miners);
		if (!m_miners.empty() && !!std::dynamic_pointer_cast<MinerType>(m_miners[0]))
			return true;
		m_miners.clear();
		m_miners.reserve(MinerType::instances());
		for (unsigned i = 0; i < MinerType::instances(); ++i)
			m_miners.push_back(std::shared_ptr<Miner>(new MinerType(std::make_pair(this, i))));
		return true;
	}

	mutable SharedMutex x_miners;
	std::vector<std::shared_ptr<Miner>> m_miners;

	mutable SharedMutex x_progress;
	MiningProgress m_progress;

	mutable SharedMutex x_work;
	WorkPackage m_work;
	BlockInfo m_header;

	SolutionFound m_onSolutionFound;
};

}
}
