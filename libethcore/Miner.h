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

namespace dev
{

namespace eth
{

struct MineInfo
{
	MineInfo() = default;
	MineInfo(bool _completed): completed(_completed) {}
	void combine(MineInfo const& _m) { requirement = std::max(requirement, _m.requirement); best = std::min(best, _m.best); hashes += _m.hashes; completed = completed || _m.completed; }
	double requirement = 0;
	double best = 1e99;
	unsigned hashes = 0;
	bool completed = false;
};

/**
 * @brief Describes the progress of a mining operation.
 */
struct MiningProgress
{
	void combine(MiningProgress const& _m) { requirement = std::max(requirement, _m.requirement); best = std::min(best, _m.best); current = std::max(current, _m.current); hashes += _m.hashes; ms = std::max(ms, _m.ms); }
	double requirement = 0;		///< The PoW requirement - as the second logarithm of the minimum acceptable hash.
	double best = 1e99;			///< The PoW achievement - as the second logarithm of the minimum found hash.
	double current = 0;			///< The most recent PoW achievement - as the second logarithm of the presently found hash.
	unsigned hashes = 0;		///< Total number of hashes computed.
	unsigned ms = 0;			///< Total number of milliseconds of mining thus far.
};

template <class PoW> class Miner;

/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
template <class PoW> class FarmFace
{
public:
	using WorkPackage = typename PoW::WorkPackage;
	using Solution = typename PoW::Solution;
	using Miner = Miner<PoW>;

	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @param _finder The miner that found it.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	virtual bool submitProof(Solution const& _p, WorkPackage const& _wp, Miner* _finder) = 0;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 */
template <class PoW> class Miner
{
public:
	using ConstructionInfo = std::pair<FarmFace<PoW>*, unsigned>;
	using WorkPackage = typename PoW::WorkPackage;
	using Solution = typename PoW::Solution;
	using FarmFace = FarmFace<PoW>;

	Miner(ConstructionInfo const& _ci):
		m_farm(_ci.first),
		m_index(_ci.second)
	{}

	// API FOR THE FARM TO CALL IN WITH

	void setWork(WorkPackage const& _work = WorkPackage())
	{
		Guard l(x_work);
		if (_work.headerHash != h256())
			kickOff(m_work);
		else if (m_work.headerHash == h256() && _work.headerHash != h256())
			pause();
		m_work = _work;
	}

	unsigned index() const { return m_index; }

protected:

	// REQUIRED TO BE REIMPLEMENTED BY A SUBCLASS:

	/**
	 * @brief Begin working on a given work package, discarding any previous work.
	 * @param _work The package for which to find a solution.
	 */
	virtual void kickOff(WorkPackage const& _work) = 0;

	/**
	 * @brief No work left to be done. Pause until told to kickOff().
	 */
	virtual void pause() = 0;

	// AVAILABLE FOR A SUBCLASS TO CALL:

	/**
	 * @brief Notes that the Miner found a solution.
	 * @param _s The solution.
	 * @return true if the solution was correct and that the miner should pause.
	 */
	bool submitProof(Solution const& _s)
	{
		if (m_farm)
		{
			Guard l(x_work);
			return m_farm->submitProof(_s, m_work, this);
		}
		return true;
	}

private:
	FarmFace* m_farm = nullptr;
	unsigned m_index;

	Mutex x_work;
	WorkPackage m_work;
};

}
}
