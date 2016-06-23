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
#include <string>
#include <boost/timer.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>
#include <libethcore/Common.h>

#define MINER_WAIT_STATE_UNKNOWN 0
#define MINER_WAIT_STATE_WORK	 1
#define MINER_WAIT_STATE_DAG	 2


#define DAG_LOAD_MODE_PARALLEL	 0
#define DAG_LOAD_MODE_SEQUENTIAL 1
#define DAG_LOAD_MODE_SINGLE	 2

#define STRATUM_PROTOCOL_STRATUM		 0
#define STRATUM_PROTOCOL_ETHPROXY		 1
#define STRATUM_PROTOCOL_ETHEREUMSTRATUM 2

using namespace std;

typedef struct {
	string host;
	string port;
	string user;
	string pass;
} cred_t;

namespace dev
{

namespace eth
{

enum class MinerType
{
	CPU,
	CL,
	CUDA,
	Mixed
};

struct MineInfo: public WorkingProgress {};

inline std::ostream& operator<<(std::ostream& _out, WorkingProgress _p)
{
	float mh = _p.rate() / 1000000.0f;
	char mhs[16];
	sprintf(mhs, "%.2f", mh);
	_out << std::string(mhs) + "MH/s";
	return _out;
}

class SolutionStats {
public:
	void accepted() { accepts++;  }
	void rejected() { rejects++;  }
	void failed()   { failures++; }

	void acceptedStale() { acceptedStales++; }
	void rejectedStale() { rejectedStales++; }


	void reset() { accepts = rejects = failures = acceptedStales = rejectedStales = 0; }

	unsigned getAccepts()			{ return accepts; }
	unsigned getRejects()			{ return rejects; }
	unsigned getFailures()			{ return failures; }
	unsigned getAcceptedStales()	{ return acceptedStales; }
	unsigned getRejectedStales()	{ return rejectedStales; }
private:
	unsigned accepts  = 0;
	unsigned rejects  = 0;
	unsigned failures = 0; 

	unsigned acceptedStales = 0;
	unsigned rejectedStales = 0;
};

inline std::ostream& operator<<(std::ostream& os, SolutionStats s)
{
	return os << "[A" << s.getAccepts() << "+" << s.getAcceptedStales() << ":R" << s.getRejects() << "+" << s.getRejectedStales() << ":F" << s.getFailures() << "]";
}

template <class PoW> class GenericMiner;


/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
template <class PoW> class GenericFarmFace
{
public:
	using WorkPackage = typename PoW::WorkPackage;
	using Solution = typename PoW::Solution;
	using Miner = GenericMiner<PoW>;

	virtual ~GenericFarmFace() {}

	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for; this will be reset if the work is accepted.
	 * @param _finder The miner that found it.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	virtual bool submitProof(Solution const& _p, Miner* _finder) = 0;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */
template <class PoW> class GenericMiner
{
public:
	using WorkPackage = typename PoW::WorkPackage;
	using Solution = typename PoW::Solution;
	using FarmFace = GenericFarmFace<PoW>;
	using ConstructionInfo = std::pair<FarmFace*, unsigned>;

	GenericMiner(ConstructionInfo const& _ci):
		m_farm(_ci.first),
		m_index(_ci.second)
	{}
	virtual ~GenericMiner() {}

	// API FOR THE FARM TO CALL IN WITH

	void setWork(WorkPackage const& _work = WorkPackage())
	{
		auto old = m_work;
		{
			Guard l(x_work);
			m_work = _work;
		}
		if (!!_work)
		{
			DEV_TIMED_ABOVE("pause", 250)
				pause();
			DEV_TIMED_ABOVE("kickOff", 250)
				kickOff();
		}
		else if (!_work && !!old)
			pause();
		m_hashCount = 0;
	}

	uint64_t hashCount() const { return m_hashCount; }

	void resetHashCount() { m_hashCount = 0; }

	unsigned index() const { return m_index; }

protected:


	// REQUIRED TO BE REIMPLEMENTED BY A SUBCLASS:

	/**
	 * @brief Begin working on a given work package, discarding any previous work.
	 * @param _work The package for which to find a solution.
	 */
	virtual void kickOff() = 0;

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
		if (!m_farm)
			return true;
		if (m_farm->submitProof(_s, this))
		{
			Guard l(x_work);
			m_work.reset();
			return true;
		}
		return false;
	}

	WorkPackage const& work() const { Guard l(x_work); return m_work; }

	void accumulateHashes(unsigned _n) { m_hashCount += _n; }

	static unsigned s_dagLoadMode;
	static volatile unsigned s_dagLoadIndex;
	static unsigned s_dagCreateDevice;
	static volatile void* s_dagInHostMemory;
private:
	FarmFace* m_farm = nullptr;
	unsigned m_index;

	uint64_t m_hashCount = 0;

	WorkPackage m_work;
	mutable Mutex x_work;

	
	bool m_dagLoaded = false;
};

}
}
