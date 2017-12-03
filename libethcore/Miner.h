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
#include "EthashAux.h"

#define MINER_WAIT_STATE_WORK	 1


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
	Mixed,
	CL,
	CUDA
};

struct HwMonitor
{
	int tempC = 0;
	int fanP = 0;
};

inline std::ostream& operator<<(std::ostream& os, HwMonitor _hw)
{
	return os << _hw.tempC << "C " << _hw.fanP << "%";
}

/// Describes the progress of a mining operation.
struct WorkingProgress
{
	uint64_t hashes = 0;		///< Total number of hashes computed.
	uint64_t ms = 0;			///< Total number of milliseconds of mining thus far.
	uint64_t rate() const { return ms == 0 ? 0 : hashes * 1000 / ms; }

	std::vector<uint64_t> minersHashes;
	std::vector<HwMonitor> minerMonitors;
	uint64_t minerRate(const uint64_t hashCount) const { return ms == 0 ? 0 : hashCount * 1000 / ms; }
};

inline std::ostream& operator<<(std::ostream& _out, WorkingProgress _p)
{
	float mh = _p.rate() / 1000000.0f;
	_out << "Speed "
		 << EthTealBold << std::fixed << std::setw(6) << std::setprecision(2) << mh << EthReset
		 << " Mh/s    ";

	for (size_t i = 0; i < _p.minersHashes.size(); ++i)
	{
		mh = _p.minerRate(_p.minersHashes[i]) / 1000000.0f;
		_out << "gpu/" << i << " " << EthTeal << std::fixed << std::setw(5) << std::setprecision(2) << mh << EthReset;
		if (_p.minerMonitors.size() == _p.minersHashes.size())
			_out << " " << EthTeal << _p.minerMonitors[i] << EthReset;
		_out << "  ";
	}

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

class Miner;


/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
class FarmFace
{
public:
	virtual ~FarmFace() = default;

	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	virtual bool submitProof(Solution const& _p) = 0;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */
class Miner: public Worker
{
public:
	Miner(std::string const& _name, FarmFace& _farm, size_t _index):
		Worker(_name + std::to_string(_index)),
		index(_index),
		farm(_farm)
	{}

	virtual ~Miner() = default;

	void setWork(WorkPackage const& _work)
	{
		{
			Guard l(x_work);
			m_work = _work;
			workSwitchStart = std::chrono::high_resolution_clock::now();
		}
		pause();
		kickOff();
		m_hashCount = 0;
	}

	uint64_t hashCount() const { return m_hashCount; }

	void resetHashCount() { m_hashCount = 0; }

	virtual HwMonitor hwmon() = 0;

protected:

	/**
	 * @brief Begin working on a given work package, discarding any previous work.
	 * @param _work The package for which to find a solution.
	 */
	virtual void kickOff() = 0;

	/**
	 * @brief No work left to be done. Pause until told to kickOff().
	 */
	virtual void pause() = 0;

	WorkPackage work() const { Guard l(x_work); return m_work; }

	void addHashCount(uint64_t _n) { m_hashCount += _n; }

	static unsigned s_dagLoadMode;
	static volatile unsigned s_dagLoadIndex;
	static unsigned s_dagCreateDevice;
	static volatile void* s_dagInHostMemory;

	const size_t index = 0;
	FarmFace& farm;
	std::chrono::high_resolution_clock::time_point workSwitchStart;

private:
	uint64_t m_hashCount = 0;

	WorkPackage m_work;
	mutable Mutex x_work;
};

}
}
