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

#include <boost/asio.hpp>
#include <boost/bind.hpp>
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
		std::function<Miner*(FarmFace&, unsigned)> create;
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
		//Collect hashrate before miner reset their work
		collectHashRate();

		// Set work to each miner
		Guard l(x_minerWork);
		if (_wp.header == m_work.header && _wp.startNonce == m_work.startNonce)
			return;
		m_work = _wp;
		for (auto const& m: m_miners)
			m->setWork(m_work);
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
			// TODO: Improve miners creation, use unique_ptr.
			m_miners.push_back(std::shared_ptr<Miner>(m_sealers[_sealer].create(*this, i)));

			// Start miners' threads. They should pause waiting for new work
			// package.
			m_miners.back()->startWorking();
		}
		m_isMining = true;
		m_lastSealer = _sealer;
		b_lastMixed = mixed;

		if (!p_hashrateTimer) {
			p_hashrateTimer = new boost::asio::deadline_timer(m_io_service, boost::posix_time::milliseconds(1000));
			p_hashrateTimer->async_wait(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error));
			if (m_serviceThread.joinable()) {
				m_io_service.reset();
			}
			else {
				m_serviceThread = std::thread{ boost::bind(&boost::asio::io_service::run, &m_io_service) };
			}
		}

		return true;
	}

	/**
	 * @brief Stop all mining activities.
	 */
	void stop()
	{
		{
			Guard l(x_minerWork);
			m_miners.clear();
			m_isMining = false;
		}

		m_io_service.stop();
		m_serviceThread.join();

		if (p_hashrateTimer) {
			p_hashrateTimer->cancel();
			p_hashrateTimer = nullptr;
		}
	}

	void collectHashRate()
	{
		WorkingProgress p;
		Guard l2(x_minerWork);
		p.ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_lastStart).count();
		//Collect
		for (auto const& i : m_miners)
		{
			uint64_t minerHashCount = i->hashCount();
			p.hashes += minerHashCount;
			p.minersHashes.push_back(minerHashCount);
		}

		//Reset
		for (auto const& i : m_miners)
		{
			i->resetHashCount();
		}
		m_lastStart = std::chrono::steady_clock::now();

		if (p.hashes > 0) {
			m_lastProgresses.push_back(p);
		}

		// We smooth the hashrate over the last x seconds
		int allMs = 0;
		for (auto const& cp : m_lastProgresses) {
			allMs += cp.ms;
		}
		if (allMs > m_hashrateSmoothInterval) {
			m_lastProgresses.erase(m_lastProgresses.begin());
		}
	}

	void processHashRate(const boost::system::error_code& ec) {

		if (!ec) {
			collectHashRate();
		}

		// Restart timer 	
		p_hashrateTimer->expires_at(p_hashrateTimer->expires_at() + boost::posix_time::milliseconds(1000));
		p_hashrateTimer->async_wait(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error));
	}
	
	/**
	 * @brief Stop all mining activities and Starts them again
	 */
	void restart()
	{
		stop();
		start(m_lastSealer, b_lastMixed);
		
		if (m_onMinerRestart) {
			m_onMinerRestart();
		}
	}
		
	bool isMining() const
	{
		return m_isMining;
	}

	/**
	 * @brief Get information on the progress of mining this work package.
	 * @return The progress with mining so far.
	 */
	WorkingProgress const& miningProgress(bool hwmon = false) const
	{
		WorkingProgress p;
		p.ms = 0;
		p.hashes = 0;
		{
			Guard l2(x_minerWork);
			for (auto const& i : m_miners) {
				p.minersHashes.push_back(0);
				if (hwmon)
					p.minerMonitors.push_back(i->hwmon());
			}
		}

		for (auto const& cp : m_lastProgresses) {
			p.ms += cp.ms;
			p.hashes += cp.hashes;
			for (unsigned int i = 0; i < cp.minersHashes.size(); i++)
			{
				p.minersHashes.at(i) += cp.minersHashes.at(i);
			}
		}

		Guard l(x_progress);
		m_progress = p;
		return m_progress;
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
	using MinerRestart = std::function<void()>;

	/**
	 * @brief Provides a valid header based upon that received previously with setWork().
	 * @param _bi The now-valid header.
	 * @return true if the header was good and that the Farm should pause until more work is submitted.
	 */
	void onSolutionFound(SolutionFound const& _handler) { m_onSolutionFound = _handler; }
	void onMinerRestart(MinerRestart const& _handler) { m_onMinerRestart = _handler; }

	WorkPackage work() const { Guard l(x_minerWork); return m_work; }

	std::chrono::steady_clock::time_point farmLaunched() {
		return m_farm_launched;
	}

	string farmLaunchedFormatted() {
		auto d = std::chrono::steady_clock::now() - m_farm_launched;
		int hsize = 3;
		auto hhh = std::chrono::duration_cast<std::chrono::hours>(d);
		if (hhh.count() < 100) {
			hsize = 2;
		}
		d -= hhh;
		auto mm = std::chrono::duration_cast<std::chrono::minutes>(d);
		std::ostringstream stream;
		stream << "Time: " << std::setfill('0') << std::setw(hsize) << hhh.count() << ':' << std::setfill('0') << std::setw(2) << mm.count();
		return stream.str();
	}

private:
	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	bool submitProof(Solution const& _s) override
	{
		assert(m_onSolutionFound);
		return m_onSolutionFound(_s);
	}

	mutable Mutex x_minerWork;
	std::vector<std::shared_ptr<Miner>> m_miners;
	WorkPackage m_work;

	std::atomic<bool> m_isMining = {false};

	mutable Mutex x_progress;
	mutable WorkingProgress m_progress;

	mutable Mutex x_hwmons;

	SolutionFound m_onSolutionFound;
	MinerRestart m_onMinerRestart;

	std::map<std::string, SealerDescriptor> m_sealers;
	std::string m_lastSealer;
	bool b_lastMixed = false;

	std::chrono::steady_clock::time_point m_lastStart;
	int m_hashrateSmoothInterval = 10000;
	std::thread m_serviceThread;  ///< The IO service thread.
	boost::asio::io_service m_io_service;
	boost::asio::deadline_timer * p_hashrateTimer = nullptr;
	std::vector<WorkingProgress> m_lastProgresses;

	mutable SolutionStats m_solutionStats;
	std::chrono::steady_clock::time_point m_farm_launched = std::chrono::steady_clock::now();
}; 

}
}
