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
#include <libhwmon/wrapnvml.h>
#include <libhwmon/wrapadl.h>
#if defined(__linux)
#include <libhwmon/wrapamdsysfs.h>
#endif

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
	unsigned tstart, tstop;

	struct SealerDescriptor
	{
		std::function<unsigned()> instances;
		std::function<Miner*(FarmFace&, unsigned)> create;
	};

	Farm(boost::asio::io_service & io_service): 
		m_io_strand(io_service),
		m_hashrateTimer(io_service)
	{

		// Init HWMON
		adlh = wrap_adl_create();
#if defined(__linux)
		sysfsh = wrap_amdsysfs_create();
#endif
		nvmlh = wrap_nvml_create();

		// Initialize nonce_scrambler
		shuffle();

	}

	~Farm()
	{
		// Deinit HWMON
		if (adlh)
			wrap_adl_destroy(adlh);
#if defined(__linux)
		if (sysfsh)
			wrap_amdsysfs_destroy(sysfsh);
#endif
		if (nvmlh)
			wrap_nvml_destroy(nvmlh);

		// Stop mining
		stop();
	}

	/**
	* @brief Randomizes the nonce scrambler
	*/
	void shuffle() {

		// Given that all nonces are equally likely to solve the problem
		// we could reasonably always start the nonce search ranges
		// at a fixed place, but that would be boring. Provide a once
		// per run randomized start place, without creating much overhead.
		random_device engine;
		m_nonce_scrambler = uniform_int_distribution<uint64_t>()(engine);

	}

	/**
	 * @brief Sets the current mining mission.
	 * @param _wp The work package we wish to be mining.
	 */
	void setWork(WorkPackage const& _wp)
	{
		// Collect hashrate before miner reset their work
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

		m_isMining.store(true, std::memory_order_relaxed);
		m_lastSealer = _sealer;
		b_lastMixed = mixed;

		// Start hashrate collector
		m_hashrateTimer.expires_from_now(boost::posix_time::milliseconds(1000));
		m_hashrateTimer.async_wait(m_io_strand.wrap(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error)));

		return true;
	}

	/**
	 * @brief Stop all mining activities.
	 */
	void stop()
	{
		// Avoid re-entering if not actually mining.
		// This, in fact, is also called by destructor
		if (isMining()) {

			{
				Guard l(x_minerWork);
				m_miners.clear();
				m_isMining.store(false, std::memory_order_relaxed);
			}

			m_hashrateTimer.cancel();

			m_lastProgresses.clear();

		}

	}

    void collectHashRate()
    {

        auto now = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(x_minerWork);

        WorkingProgress p;
        p.ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastStart).count();
        m_lastStart = now;

        // Collect & Reset
        for (auto const& i : m_miners)
        {
            uint64_t minerHashCount = i->hashCount();
			i->resetHashCount();
            p.hashes += minerHashCount;
            p.minersHashes.push_back(minerHashCount);
        }

        if (p.hashes > 0)
            m_lastProgresses.push_back(p);

        // We smooth the hashrate over the last x seconds
        uint64_t allMs = 0;
        for (auto const& cp : m_lastProgresses)
            allMs += cp.ms;

        if (allMs > m_hashrateSmoothInterval)
            m_lastProgresses.erase(m_lastProgresses.begin());
    }

	void processHashRate(const boost::system::error_code& ec) {

		if (!ec) {

			// Stop mining causes m_hashrateTimer to cancel but
			// io_service cannot guarantee this event is cancelled (it may be too close to deadline)
			// Thus do not process if not mining.
			if (!isMining()) return;

			collectHashRate();

			// Resubmit timer only if actually mining
			m_hashrateTimer.expires_from_now(boost::posix_time::milliseconds(1000));
			m_hashrateTimer.async_wait(m_io_strand.wrap(boost::bind(&Farm::processHashRate, this, boost::asio::placeholders::error)));
		}
	}
	
	/**
	 * @brief Stop all mining activities and Starts them again
	 */
	void restart()
	{
		if (m_onMinerRestart) {
			m_onMinerRestart();
		}
	}

	/**
	* @brief Stop all mining activities and Starts them again (async post)
	*/
	void restart_async()
	{
		m_io_strand.get_io_service().post(m_io_strand.wrap(boost::bind(&Farm::restart, this)));
	}

	bool isMining() const
	{
		return m_isMining.load(std::memory_order_relaxed);
	}

    /**
     * @brief Get information on the progress of mining this work package.
     * @return The progress with mining so far.
     */
    WorkingProgress const& miningProgress(bool hwmon = false, bool power = false) const
    {
        std::lock_guard<std::mutex> lock(x_minerWork);
        WorkingProgress p;
        p.ms = 0;
        p.hashes = 0;
        for (auto const& i : m_miners)
        {
			p.miningIsPaused.push_back(i->is_mining_paused());
			p.minersHashes.push_back(0);
			if (hwmon) {
				HwMonitorInfo hwInfo = i->hwmonInfo();
				HwMonitor hw;
				unsigned int tempC = 0, fanpcnt = 0, powerW = 0;
				if (hwInfo.deviceIndex >= 0) {
					if (hwInfo.deviceType == HwMonitorInfoType::NVIDIA && nvmlh) {
						int typeidx = 0;
						if(hwInfo.indexSource == HwMonitorIndexSource::CUDA){
							typeidx = nvmlh->cuda_nvml_device_id[hwInfo.deviceIndex];
						}
						else if(hwInfo.indexSource == HwMonitorIndexSource::OPENCL){
							typeidx = nvmlh->opencl_nvml_device_id[hwInfo.deviceIndex];
						}
						else{
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_nvml_get_tempC(nvmlh, typeidx, &tempC);
						wrap_nvml_get_fanpcnt(nvmlh, typeidx, &fanpcnt);
						if(power) {
							wrap_nvml_get_power_usage(nvmlh, typeidx, &powerW);
						}
					}
					else if (hwInfo.deviceType == HwMonitorInfoType::AMD && adlh) {
						int typeidx = 0;
						if(hwInfo.indexSource == HwMonitorIndexSource::OPENCL){
							typeidx = adlh->opencl_adl_device_id[hwInfo.deviceIndex];
						}
						else{
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_adl_get_tempC(adlh, typeidx, &tempC);
						wrap_adl_get_fanpcnt(adlh, typeidx, &fanpcnt);
						if(power) {
							wrap_adl_get_power_usage(adlh, typeidx, &powerW);
						}
					}
#if defined(__linux)
					// Overwrite with sysfs data if present
					if (hwInfo.deviceType == HwMonitorInfoType::AMD && sysfsh) {
						int typeidx = 0;
						if(hwInfo.indexSource == HwMonitorIndexSource::OPENCL){
							typeidx = sysfsh->opencl_sysfs_device_id[hwInfo.deviceIndex];
						}
						else{
							//Unknown, don't map
							typeidx = hwInfo.deviceIndex;
						}
						wrap_amdsysfs_get_tempC(sysfsh, typeidx, &tempC);
						wrap_amdsysfs_get_fanpcnt(sysfsh, typeidx, &fanpcnt);
						if(power) {
							wrap_amdsysfs_get_power_usage(sysfsh, typeidx, &powerW);
						}
					}
#endif
				}

				i->update_temperature(tempC);

				hw.tempC = tempC;
				hw.fanP = fanpcnt;
				hw.powerW = powerW/((double)1000.0);
				p.minerMonitors.push_back(hw);
			}
        }

        for (auto const& cp : m_lastProgresses)
        {
            p.ms += cp.ms;
            p.hashes += cp.hashes;
            for (unsigned int i = 0; i < cp.minersHashes.size() && i < p.minersHashes.size(); i++)
            {
                p.minersHashes.at(i) += cp.minersHashes.at(i);
            }
        }

        m_progress = p;
        return m_progress;
    }

	SolutionStats getSolutionStats() {
		return m_solutionStats;
	}

	void failedSolution() override {
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

	using SolutionFound = std::function<void(Solution const&)>;
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

	void set_pool_addresses(string host, unsigned port) {
		stringstream ssPoolAddresses;
		ssPoolAddresses << host << ':' << port;
		m_pool_addresses = ssPoolAddresses.str();
	}

	string get_pool_addresses() {
		return m_pool_addresses;
	}

	uint64_t get_nonce_scrambler() override
	{
		return m_nonce_scrambler;
	}

	void setTStartTStop(unsigned tstart, unsigned tstop)
	{
		m_tstart = tstart;
		m_tstop = tstop;
	}

	unsigned get_tstart() override
	{
		return m_tstart;
	}

	unsigned get_tstop() override
	{
		return m_tstop;
	}

private:
	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	void submitProof(Solution const& _s) override
	{
		assert(m_onSolutionFound);
		m_onSolutionFound(_s);
	}

	mutable Mutex x_minerWork;
	std::vector<std::shared_ptr<Miner>> m_miners;
	WorkPackage m_work;

	std::atomic<bool> m_isMining = {false};

	mutable WorkingProgress m_progress;

	SolutionFound m_onSolutionFound;
	MinerRestart m_onMinerRestart;

	std::map<std::string, SealerDescriptor> m_sealers;
	std::string m_lastSealer;
	bool b_lastMixed = false;

	std::chrono::steady_clock::time_point m_lastStart;
	uint64_t m_hashrateSmoothInterval = 10000;

	boost::asio::io_service::strand m_io_strand;
	boost::asio::deadline_timer m_hashrateTimer;
	std::vector<WorkingProgress> m_lastProgresses;

	mutable SolutionStats m_solutionStats;
	std::chrono::steady_clock::time_point m_farm_launched = std::chrono::steady_clock::now();

    string m_pool_addresses;
	uint64_t m_nonce_scrambler;

	unsigned m_tstart, m_tstop;

	wrap_nvml_handle *nvmlh = NULL;
	wrap_adl_handle *adlh = NULL;
#if defined(__linux)
	wrap_amdsysfs_handle *sysfsh = NULL;
#endif
}; 

}
}
