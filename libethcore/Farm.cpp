/*
 This file is part of ethminer.

 ethminer is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 ethminer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <libethcore/Farm.h>

namespace dev
{
namespace eth
{
Farm* Farm::m_this = nullptr;

Farm::Farm(unsigned hwmonlvl, bool noeval) : m_io_strand(g_io_service), m_collectTimer(g_io_service)
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::Farm() begin");

    m_this = this;
    m_hwmonlvl = hwmonlvl;
    m_noeval = noeval;

    // Init HWMON if needed
    if (m_hwmonlvl)
    {
        adlh = wrap_adl_create();
#if defined(__linux)
        sysfsh = wrap_amdsysfs_create();
#endif
        nvmlh = wrap_nvml_create();
    }

    // Initialize nonce_scrambler
    shuffle();

    // Start data collector timer
    // It should work for the whole lifetime of Farm
    // regardless it's mining state
    m_collectTimer.expires_from_now(boost::posix_time::milliseconds(m_collectInterval));
    m_collectTimer.async_wait(
        m_io_strand.wrap(boost::bind(&Farm::collectData, this, boost::asio::placeholders::error)));

    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::Farm() end");
}

Farm::~Farm()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::~Farm() begin");
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

    // Stop data collector
    m_collectTimer.cancel();
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::~Farm() end");
}

/**
 * @brief Randomizes the nonce scrambler
 */
void Farm::shuffle()
{
    // Given that all nonces are equally likely to solve the problem
    // we could reasonably always start the nonce search ranges
    // at a fixed place, but that would be boring. Provide a once
    // per run randomized start place, without creating much overhead.
    random_device engine;
    m_nonce_scrambler = uniform_int_distribution<uint64_t>()(engine);
}

void Farm::setWork(WorkPackage const& _wp)
{
    // Set work to each miner giving it's own starting nonce
    Guard l(x_minerWork);
    m_work = _wp;
    uint64_t _startNonce;
    if (m_work.exSizeBytes > 0)
    {
        // Equally divide the residual segment among miners
        _startNonce = m_work.startNonce;
        m_nonce_segment_with = log2(pow(2, 64 - (m_work.exSizeBytes * 4)) / m_miners.size());
    }
    else
    {
        // Get the randomly selected nonce
        _startNonce = m_nonce_scrambler;
    }

    for (unsigned int i = 0; i < m_miners.size(); i++)
    {
        m_work.startNonce = _startNonce + ((uint64_t)i << m_nonce_segment_with);
        m_miners.at(i)->setWork(m_work);
    }
}

void Farm::setSealers(std::map<std::string, SealerDescriptor> const& _sealers)
{
    m_sealers = _sealers;
}

/**
 * @brief Start a number of miners.
 */
bool Farm::start(std::string const& _sealer, bool mixed)
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::start() begin");
    Guard l(x_minerWork);
    if (!m_miners.empty() && m_lastSealer == _sealer)
    {
        DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::start() end1");
        return true;
    }
    if (!m_sealers.count(_sealer))
    {
        DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::start() end2");
        return false;
    }

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
        m_miners.push_back(std::shared_ptr<Miner>(m_sealers[_sealer].create(i)));

        // Start miners' threads. They should pause waiting for new work
        // package.
        m_miners.back()->startWorking();
    }

    m_isMining.store(true, std::memory_order_relaxed);
    m_lastSealer = _sealer;

    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::start() end");
    return true;
}

/**
 * @brief Stop all mining activities.
 */
void Farm::stop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::stop() begin");
    // Avoid re-entering if not actually mining.
    // This, in fact, is also called by destructor
    if (isMining())
    {
        {
            Guard l(x_minerWork);
            for (auto const& miner : m_miners)
                miner->triggerStopWorking();
            m_miners.clear();
            m_isMining.store(false, std::memory_order_relaxed);
        }
    }
    DEV_BUILD_LOG_PROGRAMFLOW(cnote, "Farm::stop() end");
}

void Farm::pause() 
{
    // Signal each miner to suspend mining
    Guard l(x_minerWork);
    m_paused.store(true, std::memory_order_relaxed);
    for (auto const& m : m_miners)
        m->pause(MinerPauseEnum::PauseDueToFarmPaused);
}

bool Farm::paused()
{
    return m_paused.load(std::memory_order_relaxed);
}

void Farm::resume()
{
    // Signal each miner to resume mining
    // Note ! Miners may stay suspended if other reasons
    Guard l(x_minerWork);
    m_paused.store(false, std::memory_order_relaxed);
    for (auto const& m : m_miners)
        m->resume(MinerPauseEnum::PauseDueToFarmPaused);
}

/**
 * @brief Stop all mining activities and Starts them again
 */
void Farm::restart()
{
    if (m_onMinerRestart)
    {
        m_onMinerRestart();
    }
}

/**
 * @brief Stop all mining activities and Starts them again (async post)
 */
void Farm::restart_async()
{
    m_io_strand.get_io_service().post(m_io_strand.wrap(boost::bind(&Farm::restart, this)));
}

/**
 * @brief Spawn a reboot script (reboot.bat/reboot.sh)
 * @return false if no matching file was found
 */
bool Farm::reboot(const std::vector<std::string>& args)
{
#if defined(_WIN32)
    const char* filename = "reboot.bat";
#else
    const char* filename = "reboot.sh";
#endif

    return spawn_file_in_bin_dir(filename, args);
}

string Farm::farmLaunchedFormatted()
{
    auto d = std::chrono::steady_clock::now() - m_farm_launched;
    int hsize = 3;
    auto hhh = std::chrono::duration_cast<std::chrono::hours>(d);
    if (hhh.count() < 100)
    {
        hsize = 2;
    }
    d -= hhh;
    auto mm = std::chrono::duration_cast<std::chrono::minutes>(d);
    std::ostringstream stream;
    stream << "Time: " << std::setfill('0') << std::setw(hsize) << hhh.count() << ':'
           << std::setfill('0') << std::setw(2) << mm.count();
    return stream.str();
}

/**
 * @brief Provides the description of segments each miner is working on
 * @return a JsonObject
 */
Json::Value Farm::get_nonce_scrambler_json()
{
    Json::Value jRes;
    jRes["noncescrambler"] = m_nonce_scrambler;
    jRes["segmentwidth"] = m_nonce_segment_with;

    for (size_t i = 0; i < m_miners.size(); i++)
    {
        Json::Value jSegment;
        uint64_t gpustartnonce = m_nonce_scrambler + ((uint64_t)pow(2, m_nonce_segment_with) * i);
        jSegment["gpu"] = (int)i;
        jSegment["start"] = gpustartnonce;
        jSegment["stop"] = uint64_t(gpustartnonce + (uint64_t)(pow(2, m_nonce_segment_with)));
        jRes["segments"].append(jSegment);
    }

    return jRes;
}

void Farm::setTStartTStop(unsigned tstart, unsigned tstop)
{
    m_tstart = tstart;
    m_tstop = tstop;
}

void Farm::submitProof(Solution const& _s)
{
    assert(m_onSolutionFound);

    if (!m_noeval)
    {
        Result r = EthashAux::eval(_s.work.epoch, _s.work.header, _s.nonce);
        if (r.value > _s.work.boundary)
        {
            failedSolution(_s.midx);
            cwarn << "GPU " << _s.midx
                  << " gave incorrect result. Lower overclocking values if it happens frequently.";
            return;
        }
    }

    m_onSolutionFound(_s);

#ifdef DEV_BUILD
    if (g_logOptions & LOG_SUBMIT)
        cnote << "Submit time: "
             << std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - _s.tstamp)
                    .count()
             << " us.";
#endif
    
}


// Collects data about hashing and hardware status
void Farm::collectData(const boost::system::error_code& ec)
{
    if (ec)
        return;

    WorkingProgress progress;

    // Process miners
    for (auto const& miner : m_miners)
    {
        // Collect and reset hashrates
        if (!miner->paused())
        {
            auto hr = miner->RetrieveHashRate();
            progress.hashRate += hr;
            progress.minersHashRates.push_back(hr);
            progress.miningIsPaused.push_back(false);
        }
        else
        {
            progress.minersHashRates.push_back(0.0);
            progress.miningIsPaused.push_back(true);
        }

        if (m_hwmonlvl)
        {
            HwMonitorInfo hwInfo = miner->hwmonInfo();
            HwMonitor hw;
            unsigned int tempC = 0, fanpcnt = 0, powerW = 0;
            if (hwInfo.deviceIndex >= 0)
            {
                if (hwInfo.deviceType == HwMonitorInfoType::NVIDIA && nvmlh)
                {
                    int typeidx = 0;
                    if (hwInfo.indexSource == HwMonitorIndexSource::CUDA)
                    {
                        typeidx = nvmlh->cuda_nvml_device_id[hwInfo.deviceIndex];
                    }
                    else if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
                    {
                        typeidx = nvmlh->opencl_nvml_device_id[hwInfo.deviceIndex];
                    }
                    else
                    {
                        // Unknown, don't map
                        typeidx = hwInfo.deviceIndex;
                    }
                    wrap_nvml_get_tempC(nvmlh, typeidx, &tempC);
                    wrap_nvml_get_fanpcnt(nvmlh, typeidx, &fanpcnt);
                    if (m_hwmonlvl == 2)
                    {
                        wrap_nvml_get_power_usage(nvmlh, typeidx, &powerW);
                    }
                }
                else if (hwInfo.deviceType == HwMonitorInfoType::AMD && adlh)
                {
                    int typeidx = 0;
                    if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
                    {
                        typeidx = adlh->opencl_adl_device_id[hwInfo.deviceIndex];
                    }
                    else
                    {
                        // Unknown, don't map
                        typeidx = hwInfo.deviceIndex;
                    }
                    wrap_adl_get_tempC(adlh, typeidx, &tempC);
                    wrap_adl_get_fanpcnt(adlh, typeidx, &fanpcnt);
                    if (m_hwmonlvl == 2)
                    {
                        wrap_adl_get_power_usage(adlh, typeidx, &powerW);
                    }
                }
#if defined(__linux)
                // Overwrite with sysfs data if present
                if (hwInfo.deviceType == HwMonitorInfoType::AMD && sysfsh)
                {
                    int typeidx = 0;
                    if (hwInfo.indexSource == HwMonitorIndexSource::OPENCL)
                    {
                        typeidx = sysfsh->opencl_sysfs_device_id[hwInfo.deviceIndex];
                    }
                    else
                    {
                        // Unknown, don't map
                        typeidx = hwInfo.deviceIndex;
                    }
                    wrap_amdsysfs_get_tempC(sysfsh, typeidx, &tempC);
                    wrap_amdsysfs_get_fanpcnt(sysfsh, typeidx, &fanpcnt);
                    if (m_hwmonlvl == 2)
                    {
                        wrap_amdsysfs_get_power_usage(sysfsh, typeidx, &powerW);
                    }
                }
#endif
            }

            // If temperature control has been enabled call
            // check threshold 
            if (m_tstop)
            {
                bool paused = miner->pauseTest(MinerPauseEnum::PauseDueToOverHeating);
                if (!paused && (tempC >= m_tstop))
                    miner->pause(MinerPauseEnum::PauseDueToOverHeating);
                if (paused && (tempC <= m_tstart))
                    miner->resume(MinerPauseEnum::PauseDueToOverHeating);
            }

            hw.tempC = tempC;
            hw.fanP = fanpcnt;
            hw.powerW = powerW / ((double)1000.0);
            progress.minerMonitors.push_back(hw);
        }
        miner->TriggerHashRateUpdate();
    }

    m_progress = progress;

    // Resubmit timer for another loop
    m_collectTimer.expires_from_now(boost::posix_time::milliseconds(m_collectInterval));
    m_collectTimer.async_wait(
        m_io_strand.wrap(boost::bind(&Farm::collectData, this, boost::asio::placeholders::error)));
}

bool Farm::spawn_file_in_bin_dir(const char* filename, const std::vector<std::string>& args)
{
    std::string fn = boost::dll::program_location().parent_path().string() +
                     "/" +  // boost::filesystem::path::preferred_separator
                     filename;
    try
    {
        if (!boost::filesystem::exists(fn))
            return false;

        /* anything in the file */
        if (!boost::filesystem::file_size(fn))
            return false;

#if defined(__linux)
        struct stat sb;
        if (stat(fn.c_str(), &sb) != 0)
            return false;
        /* just check if any exec flag is set.
           still execution can fail (not the uid, not in the group, selinux, ...)
         */
        if ((sb.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0)
            return false;
#endif
        /* spawn it (no wait,...) - fire and forget! */
        boost::process::spawn(fn, args);
        return true;
    }
    catch (...)
    {
    }
    return false;
}


}  // namespace eth
}  // namespace dev
