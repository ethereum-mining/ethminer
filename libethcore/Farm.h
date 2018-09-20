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

#pragma once

#include <atomic>
#include <list>
#include <thread>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/dll.hpp>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include <json/json.h>

#include <libdevcore/Common.h>
#include <libdevcore/Worker.h>
#include <libethcore/BlockHeader.h>
#include <libethcore/Miner.h>
#include <libhwmon/wrapadl.h>
#include <libhwmon/wrapnvml.h>
#if defined(__linux)
#include <libhwmon/wrapamdsysfs.h>
#include <sys/stat.h>
#endif

extern boost::asio::io_service g_io_service;

namespace dev
{
namespace eth
{
/**
 * @brief A collective of Miners.
 * Miners ask for work, then submit proofs
 * @threadsafe
 */
class Farm : public FarmFace
{
public:
    unsigned tstart = 0, tstop = 0;

    struct SealerDescriptor
    {
        std::function<unsigned()> instances;
        std::function<Miner*(unsigned)> create;
    };

    Farm(bool hwmon, bool pwron);

    ~Farm();

    static Farm& f() { return *m_this; }

    /**
     * @brief Randomizes the nonce scrambler
     */
    void shuffle();

    /**
     * @brief Sets the current mining mission.
     * @param _wp The work package we wish to be mining.
     */
    void setWork(WorkPackage const& _wp)
    {
        // Set work to each miner
        Guard l(x_minerWork);
        m_work = _wp;
        for (auto const& m : m_miners)
            m->setWork(m_work);
    }

    void setSealers(std::map<std::string, SealerDescriptor> const& _sealers);

    /**
     * @brief Start a number of miners.
     */
    bool start(std::string const& _sealer, bool mixed);

    /**
     * @brief Stop all mining activities.
     */
    void stop();

    /**
     * @brief Stop all mining activities and Starts them again
     */
    void restart();

    /**
     * @brief Stop all mining activities and Starts them again (async post)
     */
    void restart_async();

    bool isMining() const { return m_isMining.load(std::memory_order_relaxed); }

    /**
     * @brief Spawn a reboot script (reboot.bat/reboot.sh)
     * @return false if no matching file was found
     */
    bool reboot(const std::vector<std::string>& args);

    /**
     * @brief Get information on the progress of mining this work package.
     * @return The progress with mining so far.
     */
    WorkingProgress const& miningProgress() const
    {
        return m_progress;
    }

    std::vector<std::shared_ptr<Miner>> getMiners() { return m_miners; }

    std::shared_ptr<Miner> getMiner(unsigned index)
    {
        if (index >= m_miners.size())
            return nullptr;
        return m_miners[index];
    }

    SolutionStats getSolutionStats() { return m_solutionStats; } // returns a copy

    void failedSolution(unsigned _miner_index) override { m_solutionStats.failed(_miner_index); }

    void acceptedSolution(bool _stale, unsigned _miner_index)
    {
        if (!_stale)
        {
            m_solutionStats.accepted(_miner_index);
        }
        else
        {
            m_solutionStats.acceptedStale(_miner_index);
        }
    }

    void rejectedSolution(unsigned _miner_index) { m_solutionStats.rejected(_miner_index); }

    using SolutionFound = std::function<void(const Solution&)>;
    using MinerRestart = std::function<void()>;

    /**
     * @brief Provides a valid header based upon that received previously with setWork().
     * @param _bi The now-valid header.
     * @return true if the header was good and that the Farm should pause until more work is
     * submitted.
     */
    void onSolutionFound(SolutionFound const& _handler) { m_onSolutionFound = _handler; }
    void onMinerRestart(MinerRestart const& _handler) { m_onMinerRestart = _handler; }

    WorkPackage work() const
    {
        Guard l(x_minerWork);
        return m_work;
    }

    std::chrono::steady_clock::time_point farmLaunched() { return m_farm_launched; }

    string farmLaunchedFormatted();

    uint64_t get_nonce_scrambler() override { return m_nonce_scrambler; }

    unsigned get_segment_width() override { return m_nonce_segment_with; }

    void set_nonce_scrambler(uint64_t n) { m_nonce_scrambler = n; }

    void set_nonce_segment_width(unsigned n) { m_nonce_segment_with = n; }

    /**
     * @brief Provides the description of segments each miner is working on
     * @return a JsonObject
     */
    Json::Value get_nonce_scrambler_json();

    void setTStartTStop(unsigned tstart, unsigned tstop);

    unsigned get_tstart() override { return m_tstart; }

    unsigned get_tstop() override { return m_tstop; }

    /**
     * @brief Called from a Miner to note a WorkPackage has a solution.
     * @param _s The solution.
     * @param _miner_index Index of the miner
     */
    void submitProof(Solution const& _s) override
    {
        assert(m_onSolutionFound);
        m_onSolutionFound(_s);
    }

private:
    // Collects data about hashing and hardware status
    void collectData(const boost::system::error_code& ec);

    /**
     * @brief Spawn a file - must be located in the directory of ethminer binary
     * @return false if file was not found or it is not executeable
     */
    bool spawn_file_in_bin_dir(const char* filename, const std::vector<std::string>& args);

    mutable Mutex x_minerWork;
    std::vector<std::shared_ptr<Miner>> m_miners;
    WorkPackage m_work;

    std::atomic<bool> m_isMining = {false};

    mutable WorkingProgress m_progress;

    SolutionFound m_onSolutionFound;
    MinerRestart m_onMinerRestart;

    std::map<std::string, SealerDescriptor> m_sealers;
    std::string m_lastSealer;

    boost::asio::io_service::strand m_io_strand;
    boost::asio::deadline_timer m_collectTimer;
    static const int m_collectInterval = 5000;

    mutable SolutionStats m_solutionStats;
    std::chrono::steady_clock::time_point m_farm_launched = std::chrono::steady_clock::now();

    string m_pool_addresses;

    // StartNonce (non-NiceHash Mode) and
    // segment width assigned to each GPU as exponent of 2
    uint64_t m_nonce_scrambler;
    unsigned int m_nonce_segment_with = 40;

    // Switches for hw monitoring and power drain monitoring
    bool m_hwmon, m_pwron;

    // Hardware monitoring temperatures
    unsigned m_tstart = 0, m_tstop = 0;

    // Wrappers for hardware monitoring libraries
    wrap_nvml_handle* nvmlh = nullptr;
    wrap_adl_handle* adlh = nullptr;

#if defined(__linux)
    wrap_amdsysfs_handle* sysfsh = nullptr;
#endif

    static Farm* m_this;
};

}  // namespace eth
}  // namespace dev

