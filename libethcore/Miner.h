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

#include <bitset>
#include <list>
#include <numeric>
#include <string>

#include "EthashAux.h"
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>

#include <boost/thread.hpp>

#define MINER_WAIT_STATE_WORK 1

#define DAG_LOAD_MODE_PARALLEL 0
#define DAG_LOAD_MODE_SEQUENTIAL 1
#define DAG_LOAD_MODE_SINGLE 2

using namespace std;

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

enum class HwMonitorInfoType
{
    UNKNOWN,
    NVIDIA,
    AMD
};

enum class HwMonitorIndexSource
{
    UNKNOWN,
    OPENCL,
    CUDA
};

struct HwMonitorInfo
{
    HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
    HwMonitorIndexSource indexSource = HwMonitorIndexSource::UNKNOWN;
    int deviceIndex = -1;
};

struct HwMonitor
{
    int tempC = 0;
    int fanP = 0;
    double powerW = 0;
};

std::ostream& operator<<(std::ostream& os, const HwMonitor& _hw);

class FormattedMemSize
{
public:
    explicit FormattedMemSize(uint64_t s) noexcept { m_size = s; }
    uint64_t m_size;
};

std::ostream& operator<<(std::ostream& os, const FormattedMemSize& s);

/// Pause mining
enum MinerPauseEnum
{
    PauseDueToOverHeating,
    PauseDueToAPIRequest,
    PauseDueToFarmPaused,
    Pause_MAX  // Must always be last as a placeholder of max count
};


/// Describes the progress of a mining operation.
struct WorkingProgress
{
    float hashRate = 0.0;

    std::vector<float> minersHashRates;
    std::vector<bool> miningIsPaused;
    std::vector<HwMonitor> minerMonitors;
};

std::ostream& operator<<(std::ostream& _out, const WorkingProgress& _p);

class SolutionStats  // Only updated by Poolmanager thread!
{
public:
    void reset()
    {
        m_accepts = {};
        m_rejects = {};
        m_failures = {};
        m_acceptedStales = {};
    }

    void accepted(unsigned miner_index)
    {
        if (m_accepts.size() <= miner_index)
            m_accepts.resize(miner_index + 1);
        m_accepts[miner_index]++;
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, m_tpInitalized);
        m_lastUpdated[miner_index] = std::chrono::steady_clock::now();
    }
    void rejected(unsigned miner_index)
    {
        if (m_rejects.size() <= miner_index)
            m_rejects.resize(miner_index + 1);
        m_rejects[miner_index]++;
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, m_tpInitalized);
        m_lastUpdated[miner_index] = std::chrono::steady_clock::now();
    }
    void failed(unsigned miner_index)
    {
        if (m_failures.size() <= miner_index)
            m_failures.resize(miner_index + 1);
        m_failures[miner_index]++;
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, m_tpInitalized);
        m_lastUpdated[miner_index] = std::chrono::steady_clock::now();
    }
    void acceptedStale(unsigned miner_index)
    {
        if (m_acceptedStales.size() <= miner_index)
            m_acceptedStales.resize(miner_index + 1);
        m_acceptedStales[miner_index]++;
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, m_tpInitalized);
        m_lastUpdated[miner_index] = std::chrono::steady_clock::now();
    }

    unsigned getAccepts() const { return accumulate(m_accepts.begin(), m_accepts.end(), 0); }
    unsigned getRejects() const { return accumulate(m_rejects.begin(), m_rejects.end(), 0); }
    unsigned getFailures() const { return accumulate(m_failures.begin(), m_failures.end(), 0); }
    unsigned getAcceptedStales() const
    {
        return accumulate(m_acceptedStales.begin(), m_acceptedStales.end(), 0);
    }

    unsigned getAccepts(unsigned miner_index) const
    {
        if (m_accepts.size() <= miner_index)
            return 0;
        return m_accepts[miner_index];
    }
    unsigned getRejects(unsigned miner_index) const
    {
        if (m_rejects.size() <= miner_index)
            return 0;
        return m_rejects[miner_index];
    }
    unsigned getFailures(unsigned miner_index) const
    {
        if (m_failures.size() <= miner_index)
            return 0;
        return m_failures[miner_index];
    }
    unsigned getAcceptedStales(unsigned miner_index) const
    {
        if (m_acceptedStales.size() <= miner_index)
            return 0;
        return m_acceptedStales[miner_index];
    }
    std::chrono::steady_clock::time_point getLastUpdated(unsigned miner_index) const
    {
        if (m_lastUpdated.size() <= miner_index)
            return m_tpInitalized;
        return m_lastUpdated[miner_index];
    }
    std::chrono::steady_clock::time_point getLastUpdated() const
    {
        /* return the newest update time of all GPUs */
        if (!m_lastUpdated.size())
            return m_tpInitalized;
        auto max_index = std::max_element(m_lastUpdated.begin(), m_lastUpdated.end());
        return m_lastUpdated[std::distance(m_lastUpdated.begin(), max_index)];
    }

    std::string getString(unsigned miner_index)
    {
        ostringstream r;

        r << "A" << getAccepts(miner_index);
        auto stales = getAcceptedStales(miner_index);
        if (stales)
            r << "+" << stales;
        auto rejects = getRejects(miner_index);
        if (rejects)
            r << ":R" << rejects;
        auto failures = getFailures(miner_index);
        if (failures)
            r << ":F" << failures;
        return r.str();
    }

private:
    std::vector<unsigned> m_accepts = {};
    std::vector<unsigned> m_rejects = {};
    std::vector<unsigned> m_failures = {};
    std::vector<unsigned> m_acceptedStales = {};
    std::vector<std::chrono::steady_clock::time_point> m_lastUpdated = {};
    const std::chrono::steady_clock::time_point m_tpInitalized = std::chrono::steady_clock::now();
};

std::ostream& operator<<(std::ostream& os, const SolutionStats& s);

/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
class FarmFace
{
public:
    FarmFace() { m_this = this; }
    static FarmFace& f() { return *m_this; };

    virtual ~FarmFace() = default;
    virtual unsigned get_tstart() = 0;
    virtual unsigned get_tstop() = 0;

    /**
     * @brief Called from a Miner to note a WorkPackage has a solution.
     * @param _p The solution.
     * @return true iff the solution was good (implying that mining should be .
     */
    virtual void submitProof(Solution const& _p) = 0;
    virtual void failedSolution(unsigned _miner_index) = 0;
    virtual uint64_t get_nonce_scrambler() = 0;
    virtual unsigned get_segment_width() = 0;

private:
    static FarmFace* m_this;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */
#define MAX_MINERS 32U

class Miner : public Worker
{
public:
    Miner(std::string const& _name, size_t _index)
      : Worker(_name + std::to_string(_index)), m_index(_index)
    {}

    virtual ~Miner() = default;

    /**
     * @brief Assigns hashing work to this instance
     */
    void setWork(WorkPackage const& _work);

    unsigned Index() { return m_index; };

    HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    /**
     * @brief Pauses mining setting a reason flag
     */
    void pause(MinerPauseEnum what);

    /**
     * @brief Whether or not this miner is paused for any reason
     */
    bool paused();

    /**
     * @brief Checks if the given reason for pausing is currently active
    */
    bool pauseTest(MinerPauseEnum what);

    /**
     * @brief Returns the human readable reason for this miner being paused
     */
    std::string pausedString();

    /**
     * @brief Cancels a pause flag.
     * @note Miner can be paused for multiple reasons at a time.
     */
    void resume(MinerPauseEnum fromwhat);

    float RetrieveHashRate() noexcept { return m_hashRate.load(std::memory_order_relaxed); }

    void TriggerHashRateUpdate() noexcept
    {
        bool b = false;
        if (m_hashRateUpdate.compare_exchange_strong(b, true))
            return;
        // GPU didn't respond to last trigger, assume it's dead.
        // This can happen on CUDA if:
        //   runtime of --cuda-grid-size * --cuda-streams exceeds time of m_collectInterval
        m_hashRate = 0.0;
    }

protected:
    /**
     * @brief No work left to be done. Pause until told to kickOff().
     */
    virtual void kick_miner() = 0;

    /**
     * @brief Returns current workpackage this miner is working on
     */
    WorkPackage work() const;

    void updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept;

    static unsigned s_dagLoadMode;
    static unsigned s_dagLoadIndex;
    static unsigned s_dagCreateDevice;
    static uint8_t* s_dagInHostMemory;
    const unsigned m_index = 0;

#ifdef DEV_BUILD
    std::chrono::steady_clock::time_point m_workSwitchStart;
#endif

    HwMonitorInfo m_hwmoninfo;
    mutable boost::mutex x_work;
    mutable boost::mutex x_pause;
    boost::condition_variable m_new_work_signal;
    boost::condition_variable m_dag_loaded_signal;

private:
    bitset<MinerPauseEnum::Pause_MAX> m_pauseFlags;
    WorkPackage m_work;

    std::chrono::steady_clock::time_point m_hashTime = std::chrono::steady_clock::now();
    std::atomic<float> m_hashRate = {0.0};
    uint64_t m_groupCount = 0;
    atomic<bool> m_hashRateUpdate = {false};
};

}  // namespace eth
}  // namespace dev
