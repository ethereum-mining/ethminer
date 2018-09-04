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

#include <list>
#include <string>
#include <thread>
#include <numeric>

#include <boost/circular_buffer.hpp>
#include <boost/timer.hpp>

#include "EthashAux.h"
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>

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
typedef enum {
    MINING_NOT_PAUSED = 0x00000000,
    MINING_PAUSED_WAIT_FOR_T_START = 0x00000001,
    MINING_PAUSED_API = 0x00000002
    // MINING_PAUSED_USER             = 0x00000004,
    // MINING_PAUSED_ERROR            = 0x00000008
} MinigPauseReason;

struct MiningPause
{
    std::atomic<uint64_t> m_mining_paused_flag = {MinigPauseReason::MINING_NOT_PAUSED};

    void set_mining_paused(MinigPauseReason pause_reason)
    {
        m_mining_paused_flag.fetch_or(pause_reason, std::memory_order_seq_cst);
    }

    void clear_mining_paused(MinigPauseReason pause_reason)
    {
        m_mining_paused_flag.fetch_and(~pause_reason, std::memory_order_seq_cst);
    }

    MinigPauseReason get_mining_paused()
    {
        return (MinigPauseReason)m_mining_paused_flag.load(std::memory_order_relaxed);
    }

    bool is_mining_paused()
    {
        return (m_mining_paused_flag.load(std::memory_order_relaxed) !=
                MinigPauseReason::MINING_NOT_PAUSED);
    }
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

class SolutionStats
{
public:
    void accepted() { accepts++; }
    void rejected() { rejects++; }
    void failed() { failures++; }

    void acceptedStale() { acceptedStales++; }

    void reset() { accepts = rejects = failures = acceptedStales = 0; }

    unsigned getAccepts() const { return accepts; }
    unsigned getRejects() const { return rejects; }
    unsigned getFailures() const { return failures; }
    unsigned getAcceptedStales() const { return acceptedStales; }

private:
    unsigned accepts = 0;
    unsigned rejects = 0;
    unsigned failures = 0;

    unsigned acceptedStales = 0;
};

std::ostream& operator<<(std::ostream& os, const SolutionStats& s);

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
    virtual unsigned get_tstart() = 0;
    virtual unsigned get_tstop() = 0;
    /**
     * @brief Called from a Miner to note a WorkPackage has a solution.
     * @param _p The solution.
     * @return true iff the solution was good (implying that mining should be .
     */
    virtual void submitProof(Solution const& _p) = 0;
    virtual void failedSolution() = 0;
    virtual uint64_t get_nonce_scrambler() = 0;
    virtual unsigned get_segment_width() = 0;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */
#define LOG2_MAX_MINERS 5u
#define MAX_MINERS (1u << LOG2_MAX_MINERS)

class Miner : public Worker
{
public:
    Miner(std::string const& _name, FarmFace& _farm, size_t _index)
      : Worker(_name + std::to_string(_index)), index(_index), farm(_farm)
    {
    }

    virtual ~Miner() = default;

    void setWork(WorkPackage const& _work)
    {
        {
            Guard l(x_work);
            m_work = _work;
            if (g_logVerbosity >= 6)
                workSwitchStart = std::chrono::steady_clock::now();
        }
        kick_miner();
    }

    unsigned Index() { return index; };

    HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    uint64_t get_start_nonce()
    {
        // Each GPU is given a non-overlapping 2^40 range to search
        // return farm.get_nonce_scrambler() + ((uint64_t) index << 40);

        // Now segment size is adjustable
        return farm.get_nonce_scrambler() + (uint64_t)(pow(2, farm.get_segment_width()) * index);
    }

    void update_temperature(unsigned temperature)
    {
        /*
         cnote << "Setting temp" << temperature << " for gpu" << index <<
                  " tstop=" << farm.get_tstop() << " tstart=" << farm.get_tstart();
        */
        bool _wait_for_tstart_temp = (m_mining_paused.get_mining_paused() &
                                         MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START) ==
                                     MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START;
        if (!_wait_for_tstart_temp)
        {
            unsigned tstop = farm.get_tstop();
            if (tstop && temperature >= tstop)
            {
                cwarn << "Pause mining on gpu" << index << " : temperature " << temperature
                      << " is equal/above --tstop " << tstop;
                m_mining_paused.set_mining_paused(MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START);
            }
        }
        else
        {
            unsigned tstart = farm.get_tstart();
            if (tstart && temperature <= tstart)
            {
                cnote << "(Re)starting mining on gpu" << index << " : temperature " << temperature
                      << " is now below/equal --tstart " << tstart;
                m_mining_paused.clear_mining_paused(
                    MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START);
            }
        }
    }

    void set_mining_paused(MinigPauseReason pause_reason)
    {
        m_mining_paused.set_mining_paused(pause_reason);
    }

    void clear_mining_paused(MinigPauseReason pause_reason)
    {
        m_mining_paused.clear_mining_paused(pause_reason);
    }

    bool is_mining_paused() { return m_mining_paused.is_mining_paused(); }

    float RetrieveHashRate() { return m_hashRate.load(std::memory_order_relaxed); }

protected:
    /**
     * @brief No work left to be done. Pause until told to kickOff().
     */
    virtual void kick_miner() = 0;

    WorkPackage work() const
    {
        Guard l(x_work);
        return m_work;
    }

    inline void updateHashRate(uint64_t _n)
    {
        using namespace std::chrono;
        steady_clock::time_point t = steady_clock::now();
        auto us = duration_cast<microseconds>(t - m_hashTime).count();
        m_hashTime = t;

        float hr = 0.0;
        if (us)
            hr = (float(_n) * 1.0e6f) / us;
        m_hashRate.store(hr, std::memory_order_relaxed);
    }

    static unsigned s_dagLoadMode;
    static unsigned s_dagLoadIndex;
    static unsigned s_dagCreateDevice;
    static uint8_t* s_dagInHostMemory;
    static bool s_exit;
    static bool s_noeval;

    const size_t index = 0;
    FarmFace& farm;
    std::chrono::steady_clock::time_point workSwitchStart;
    HwMonitorInfo m_hwmoninfo;

private:
    MiningPause m_mining_paused;
    WorkPackage m_work;
    mutable Mutex x_work;
    std::chrono::steady_clock::time_point m_hashTime = std::chrono::steady_clock::now();
    std::atomic<float> m_hashRate = {0.0};
};

}  // namespace eth
}  // namespace dev
