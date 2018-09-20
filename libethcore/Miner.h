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

    bool is_mining_paused(const MinigPauseReason& pause_reason)
    {
        return (pause_reason != MinigPauseReason::MINING_NOT_PAUSED);
    }

    bool is_mining_paused()
    {
        return is_mining_paused(get_mining_paused());
    }

    std::string get_mining_paused_string(const MinigPauseReason& pause_reason)
    {
        std::string r;

        if (pause_reason & MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START)
            r = "temperature";

        if (pause_reason & MinigPauseReason::MINING_PAUSED_API)
            r += (string)(r.empty() ? "" : ",") + "api";

        return r;
    }

    std::string get_mining_paused_string()
    {
        return get_mining_paused_string(get_mining_paused());
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
        auto now = std::chrono::steady_clock::now();
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, now);
        m_lastUpdated[miner_index] = now;
    }
    void rejected(unsigned miner_index)
    {
        if (m_rejects.size() <= miner_index)
            m_rejects.resize(miner_index + 1);
        m_rejects[miner_index]++;
        auto now = std::chrono::steady_clock::now();
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, now);
        m_lastUpdated[miner_index] = now;
    }
    void failed(unsigned miner_index)
    {
        if (m_failures.size() <= miner_index)
            m_failures.resize(miner_index + 1);
        m_failures[miner_index]++;
        auto now = std::chrono::steady_clock::now();
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, now);
        m_lastUpdated[miner_index] = now;
    }
    void acceptedStale(unsigned miner_index)
    {
        if (m_acceptedStales.size() <= miner_index)
            m_acceptedStales.resize(miner_index + 1);
        m_acceptedStales[miner_index]++;
        auto now = std::chrono::steady_clock::now();
        if (m_lastUpdated.size() <= miner_index)
            m_lastUpdated.resize(miner_index + 1, now);
        m_lastUpdated[miner_index] = now;
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
            return std::chrono::steady_clock::now();
        return m_lastUpdated[miner_index];
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
#define LOG2_MAX_MINERS 5u
#define MAX_MINERS (1u << LOG2_MAX_MINERS)

class Miner : public Worker
{
public:
    Miner(std::string const& _name, size_t _index)
      : Worker(_name + std::to_string(_index)), m_index(_index)
    {
    }

    virtual ~Miner() = default;

    void setWork(WorkPackage const& _work)
    {
        {
            Guard l(x_work);
            m_work = _work;
            if (m_work.exSizeBits >= 0)
            {
                // This can support up to 2^c_log2MaxMiners devices.
                m_work.startNonce =
                    m_work.startNonce +
                    ((uint64_t)m_index << (64 - LOG2_MAX_MINERS - m_work.exSizeBits));
            }
            else
            {
                // Each GPU is given a non-overlapping 2^40 range to search
                // return farm.get_nonce_scrambler() + ((uint64_t) m_index << 40);

                // Now segment size is adjustable
                m_work.startNonce = FarmFace::f().get_nonce_scrambler() +
                                    ((uint64_t)m_index << FarmFace::f().get_segment_width());
            }

#ifdef DEV_BUILD
            m_workSwitchStart = std::chrono::steady_clock::now();
#endif
        }
        kick_miner();
    }

    unsigned Index() { return m_index; };

    HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    void update_temperature(unsigned temperature)
    {
        /*
         cnote << "Setting temp" << temperature << " for gpu" << m_index <<
                  " tstop=" << FarmFace::f().get_tstop() << " tstart=" <<
         FarmFace::f().get_tstart();
        */
        bool _wait_for_tstart_temp = (m_mining_paused.get_mining_paused() &
                                         MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START) ==
                                     MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START;
        if (!_wait_for_tstart_temp)
        {
            unsigned tstop = FarmFace::f().get_tstop();
            if (tstop && temperature >= tstop)
            {
                cwarn << "Pause mining on gpu" << m_index << " : temperature " << temperature
                      << " is equal/above --tstop " << tstop;
                m_mining_paused.set_mining_paused(MinigPauseReason::MINING_PAUSED_WAIT_FOR_T_START);
            }
        }
        else
        {
            unsigned tstart = FarmFace::f().get_tstart();
            if (tstart && temperature <= tstart)
            {
                cnote << "(Re)starting mining on gpu" << m_index << " : temperature " << temperature
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

    MinigPauseReason get_mining_paused() { return m_mining_paused.get_mining_paused(); }

    bool is_mining_paused() { return m_mining_paused.is_mining_paused(); }

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

    WorkPackage work() const
    {
        Guard l(x_work);
        return m_work;
    }

    void updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept
    {
        m_groupCount += _increment;
        bool b = true;
        if (!m_hashRateUpdate.compare_exchange_strong(b, false))
            return;
        using namespace std::chrono;
        auto t = steady_clock::now();
        auto us = duration_cast<microseconds>(t - m_hashTime).count();
        m_hashTime = t;

        m_hashRate.store(us ? (float(m_groupCount * _groupSize) * 1.0e6f) / us : 0.0f,
            std::memory_order_relaxed);
        m_groupCount = 0;
    }

    static unsigned s_dagLoadMode;
    static unsigned s_dagLoadIndex;
    static unsigned s_dagCreateDevice;
    static uint8_t* s_dagInHostMemory;
    static bool s_exit;
    static bool s_noeval;

    const size_t m_index = 0;
#ifdef DEV_BUILD
    std::chrono::steady_clock::time_point m_workSwitchStart;
#endif
    HwMonitorInfo m_hwmoninfo;

private:
    MiningPause m_mining_paused;
    WorkPackage m_work;
    mutable Mutex x_work;
    std::chrono::steady_clock::time_point m_hashTime = std::chrono::steady_clock::now();
    std::atomic<float> m_hashRate = {0.0};
    uint64_t m_groupCount = 0;
    atomic<bool> m_hashRateUpdate = {false};
};

}  // namespace eth
}  // namespace dev
