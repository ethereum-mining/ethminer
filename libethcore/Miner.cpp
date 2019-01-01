/*
 This file is part of ethereum.

 ethminer is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 ethereum is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Miner.h"

namespace dev
{
namespace eth
{
unsigned Miner::s_dagLoadMode = 0;
unsigned Miner::s_dagLoadIndex = 0;
unsigned Miner::s_minersCount = 0;

FarmFace* FarmFace::m_this = nullptr;

Miner::Miner(std::string const& _name, unsigned _index)
  : Worker(_name + std::to_string(_index)), m_index(_index)
{
    m_work_latest.header = h256();
}

DeviceDescriptor Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {
        boost::mutex::scoped_lock l(x_work);
        m_work_latest = _work;
#ifdef _DEVELOPER
        m_workSwitchStart = std::chrono::steady_clock::now();
#endif
    }

    kick_miner();
}

void Miner::stopWorking()
{
    Worker::stopWorking();
    kick_miner();
}

void Miner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
    m_new_work_signal.notify_one();
}

void Miner::pause(MinerPauseEnum what)
{
    boost::mutex::scoped_lock l(x_pause);
    m_pauseFlags.set(what);
    kick_miner();
}

bool Miner::paused()
{
    boost::mutex::scoped_lock l(x_pause);
    return m_pauseFlags.any();
}

bool Miner::pauseTest(MinerPauseEnum what)
{
    boost::mutex::scoped_lock l(x_pause);
    return m_pauseFlags.test(what);
}

std::string Miner::pausedString()
{
    boost::mutex::scoped_lock l(x_pause);
    std::string retVar;
    if (m_pauseFlags.any())
    {
        for (int i = 0; i < MinerPauseEnum::Pause_MAX; i++)
        {
            if (m_pauseFlags[(MinerPauseEnum)i])
            {
                if (!retVar.empty())
                    retVar.append("; ");

                if (i == MinerPauseEnum::PauseDueToOverHeating)
                    retVar.append("Overheating");
                else if (i == MinerPauseEnum::PauseDueToAPIRequest)
                    retVar.append("Api request");
                else if (i == MinerPauseEnum::PauseDueToFarmPaused)
                    retVar.append("Farm suspended");
                else if (i == MinerPauseEnum::PauseDueToInsufficientMemory)
                    retVar.append("Insufficient GPU memory");
                else if (i == MinerPauseEnum::PauseDueToInitEpochError)
                    retVar.append("Epoch initialization error");
            }
        }
    }
    return retVar;
}

void Miner::resume(MinerPauseEnum fromwhat)
{
    boost::mutex::scoped_lock l(x_pause);
    m_pauseFlags.reset(fromwhat);
    if (!m_pauseFlags.any())
        kick_miner();
}

float Miner::RetrieveHashRate() noexcept
{
    return m_hashRate.load(std::memory_order_relaxed);
}

void Miner::TriggerHashRateUpdate() noexcept
{
    bool b = false;
    if (m_hashRateUpdate.compare_exchange_strong(b, true))
        return;
    // GPU didn't respond to last trigger, assume it's dead.
    // This can happen on CUDA if:
    //   runtime of --cuda-grid-size * --cuda-streams exceeds time of m_collectInterval
    m_hashRate = 0.0;
}

bool Miner::initEpoch()
{
    // When loading of DAG is sequential wait for
    // this instance to become current
    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
    {
        while (s_dagLoadIndex < m_index)
        {
            boost::system_time const timeout =
                boost::get_system_time() + boost::posix_time::seconds(3);
            boost::mutex::scoped_lock l(x_work);
            m_dag_loaded_signal.timed_wait(l, timeout);
        }
        if (shouldStop())
            return false;
    }

    // Run the internal initialization
    // specific for miner
    bool result = initEpoch_internal();

    // Advance to next miner or reset to zero for
    // next run if all have processed
    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
    {
        s_dagLoadIndex = (m_index + 1);
        if (s_minersCount == s_dagLoadIndex)
            s_dagLoadIndex = 0;
        else
            m_dag_loaded_signal.notify_all();
    }

    return result;
}

bool Miner::initEpoch_internal()
{
    // If not overridden in derived class
    this_thread::sleep_for(std::chrono::seconds(5));
    return true;
}

void Miner::minerLoop()
{

    bool newEpoch, newProgPoWPeriod;

    // Don't catch exceptions here !!
    // They will be handled in workLoop implemented in derived class
    while (!shouldStop())
    {
        // Wait for work or 3 seconds (whichever the first)
        if (!m_new_work.load(memory_order_relaxed))
        {
            boost::system_time const timeout =
                boost::get_system_time() + boost::posix_time::seconds(3);
            boost::mutex::scoped_lock l(x_work);
            m_new_work_signal.timed_wait(l, timeout);
            continue;
        }

        // Got new work
        m_new_work.store(false, memory_order_relaxed);

        if (shouldStop())  // Exit ! Request to terminate
            break;
        if (paused() || !m_work_latest)  // Wait ! Gpu is not ready or there is no work
            continue;

        // Copy latest work into active slot
        {
            boost::mutex::scoped_lock l(x_work);
            newEpoch = (m_work_latest.epoch != m_work_active.epoch);
            newProgPoWPeriod = (m_work_latest.block / PROGPOW_PERIOD != m_work_active.period);
            m_work_active = m_work_latest;
            l.unlock();
        }

        // Epoch change ?
        if (newEpoch)
        {
            if (!initEpoch())
                break;  // This will simply exit the thread

            // As DAG generation takes a while we need to
            // ensure we're on latest job, not on the one
            // which triggered the epoch change
            if (m_new_work.load(memory_order_relaxed))
                continue;
        }

        if (m_work_active.algo == "ethash")
        {
            // Start ethash searching
            ethash_search();
        }
        else if (m_work_active.algo == "progpow")
        {

            m_work_active.period = m_work_active.block / PROGPOW_PERIOD;
            if (newProgPoWPeriod)
            {
                unloadProgPoWKernel();

                uint32_t dagelms = (unsigned)(m_epochContext.dagSize / ETHASH_MIX_BYTES);
                compileProgPoWKernel(m_work_active.block, dagelms);

                // During compilation a new job might have reached
                if (m_new_work.load(memory_order_relaxed))
                    continue;
            }

            // Start progpow searching
            progpow_search();
        }
        else
        {
            throw std::runtime_error("Algo : " + m_work_active.algo + " not yet implemented");
        }
    }

    unloadProgPoWKernel();
}

void Miner::updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept
{
    m_groupCount += _increment;
    bool b = true;
    if (!m_hashRateUpdate.compare_exchange_strong(b, false))
        return;
    using namespace std::chrono;
    auto t = steady_clock::now();
    auto us = duration_cast<microseconds>(t - m_hashTime).count();
    m_hashTime = t;

    m_hashRate.store(
        us ? (float(m_groupCount * _groupSize) * 1.0e6f) / us : 0.0f, std::memory_order_relaxed);
    m_groupCount = 0;
}

}  // namespace eth
}  // namespace dev
