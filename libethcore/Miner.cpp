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

FarmFace* FarmFace::m_this = nullptr;

std::ostream& operator<<(std::ostream& os, const HwMonitor& _hw)
{
    os << _hw.tempC << "C " << _hw.fanP << "%";
    if (_hw.powerW)
        os << ' ' << fixed << setprecision(0) << _hw.powerW << "W";
    return os;
}

std::ostream& operator<<(std::ostream& _out, const WorkingProgress& _p)
{
    float mh = _p.hashRate / 1000000.0f;
    _out << "Speed " << EthTealBold << std::fixed << std::setprecision(2) << mh << EthReset
         << " Mh/s";

    for (size_t i = 0; i < _p.minersHashRates.size(); ++i)
    {
        mh = _p.minersHashRates[i] / 1000000.0f;

        if (_p.miningIsPaused.size() == _p.minersHashRates.size())
        {
            // red color if mining is paused on this gpu
            if (_p.miningIsPaused[i])
            {
                _out << EthRed;
            }
        }

        _out << " gpu" << i << " " << EthTeal << std::fixed << std::setprecision(2) << mh
             << EthReset;
        if (_p.minerMonitors.size() == _p.minersHashRates.size())
            _out << " " << EthTeal << _p.minerMonitors[i] << EthReset;
    }

    return _out;
}

void Miner::setDescriptor(DeviceDescriptorType& _descriptor) 
{
    m_deviceDescriptor = _descriptor;
}

DeviceDescriptorType Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {

        boost::mutex::scoped_lock l(x_work);

        // Void work if this miner is paused
        if (paused())
            m_work.header = h256();
        else
            m_work = _work;

#ifdef DEV_BUILD
        m_workSwitchStart = std::chrono::steady_clock::now();
#endif
    }

    kick_miner();
}

void Miner::pause(MinerPauseEnum what) 
{
    boost::mutex::scoped_lock l(x_pause);
    m_pauseFlags.set(what);
    m_work.header = h256();
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
    //if (!m_pauseFlags.any())
    //{
    //    // TODO Push most recent job from farm ?
    //    // If we do not push a new job the miner will stay idle
    //    // till a new job arrives
    //}
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

    // Advance to next miner
    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
    {
        s_dagLoadIndex = (m_index + 1);
        m_dag_loaded_signal.notify_all();
    }

    return result;
}

WorkPackage Miner::work() const
{
    boost::mutex::scoped_lock l(x_work);
    return m_work;
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
