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

unsigned Miner::s_dagCreateDevice = 0;

uint8_t* Miner::s_dagInHostMemory = nullptr;

FarmFace* FarmFace::m_this = nullptr;

bool Miner::s_exit = false;


std::ostream& operator<<(std::ostream& os, const HwMonitor& _hw)
{
    os << _hw.tempC << "C " << _hw.fanP << "%";
    if (_hw.powerW)
        os << ' ' << fixed << setprecision(0) << _hw.powerW << "W";
    return os;
}

std::ostream& operator<<(std::ostream& os, const FormattedMemSize& s)
{
    static const char* suffixes[] = {"bytes", "KB", "MB", "GB"};
    double d = double(s.m_size);
    unsigned i;
    for (i = 0; i < 3; i++)
    {
        if (d < 1024.0)
            break;
        d /= 1024.0;
    }
    return os << fixed << setprecision(3) << d << ' ' << suffixes[i];
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

std::ostream& operator<<(std::ostream& os, const SolutionStats& s)
{
    os << "A" << s.getAccepts();
    auto stales = s.getAcceptedStales();
    if (stales)
        os << "+" << stales;
    auto rejects = s.getRejects();
    if (rejects)
        os << ":R" << rejects;
    auto failures = s.getFailures();
    if (failures)
        os << ":F" << failures;
    return os;
}

void Miner::setWork(WorkPackage const& _work)
{
    {

        boost::mutex::scoped_lock l(x_work);

        // Void work if this miner is paused
        if (paused())
            m_work.header = h256();
        else
        {

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
        }

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
