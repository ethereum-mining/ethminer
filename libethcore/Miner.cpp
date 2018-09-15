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

bool Miner::s_noeval = false;

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

}  // namespace eth
}  // namespace dev
