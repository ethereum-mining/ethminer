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

bool Miner::s_exit = false;

bool Miner::s_noeval = false;

std::ostream& operator<<(std::ostream& os, HwMonitor _hw)
{
    os << _hw.tempC << "C " << _hw.fanP << "%";
    if (_hw.powerW)
        os << ' ' << fixed << setprecision(0) << _hw.powerW << "W";
    return os;
}

std::ostream& operator<<(std::ostream& os, FormattedMemSize s)
{
    static const char* suffixes[] = {"bytes", "KB", "MB", "GB"};
    double d = s.m_size;
    unsigned i;
    for (i = 0; i < 3; i++)
    {
        if (d < 1024.0)
            break;
        d /= 1024.0;
    }
    return os << fixed << setprecision(3) << d << ' ' << suffixes[i];
}

std::ostream& operator<<(std::ostream& _out, WorkingProgress _p)
{
    float mh = _p.rate() / 1000000.0f;
    _out << "Speed " << EthTealBold << std::fixed << std::setprecision(2) << mh << EthReset
         << " Mh/s";

    for (size_t i = 0; i < _p.minersHashes.size(); ++i)
    {
        mh = _p.minerRate(_p.minersHashes[i]) / 1000000.0f;

        if (_p.miningIsPaused.size() == _p.minersHashes.size())
        {
            // red color if mining is paused on this gpu
            if (_p.miningIsPaused[i])
            {
                _out << EthRed;
            }
        }

        _out << " gpu" << i << " " << EthTeal << std::fixed << std::setprecision(2) << mh
             << EthReset;
        if (_p.minerMonitors.size() == _p.minersHashes.size())
            _out << " " << EthTeal << _p.minerMonitors[i] << EthReset;
    }

    return _out;
}

std::ostream& operator<<(std::ostream& os, SolutionStats s)
{
    os << "[A" << s.getAccepts();
    if (s.getAcceptedStales())
        os << "+" << s.getAcceptedStales();
    if (s.getRejects())
        os << ":R" << s.getRejects();
    if (s.getFailures())
        os << ":F" << s.getFailures();
    return os << "]";
}

}  // namespace eth
}  // namespace dev
