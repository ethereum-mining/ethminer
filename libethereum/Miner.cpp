/*
 This file is part of cpp-ethereum.

 cpp-ethereum is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 cpp-ethereum is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
 */
/** @file Miner.cpp
 * @author Gav Wood <i@gavwood.com>
 * @author Giacomo Tazzari
 * @date 2014
 */

#include "Miner.h"

#include <libdevcore/CommonIO.h>
#include "State.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

Miner::~Miner() {}

LocalMiner::LocalMiner(MinerHost* _host, unsigned _id):
	Worker("miner-" + toString(_id)),
	m_host(_host)
{
}

void LocalMiner::setup(MinerHost* _host, unsigned _id)
{
	m_host = _host;
	setName("miner-" + toString(_id));
}

void LocalMiner::doWork()
{
	// Do some mining.
	if (m_miningStatus != Waiting && m_miningStatus != Mined)
	{
		if (m_miningStatus == Preparing)
		{
			m_host->setupState(m_mineState);
			if (m_host->force() || m_mineState.pending().size())
				m_miningStatus = Mining;
			else
				m_miningStatus = Waiting;

			{
				Guard l(x_mineInfo);
				m_mineProgress.best = (double)-1;
				m_mineProgress.hashes = 0;
				m_mineProgress.ms = 0;
			}
		}

		if (m_miningStatus == Mining)
		{
			// Mine for a while.
			MineInfo mineInfo = m_mineState.mine(100, m_host->turbo());

			{
				Guard l(x_mineInfo);
				m_mineProgress.best = min(m_mineProgress.best, mineInfo.best);
				m_mineProgress.current = mineInfo.best;
				m_mineProgress.requirement = mineInfo.requirement;
				m_mineProgress.ms += 100;
				m_mineProgress.hashes += mineInfo.hashes;
				m_mineHistory.push_back(mineInfo);
			}
			if (mineInfo.completed)
			{
				m_mineState.completeMine();
				m_host->onComplete();
				m_miningStatus = Mined;
			}
			else
				m_host->onProgressed();
		}
	}
	else
	{
		this_thread::sleep_for(chrono::milliseconds(100));
	}
}
