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
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Miner.h"
#include "State.h"
using namespace std;
using namespace eth;

Miner::Miner(MinerHost* _host, unsigned _id):
	m_host(_host),
	m_id(_id)
{
}

void Miner::start()
{
	Guard l(x_work);
	if (!m_work)
	{
		m_stop = false;
		m_work.reset(new thread([&]()
		{
			setThreadName(("miner-" + toString(m_id)).c_str());
			m_miningStatus = Preparing;
			while (!m_stop)
				work();
		}));
	}
}

void Miner::stop()
{
	Guard l(x_work);
	if (m_work)
	{
		m_stop = true;
		m_work->join();
		m_work.reset(nullptr);
	}
}

void Miner::work()
{
	// Do some mining.
	if ((m_pendingCount || m_host->force()) && m_miningStatus != Mined)
	{
		if (m_miningStatus == Preparing)
		{
			m_miningStatus = Mining;

			m_host->setupState(m_mineState);
			m_pendingCount = m_mineState.pending().size();

			{
				Guard l(x_mineInfo);
				m_mineProgress.best = (double)-1;
				m_mineProgress.hashes = 0;
				m_mineProgress.ms = 0;
			}
		}

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
	else
	{
		this_thread::sleep_for(chrono::milliseconds(100));
	}
}
