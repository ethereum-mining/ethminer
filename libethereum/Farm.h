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
/** @file Miner.h
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <thread>
#include <list>
#include <atomic>
#include <libdevcore/Common.h>
#include <libdevcore/Worker.h>
#include <libethcore/Common.h>
#include "Miner.h"

namespace dev
{

namespace eth
{

/**
 * @brief A collective of Miners.
 * Miners ask for work, then submit proofs
 * @threadsafe
 */
template <class ProofOfWork>
class Farm: public FarmFace
{
public:
	/**
	 * @brief Sets the current mining mission.
	 * @param _bi The block (header) we wish to be mining.
	 */
	void setWork(BlockInfo const& _bi)
	{
		WriteGuard l(x_work);
		m_header = _bi;
		m_work.boundary = _bi.boundary();
		m_work.headerHash = _bi.headerHash(WithNonce);
		m_work.seedHash = _bi.seedHash();
		ReadGuard l(x_miners);
		for (auto const& m: miners)
			m->setWork(m_work);
	}

	/**
	 * @brief (Re)start miners for CPU only.
	 * @returns true if started properly.
	 */
	bool startCPU() { return start<ProofOfWork::CPUMiner>(); }

	/**
	 * @brief (Re)start miners for GPU only.
	 * @returns true if started properly.
	 */
	bool startGPU() { start<ProofOfWork::GPUMiner>(); }

	/**
	 * @brief Stop all mining activities.
	 */
	void stop()
	{
		WriteGuard l(x_miners);
		m_miners.clear();
	}

	/**
	 * @brief Get information on the progress of mining this work package.
	 * @return The progress with mining so far.
	 */
	MineProgress const& mineProgress() const { ReadGuard l(x_progress); return m_progress; }

protected:
	// TO BE REIMPLEMENTED BY THE SUBCLASS
	/**
	 * @brief Provides a valid header based upon that received previously with setWork().
	 * @param _bi The now-valid header.
	 * @return true if the header was good and that the Farm should pause until more work is submitted.
	 */
	virtual bool submitHeader(BlockInfo const& _bi) = 0;

private:
	/**
	 * @brief Called from a Miner to note a WorkPackage has a solution.
	 * @param _p The solution.
	 * @param _wp The WorkPackage that the Solution is for.
	 * @return true iff the solution was good (implying that mining should be .
	 */
	bool submitProof(ProofOfWork::Solution const& _p, WorkPackage const& _wp, NewMiner* _m) override
	{
		if (_wp.headerHash != m_work.headerHash)
			return false;

		ProofOfWork::assignResult(_p, m_header);
		if (submitHeader(m_header))
		{
			ReadGuard l(x_miners);
			for (std::shared_ptr<NewMiner> const& m: m_miners)
				if (m.get() != _m)
					m->pause();
			m_work.headerHash = h256();
			return true;
		}
		return false;
	}

	/**
	 * @brief Start a number of miners.
	 */
	template <class MinerType>
	bool start()
	{
		WriteGuard l(x_miners);
		if (!m_miners.empty() && !!std::dynamic_pointer_cast<MinerType>(m_miners[0]))
			return true;
		m_miners.clear();
		m_miners.reserve(MinerType::instances());
		for (unsigned i = 0; i < MinerType::instances(); ++i)
			m_miners.push_back(new MinerType(std::make_pair(this, i)));
		return true;
	}

	mutable SharedMutex x_miners;
	std::vector<std::shared_ptr<NewMiner>> m_miners;

	mutable SharedMutex x_progress;
	MineProgress m_progress;

	mutable SharedMutex x_work;
	WorkPackage m_work;
	BlockInfo m_header;
};

}
}
