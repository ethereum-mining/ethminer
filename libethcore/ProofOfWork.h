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
/** @file ProofOfWork.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * ProofOfWork algorithm. Or not.
 */

#pragma once

#include <chrono>
#include <thread>
#include <cstdint>
#include <libdevcrypto/SHA3.h>
#include "Common.h"
#include "BlockInfo.h"
#include "Miner.h"

#define FAKE_DAGGER 1

class ethash_cl_miner;

namespace dev
{
namespace eth
{

struct MineInfo
{
	MineInfo() = default;
	MineInfo(bool _completed): completed(_completed) {}
	void combine(MineInfo const& _m) { requirement = std::max(requirement, _m.requirement); best = std::min(best, _m.best); hashes += _m.hashes; completed = completed || _m.completed; }
	double requirement = 0;
	double best = 1e99;
	unsigned hashes = 0;
	bool completed = false;
};

class Ethash
{

public:

struct Solution
{
	Nonce nonce;
	h256 mixHash;
};

static bool verify(BlockInfo const& _header);
static void assignResult(Solution const& _r, BlockInfo& _header) { _header.nonce = _r.nonce; _header.mixHash = _r.mixHash; }

class CPUMiner: public Miner, Worker
{
public:
	CPUMiner(ConstructionInfo const& _ci): Miner(_ci), Worker("miner" + toString(index())) {}

	static unsigned instances() { return thread::hardware_concurrency(); }

protected:
	void kickOff(WorkPackage const& _work) override
	{
		stopWorking();
		m_work = _work;
		startWorking();
	}

	void pause() override { stopWorking(); }

private:
	void workLoop() override;

	WorkPackage m_work;
	MineInfo m_info;
};

#if ETH_ETHASHCL || !ETH_TRUE

class EthashCLHook;

class GPUMiner: public Miner
{
	friend class EthashCLHook;

public:
	GPUMiner(ConstructionInfo const& _ci);

	static unsigned instances() { return 1; }

protected:
	void kickOff(WorkPackage const& _work) override;
	void pause() override;

private:
	void report(uint64_t _nonce);

	std::unique_ptr<EthashCLHook> m_hook;
	std::unique_ptr<ethash_cl_miner> m_miner;
	h256 m_minerSeed;
	WorkPackage m_lastWork;	///< Work loaded into m_miner.
	MineInfo m_info;
};

#else

using GPUMiner = CPUMiner;

#endif

};

using ProofOfWork = Ethash;

}
}
