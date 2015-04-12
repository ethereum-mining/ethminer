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
struct ethash_cl_search_hook;

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

class EthashCLHook;

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

class GPUMiner: public NewMiner
{
public:
	GPUMiner(ConstructionInfo const& _ci): NewMiner(_ci)
	{

	}

	static unsigned instances() { return 1; }

	std::pair<MineInfo, Solution> mine(BlockInfo const& _header, unsigned _msTimeout = 100, bool _continue = true) override;
	unsigned defaultTimeout() const override { return 500; }

protected:
	Nonce m_last;
	BlockInfo m_lastHeader;
	Nonce m_mined;
	std::unique_ptr<ethash_cl_miner> m_miner;
	std::unique_ptr<EthashCLHook> m_hook;
};

#else

using GPUMiner = CPUMiner;

#endif

};

using ProofOfWork = Ethash;

}
}
