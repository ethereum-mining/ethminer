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
/** @file Ethash.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * A proof of work algorithm.
 */

#pragma once

#include <chrono>
#include <thread>
#include <cstdint>
#include <libdevcore/CommonIO.h>
#include "Common.h"
#include "BlockInfo.h"
#include "Miner.h"

class ethash_cl_miner;

namespace dev
{
namespace eth
{

class EthashCLHook;

class Ethash
{
public:
	using Miner = GenericMiner<Ethash>;

	struct Solution
	{
		Nonce nonce;
		h256 mixHash;
	};

	struct Result
	{
		h256 value;
		h256 mixHash;
	};

	struct WorkPackage
	{
		h256 boundary;
		h256 headerHash;	///< When h256() means "pause until notified a new work package is available".
		h256 seedHash;
	};

	static const WorkPackage NullWorkPackage;

	static std::string name();
	static unsigned revision();
	static bool verify(BlockInfo const& _header);
	static bool preVerify(BlockInfo const& _header);
	static void assignResult(Solution const& _r, BlockInfo& _header) { _header.nonce = _r.nonce; _header.mixHash = _r.mixHash; }
	static void prep(BlockInfo const& _header);

	class CPUMiner: public Miner, Worker
	{
	public:
		CPUMiner(ConstructionInfo const& _ci): Miner(_ci), Worker("miner" + toString(index())) {}

		static unsigned instances() { return std::thread::hardware_concurrency(); }

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
	class GPUMiner: public Miner
	{
		friend class dev::eth::EthashCLHook;

	public:
		GPUMiner(ConstructionInfo const& _ci);

		static unsigned instances() { return 1; }

	protected:
		void kickOff(WorkPackage const& _work) override;
		void pause() override;

	private:
		bool report(uint64_t _nonce);

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
using Solution = Ethash::Solution;

}
}
