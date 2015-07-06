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
		WorkPackage() = default;

		void reset() { headerHash = h256(); }
		operator bool() const { return headerHash != h256(); }

		h256 boundary;
		h256 headerHash;	///< When h256() means "pause until notified a new work package is available".
		h256 seedHash;
	};

	static const WorkPackage NullWorkPackage;

	static std::string name();
	static unsigned revision();
	static void prep(BlockInfo const& _header, std::function<int(unsigned)> const& _f = std::function<int(unsigned)>());
	static void ensurePrecomputed(unsigned _number);
	static bool verify(BlockInfo const& _header);
	static bool preVerify(BlockInfo const& _header);
	static WorkPackage package(BlockInfo const& _header);
	static void assignResult(Solution const& _r, BlockInfo& _header) { _header.nonce = _r.nonce; _header.mixHash = _r.mixHash; }

	class CPUMiner: public Miner, Worker
	{
	public:
		CPUMiner(ConstructionInfo const& _ci): Miner(_ci), Worker("miner" + toString(index())) {}

		static unsigned instances() { return s_numInstances > 0 ? s_numInstances : std::thread::hardware_concurrency(); }
		static std::string platformInfo();
		static void listDevices() {}
		static bool configureGPU(unsigned, unsigned, unsigned, unsigned, unsigned, bool, unsigned,  boost::optional<uint64_t>) { return false; }
		static void setNumInstances(unsigned _instances) { s_numInstances = std::min<unsigned>(_instances, std::thread::hardware_concurrency()); }
	protected:
		void kickOff() override
		{
			stopWorking();
			startWorking();
		}

		void pause() override { stopWorking(); }

	private:
		void workLoop() override;
		static unsigned s_numInstances;
	};

#if ETH_ETHASHCL || !ETH_TRUE
	class GPUMiner: public Miner, Worker
	{
		friend class dev::eth::EthashCLHook;

	public:
		GPUMiner(ConstructionInfo const& _ci);
		~GPUMiner();

		static unsigned instances() { return s_numInstances > 0 ? s_numInstances : 1; }
		static std::string platformInfo();
		static unsigned getNumDevices();
		static void listDevices();
		static bool configureGPU(
			unsigned _localWorkSize,
			unsigned _globalWorkSizeMultiplier,
			unsigned _msPerBatch,
			unsigned _platformId,
			unsigned _deviceId,
			bool _allowCPU,
			unsigned _extraGPUMemory,
			boost::optional<uint64_t> _currentBlock
		);
		static void setNumInstances(unsigned _instances) { s_numInstances = std::min<unsigned>(_instances, getNumDevices()); }

	protected:
		void kickOff() override;
		void pause() override;

	private:
		void workLoop() override;
		bool report(uint64_t _nonce);

		using Miner::accumulateHashes;

		EthashCLHook* m_hook = nullptr;
		ethash_cl_miner* m_miner = nullptr;

		h256 m_minerSeed;		///< Last seed in m_miner
		static unsigned s_platformId;
		static unsigned s_deviceId;
		static unsigned s_numInstances;
	};
#else
	using GPUMiner = CPUMiner;
#endif
	/// Default value of the local work size. Also known as workgroup size.
	static const unsigned defaultLocalWorkSize;
	/// Default value of the global work size as a multiplier of the local work size
	static const unsigned defaultGlobalWorkSizeMultiplier;
	/// Default value of the milliseconds per global work size (per batch)
	static const unsigned defaultMSPerBatch;
};

}
}
