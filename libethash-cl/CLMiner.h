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
/** @file EthashGPUMiner.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#pragma once

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_ENABLE_EXCEPTIONS 1

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

#include "CL/cl.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// apple fix
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#endif

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD     2
#define OPENCL_PLATFORM_CLOVER  3


namespace dev
{
namespace eth
{

class CLMiner: public Miner, Worker
{
public:
	/* -- default values -- */
	/// Default value of the local work size. Also known as workgroup size.
	static const unsigned c_defaultLocalWorkSize = 128;
	/// Default value of the global work size as a multiplier of the local work size
	static const unsigned c_defaultGlobalWorkSizeMultiplier = 8192;

	CLMiner(ConstructionInfo const& _ci);
	~CLMiner();

	bool found(uint64_t nonce);
	bool searched(uint32_t _count);

	static unsigned instances() { return s_numInstances > 0 ? s_numInstances : 1; }
	static unsigned getNumDevices();
	static void listDevices();
	static bool configureGPU(
		unsigned _localWorkSize,
		unsigned _globalWorkSizeMultiplier,
		unsigned _platformId,
		uint64_t _currentBlock,
		unsigned _dagLoadMode,
		unsigned _dagCreateDevice
	);
	static void setNumInstances(unsigned _instances) { s_numInstances = std::min<unsigned>(_instances, getNumDevices()); }
	static void setDevices(unsigned * _devices, unsigned _selectedDeviceCount)
	{
		for (unsigned i = 0; i < _selectedDeviceCount; i++)
		{
			s_devices[i] = _devices[i];
		}
	}

protected:
	void kickOff() override;
	void pause() override;

private:
	void workLoop() override;
	bool report(uint64_t _nonce);

	bool init(const h256& seed);

	Mutex x_hook;
	bool m_hook_abort = false;
	Notified<bool> m_hook_aborted = {true};

	cl::Context m_context;
	cl::CommandQueue m_queue;
	cl::Kernel m_searchKernel;
	cl::Kernel m_dagKernel;
	cl::Buffer m_dag;
	cl::Buffer m_light;
	cl::Buffer m_header;
	cl::Buffer m_searchBuffer;
	unsigned m_globalWorkSize = 0;
	unsigned m_workgroupSize = 0;

	/// The seed the miner was initialized with.
	/// Init with non-zero hash to distinct from the seed of epoch 0.
	h256 m_minerSeed = h256{1u};

	static unsigned s_platformId;
	static unsigned s_numInstances;
	static int s_devices[16];

	/// The local work size for the search
	static unsigned s_workgroupSize;
	/// The initial global work size for the searches
	static unsigned s_initialGlobalWorkSize;

};

}
}
