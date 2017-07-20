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
/** @file EthashGPUMiner.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#include "CLMiner.h"
#include <libethash/internal.h>
#include "ethash_cl_miner.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

unsigned CLMiner::s_workgroupSize = CLMiner::c_defaultLocalWorkSize;
unsigned CLMiner::s_initialGlobalWorkSize = CLMiner::c_defaultGlobalWorkSizeMultiplier * CLMiner::c_defaultLocalWorkSize;

// FIXME: Make local
std::vector<cl::Platform> getPlatforms()
{
	vector<cl::Platform> platforms;
	try
	{
		cl::Platform::get(&platforms);
	}
	catch(cl::Error const& err)
	{
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			cwarn << "No OpenCL platforms found";
		else
#endif
			throw err;
	}
	return platforms;
}

// FIXME: Make local
std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	unsigned platform_num = min<unsigned>(_platformId, _platforms.size() - 1);
	try
	{
		_platforms[platform_num].getDevices(
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
			&devices
		);
	}
	catch (cl::Error const& err)
	{
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

bool CLMiner::found(uint64_t const* _nonces, uint32_t _count)
{
	for (uint32_t i = 0; i < _count; ++i)
		if (report(_nonces[i]))
			return (m_hook_aborted = true);
	return shouldStop();
}

bool CLMiner::searched(uint64_t _startNonce, uint32_t _count)
{
	(void) _startNonce;
	UniqueGuard l(x_hook);
	accumulateHashes(_count);
	if (m_hook_abort || shouldStop())
		return (m_hook_aborted = true);
	return false;
}

}
}

unsigned CLMiner::s_platformId = 0;
unsigned CLMiner::s_numInstances = 0;
int CLMiner::s_devices[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

CLMiner::CLMiner(ConstructionInfo const& _ci):
	Miner(_ci),
	Worker("openclminer" + toString(index()))
{
}

CLMiner::~CLMiner()
{
	pause();
	delete m_miner;
}

bool CLMiner::report(uint64_t _nonce)
{
	WorkPackage w = work();  // Copy work package to avoid repeated mutex lock.
	Result r = EthashAux::eval(w.seedHash, w.headerHash, _nonce);
	if (r.value < w.boundary)
		return submitProof(Solution{_nonce, r.mixHash, w.headerHash, w.seedHash, w.boundary});
	return false;
}

void CLMiner::kickOff()
{
	{
		UniqueGuard l(x_hook);
		m_hook_aborted = m_hook_abort = false;
	}
	startWorking();
}

namespace
{
uint64_t randomNonce()
{
	static std::mt19937_64 s_gen(std::random_device{}());
	return std::uniform_int_distribution<uint64_t>{}(s_gen);
}
}

void CLMiner::workLoop()
{
	// take local copy of work since it may end up being overwritten by kickOff/pause.
	try {
		WorkPackage w = work();
		cnote << "set work; seed: " << "#" + w.seedHash.hex().substr(0, 8) + ", target: " << "#" + w.boundary.hex().substr(0, 12);
		if (!m_miner || m_minerSeed != w.seedHash)
		{
			if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
			{
				while (s_dagLoadIndex < index()) {
					this_thread::sleep_for(chrono::seconds(1));
				}
			}

			cnote << "Initialising miner...";
			m_minerSeed = w.seedHash;

			delete m_miner;
			m_miner = new ethash_cl_miner;

			unsigned device = s_devices[index()] > -1 ? s_devices[index()] : index();

			EthashAux::LightType light;
			light = EthashAux::light(w.seedHash);
			bytesConstRef lightData = light->data();

			m_miner->init(light->light, lightData.data(), lightData.size(), s_platformId,  device, s_workgroupSize, s_initialGlobalWorkSize);
			s_dagLoadIndex++;
		}

		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)w.boundary >> 192);

		uint64_t startNonce = 0;
		if (w.exSizeBits >= 0)
			startNonce = w.startNonce | ((uint64_t)index() << (64 - 4 - w.exSizeBits)); // this can support up to 16 devices
		else
			startNonce = randomNonce();
		m_miner->search(w.headerHash.data(), upper64OfBoundary, *this, startNonce);
	}
	catch (cl::Error const& _e)
	{
		delete m_miner;
		m_miner = nullptr;
		cwarn << "Error GPU mining: " << _e.what() << "(" << _e.err() << ")";
	}
}

void CLMiner::pause()
{
	{
		UniqueGuard l(x_hook);
		if (m_hook_aborted)
			return;

		m_hook_abort = true;
	}
	// m_abort is true so now searched()/found() will return true to abort the search.
	// we hang around on this thread waiting for them to point out that they have aborted since
	// otherwise we may end up deleting this object prior to searched()/found() being called.
	m_hook_aborted.wait(true);
	stopWorking();
}

unsigned CLMiner::getNumDevices()
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return 0;

	vector<cl::Device> devices = getDevices(platforms, s_platformId);
	if (devices.empty())
	{
		cwarn << "No OpenCL devices found.";
		return 0;
	}
	return devices.size();
}

void CLMiner::listDevices()
{
	string outString ="\nListing OpenCL devices.\nFORMAT: [deviceID] deviceName\n";
	unsigned int i = 0;

	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return;
	for (unsigned j = 0; j < platforms.size(); ++j)
	{
		vector<cl::Device> devices = getDevices(platforms, j);
		for (auto const& device: devices)
		{
			outString += "[" + to_string(i) + "] " + device.getInfo<CL_DEVICE_NAME>() + "\n";
			outString += "\tCL_DEVICE_TYPE: ";
			switch (device.getInfo<CL_DEVICE_TYPE>())
			{
			case CL_DEVICE_TYPE_CPU:
				outString += "CPU\n";
				break;
			case CL_DEVICE_TYPE_GPU:
				outString += "GPU\n";
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				outString += "ACCELERATOR\n";
				break;
			default:
				outString += "DEFAULT\n";
				break;
			}
			outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " + to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
			++i;
		}
	}
	std::cout << outString;
}

bool CLMiner::configureGPU(
	unsigned _localWorkSize,
	unsigned _globalWorkSizeMultiplier,
	unsigned _platformId,
	uint64_t _currentBlock,
	unsigned _dagLoadMode,
	unsigned _dagCreateDevice
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;

	s_platformId = _platformId;

	_localWorkSize = ((_localWorkSize + 7) / 8) * 8;
	s_workgroupSize = _localWorkSize;
	s_initialGlobalWorkSize = _globalWorkSizeMultiplier * _localWorkSize;

	uint64_t dagSize = ethash_get_datasize(_currentBlock);

	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return false;
	if (_platformId >= platforms.size())
		return false;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	for (auto const& device: devices)
	{
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result >= dagSize)
		{
			cnote <<
				"Found suitable OpenCL device [" << device.getInfo<CL_DEVICE_NAME>()
												 << "] with " << result << " bytes of GPU memory";
			return true;
		}

		cnote <<
			"OpenCL device " << device.getInfo<CL_DEVICE_NAME>()
							 << " has insufficient GPU memory." << result <<
							 " bytes of memory found < " << dagSize << " bytes of memory required";
	}

	cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl;
	return false;
}
