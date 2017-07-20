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
#include "CLMiner_kernel.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

unsigned CLMiner::s_workgroupSize = CLMiner::c_defaultLocalWorkSize;
unsigned CLMiner::s_initialGlobalWorkSize = CLMiner::c_defaultGlobalWorkSizeMultiplier * CLMiner::c_defaultLocalWorkSize;

constexpr size_t c_maxSearchResults = 1;

struct CLChannel: public LogChannel
{
	static const char* name() { return EthOrange " cl"; }
	static const int verbosity = 2;
	static const bool debug = false;
};
#define cllog clog(CLChannel)
#define ETHCL_LOG(_contents) cllog << _contents

namespace
{

void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

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
	Worker("cl-" + std::to_string(_ci.second))
{}

CLMiner::~CLMiner()
{
	pause();
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
		cllog << "Set work. Header" << w.headerHash << "target" << w.boundary.hex().substr(0, 12);
		if (m_minerSeed != w.seedHash)
		{
			if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
			{
				while (s_dagLoadIndex < index()) {
					this_thread::sleep_for(chrono::seconds(1));
				}
			}

			cllog << "Initialising miner with seed" << w.seedHash;
			m_minerSeed = w.seedHash;

			unsigned device = s_devices[index()] > -1 ? s_devices[index()] : index();

			EthashAux::LightType light;
			light = EthashAux::light(w.seedHash);
			bytesConstRef lightData = light->data();

			init(light->light, lightData.data(), lightData.size(), s_platformId,  device, s_workgroupSize, s_initialGlobalWorkSize);
			s_dagLoadIndex++;
		}

		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)w.boundary >> 192);

		uint64_t startNonce = 0;
		if (w.exSizeBits >= 0)
			startNonce = w.startNonce | ((uint64_t)index() << (64 - 4 - w.exSizeBits)); // this can support up to 16 devices
		else
			startNonce = randomNonce();


		search(w.headerHash.data(), upper64OfBoundary, startNonce);
	}
	catch (cl::Error const& _e)
	{
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


bool CLMiner::init(
	ethash_light_t _light,
	uint8_t const* _lightData,
	uint64_t _lightSize,
	unsigned _platformId,
	unsigned _deviceId,
	unsigned _workgroupSize,
	unsigned _initialGlobalWorkSize
)
{
	// get all platforms
	try
	{
		vector<cl::Platform> platforms = getPlatforms();
		if (platforms.empty())
			return false;

		// use selected platform
		_platformId = min<unsigned>(_platformId, platforms.size() - 1);

		string platformName = platforms[_platformId].getInfo<CL_PLATFORM_NAME>();
		ETHCL_LOG("Platform: " << platformName);

		int platformId = OPENCL_PLATFORM_UNKNOWN;
		if (platformName == "NVIDIA CUDA")
		{
			platformId = OPENCL_PLATFORM_NVIDIA;
		}
		else if (platformName == "AMD Accelerated Parallel Processing")
		{
			platformId = OPENCL_PLATFORM_AMD;
		}
		else if (platformName == "Clover")
		{
			platformId = OPENCL_PLATFORM_CLOVER;
		}

		// get GPU device of the default platform
		vector<cl::Device> devices = getDevices(platforms, _platformId);
		if (devices.empty())
		{
			ETHCL_LOG("No OpenCL devices found.");
			return false;
		}

		// use selected device
		cl::Device& device = devices[min<unsigned>(_deviceId, devices.size() - 1)];
		string device_version = device.getInfo<CL_DEVICE_VERSION>();
		ETHCL_LOG("Device:   " << device.getInfo<CL_DEVICE_NAME>() << " / " << device_version);

		string clVer = device_version.substr(7, 3);
		if (clVer == "1.0" || clVer == "1.1")
		{
			if (platformId == OPENCL_PLATFORM_CLOVER)
			{
				ETHCL_LOG("OpenCL " << clVer << " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!");
			}
			else
			{
				ETHCL_LOG("OpenCL " << clVer << " not supported - minimum required version is 1.2");
				return false;
			}
		}

		char options[256];
		int computeCapability = 0;
		if (platformId == OPENCL_PLATFORM_NVIDIA) {
			cl_uint computeCapabilityMajor;
			cl_uint computeCapabilityMinor;
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &computeCapabilityMajor, NULL);
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &computeCapabilityMinor, NULL);

			computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
			int maxregs = computeCapability >= 35 ? 72 : 63;
			sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
		}
		else {
			sprintf(options, "%s", "");
		}
		// create context
		m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
		m_queue = cl::CommandQueue(m_context, device);

		// make sure that global work size is evenly divisible by the local workgroup size
		m_workgroupSize = _workgroupSize;
		m_globalWorkSize = _initialGlobalWorkSize;
		if (m_globalWorkSize % _workgroupSize != 0)
			m_globalWorkSize = ((m_globalWorkSize / _workgroupSize) + 1) * _workgroupSize;

		uint64_t dagSize = ethash_get_datasize(_light->block_number);
		uint32_t dagSize128 = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(_lightSize / sizeof(node));

		// patch source code
		// note: CLMiner_kernel is simply ethash_cl_miner_kernel.cl compiled
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		// TODO: Just use C++ raw string literal.
		string code(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		addDefinition(code, "GROUP_SIZE", _workgroupSize);
		addDefinition(code, "DAG_SIZE", dagSize128);
		addDefinition(code, "LIGHT_SIZE", lightSize64);
		addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
		addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
		addDefinition(code, "PLATFORM", platformId);
		addDefinition(code, "COMPUTE", computeCapability);

		// create miner OpenCL program
		cl::Program::Sources sources{{std::move(code)}};
		cl::Program program(m_context, sources);
		try
		{
			program.build({device}, options);
			cllog << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		}
		catch (cl::Error const&)
		{
			cwarn << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			return false;
		}

		// create buffer for dag
		try
		{
			ETHCL_LOG("Creating cache buffer");
			m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, _lightSize);
			ETHCL_LOG("Creating DAG buffer");
			m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, dagSize);
			ETHCL_LOG("Loading kernels");
			m_searchKernel = cl::Kernel(program, "ethash_search");
			m_dagKernel = cl::Kernel(program, "ethash_calculate_dag_item");
			ETHCL_LOG("Writing cache buffer");
			m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, _lightSize, _lightData);
		}
		catch (cl::Error const& err)
		{
			ETHCL_LOG("Allocating/mapping DAG buffer failed with: " << err.what() << "(" << err.err() << "). GPU can't allocate the DAG in a single chunk. Bailing.");
			return false;
		}
		// create buffer for header
		ETHCL_LOG("Creating buffer for header.");
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, 32);

		m_searchKernel.setArg(1, m_header);
		m_searchKernel.setArg(2, m_dag);
		m_searchKernel.setArg(5, ~0u);  // Pass this to stop the compiler unrolling the loops.

		// create mining buffers
		ETHCL_LOG("Creating mining buffer");
		m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, (c_maxSearchResults + 1) * sizeof(uint32_t));

		cllog << "Generating DAG";

		uint32_t const work = (uint32_t)(dagSize / sizeof(node));
		uint32_t fullRuns = work / m_globalWorkSize;
		uint32_t const restWork = work % m_globalWorkSize;
		if (restWork > 0) fullRuns++;

		m_dagKernel.setArg(1, m_light);
		m_dagKernel.setArg(2, m_dag);
		m_dagKernel.setArg(3, ~0u);

		for (uint32_t i = 0; i < fullRuns; i++)
		{
			m_dagKernel.setArg(0, i * m_globalWorkSize);
			m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, m_globalWorkSize, _workgroupSize);
			m_queue.finish();
			cllog << "DAG" << int(100.0f * i / fullRuns) << '%';
		}

	}
	catch (cl::Error const& err)
	{
		cwarn << err.what() << "(" << err.err() << ")";
		return false;
	}
	return true;
}


void CLMiner::search(uint8_t const* header, uint64_t target, uint64_t start_nonce)
{
	// Memory for zero-ing buffers. Cannot be static because crashes on macOS.
	uint32_t const c_zero = 0;

	// Update header constant buffer.
	m_queue.enqueueWriteBuffer(m_header, CL_FALSE, 0, 32, header);
	m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);

	m_searchKernel.setArg(0, m_searchBuffer);  // Supply output buffer to kernel.
	m_searchKernel.setArg(4, target);

	while (true)
	{
		// Read results.
		// TODO: could use pinned host pointer instead.
		uint32_t results[c_maxSearchResults + 1];
		m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(results), &results);
		unsigned num_found = min<unsigned>(results[0], c_maxSearchResults);

		uint64_t nonces[c_maxSearchResults];
		for (unsigned i = 0; i != num_found; ++i)
			nonces[i] = start_nonce + results[i + 1];

		// Reset search buffer if any solution found.
		if (num_found)
			m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);

		// Increase start nonce for following kernel execution.
		start_nonce += m_globalWorkSize;

		// Run the kernel.
		m_searchKernel.setArg(3, start_nonce);
		m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);

		// Report results while the kernel is running.
		// It takes some time because ethash must be re-evaluated on CPU.
		bool exit = num_found && found(nonces, num_found);
		exit |= searched(start_nonce, m_globalWorkSize); // always report searched before exit
		if (exit)
		{
			// Make sure the last buffer write has finished --
			// it reads local variable.
			m_queue.finish();
			break;
		}
	}
}
