/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ethash_cl_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <queue>
#include <vector>
#include <random>
#include <random>
#include <atomic>
#include <sstream>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include <libethash/internal.h>
#include "ethash_cl_miner.h"
#include "ethash_cl_miner_kernel.h"

#define ETHASH_BYTES 32

// workaround lame platforms
#if !CL_VERSION_1_2
#define CL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE
#define CL_MEM_HOST_READ_ONLY 0
#endif

#undef min
#undef max

using namespace std;

unsigned const ethash_cl_miner::c_defaultLocalWorkSize = 64;
unsigned const ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier = 4096; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const ethash_cl_miner::c_defaultMSPerBatch = 0;

// TODO: If at any point we can use libdevcore in here then we should switch to using a LogChannel
#if defined(_WIN32)
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char* lpOutputString);
static std::atomic_flag s_logSpin = ATOMIC_FLAG_INIT;
#define ETHCL_LOG(_contents) \
	do \
	{ \
		std::stringstream ss; \
		ss << _contents; \
		while (s_logSpin.test_and_set(std::memory_order_acquire)) {} \
		OutputDebugStringA(ss.str().c_str()); \
		cerr << ss.str() << endl << flush; \
		s_logSpin.clear(std::memory_order_release); \
	} while (false)
#else
#define ETHCL_LOG(_contents) cout << "[OPENCL]:" << _contents << endl
#endif
// Types of OpenCL devices we are interested in
#define ETHCL_QUERIED_DEVICE_TYPES (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR)

static void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

ethash_cl_miner::search_hook::~search_hook() {}

ethash_cl_miner::ethash_cl_miner()
:	m_openclOnePointOne()
{
}

ethash_cl_miner::~ethash_cl_miner()
{
	finish();
}

std::vector<cl::Platform> ethash_cl_miner::getPlatforms()
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
			ETHCL_LOG("No OpenCL platforms found");
		else
#endif
			throw err;
	}
	return platforms;
}

string ethash_cl_miner::platform_info(unsigned _platformId, unsigned _deviceId)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return {};
	// get GPU device of the selected platform
	unsigned platform_num = min<unsigned>(_platformId, platforms.size() - 1);
	vector<cl::Device> devices = getDevices(platforms, _platformId);
	if (devices.empty())
	{
		ETHCL_LOG("No OpenCL devices found.");
		return {};
	}

	// use selected default device
	unsigned device_num = min<unsigned>(_deviceId, devices.size() - 1);
	cl::Device& device = devices[device_num];
	string device_version = device.getInfo<CL_DEVICE_VERSION>();

	return "{ \"platform\": \"" + platforms[platform_num].getInfo<CL_PLATFORM_NAME>() + "\", \"device\": \"" + device.getInfo<CL_DEVICE_NAME>() + "\", \"version\": \"" + device_version + "\" }";
}

std::vector<cl::Device> ethash_cl_miner::getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	unsigned platform_num = min<unsigned>(_platformId, _platforms.size() - 1);
	try
	{
		_platforms[platform_num].getDevices(
			s_allowCPU ? CL_DEVICE_TYPE_ALL : ETHCL_QUERIED_DEVICE_TYPES,
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

unsigned ethash_cl_miner::getNumPlatforms()
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return 0;
	return platforms.size();
}

unsigned ethash_cl_miner::getNumDevices(unsigned _platformId)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return 0;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	if (devices.empty())
	{
		ETHCL_LOG("No OpenCL devices found.");
		return 0;
	}
	return devices.size();
}

bool ethash_cl_miner::configureGPU(
	unsigned _platformId,
	unsigned _localWorkSize,
	unsigned _globalWorkSize,
	unsigned _msPerBatch,
	bool _allowCPU,
	unsigned _extraGPUMemory,
	uint64_t _currentBlock
)
{
	s_workgroupSize = _localWorkSize;
	s_initialGlobalWorkSize = _globalWorkSize;
	s_msPerBatch = _msPerBatch;
	s_allowCPU = _allowCPU;
	s_extraRequiredGPUMem = _extraGPUMemory;
	// by default let's only consider the DAG of the first epoch
	uint64_t dagSize = ethash_get_datasize(_currentBlock);
	uint64_t requiredSize =  dagSize + _extraGPUMemory;
	return searchForAllDevices(_platformId, [&requiredSize](cl::Device const& _device) -> bool
		{
			cl_ulong result;
			_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
			if (result >= requiredSize)
			{
				ETHCL_LOG(
					"Found suitable OpenCL device [" << _device.getInfo<CL_DEVICE_NAME>()
					<< "] with " << result << " bytes of GPU memory"
				);
				return true;
			}

			ETHCL_LOG(
				"OpenCL device " << _device.getInfo<CL_DEVICE_NAME>()
				<< " has insufficient GPU memory." << result <<
				" bytes of memory found < " << requiredSize << " bytes of memory required"
			);
			return false;
		}
	);
}

bool ethash_cl_miner::s_allowCPU = false;
unsigned ethash_cl_miner::s_extraRequiredGPUMem;
unsigned ethash_cl_miner::s_msPerBatch = ethash_cl_miner::c_defaultMSPerBatch;
unsigned ethash_cl_miner::s_workgroupSize = ethash_cl_miner::c_defaultLocalWorkSize;
unsigned ethash_cl_miner::s_initialGlobalWorkSize = ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier * ethash_cl_miner::c_defaultLocalWorkSize;

bool ethash_cl_miner::searchForAllDevices(function<bool(cl::Device const&)> _callback)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return false;
	for (unsigned i = 0; i < platforms.size(); ++i)
		if (searchForAllDevices(i, _callback))
			return true;

	return false;
}

bool ethash_cl_miner::searchForAllDevices(unsigned _platformId, function<bool(cl::Device const&)> _callback)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return false;
	if (_platformId >= platforms.size())
		return false;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	for (cl::Device const& device: devices)
		if (_callback(device))
			return true;

	return false;
}

void ethash_cl_miner::doForAllDevices(function<void(cl::Device const&)> _callback)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return;
	for (unsigned i = 0; i < platforms.size(); ++i)
		doForAllDevices(i, _callback);
}

void ethash_cl_miner::doForAllDevices(unsigned _platformId, function<void(cl::Device const&)> _callback)
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return;
	if (_platformId >= platforms.size())
		return;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	for (cl::Device const& device: devices)
		_callback(device);
}

void ethash_cl_miner::listDevices()
{
	string outString ="\nListing OpenCL devices.\nFORMAT: [deviceID] deviceName\n";
	unsigned int i = 0;
	doForAllDevices([&outString, &i](cl::Device const _device)
		{
			outString += "[" + to_string(i) + "] " + _device.getInfo<CL_DEVICE_NAME>() + "\n";
			outString += "\tCL_DEVICE_TYPE: ";
			switch (_device.getInfo<CL_DEVICE_TYPE>())
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
			outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " + to_string(_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " + to_string(_device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " + to_string(_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
			++i;
		}
	);
	ETHCL_LOG(outString);
}

void ethash_cl_miner::finish()
{
	if (m_queue())
		m_queue.finish();
}

bool ethash_cl_miner::init(
	uint8_t const* _dag,
	uint64_t _dagSize,
	unsigned _platformId,
	unsigned _deviceId
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
		ETHCL_LOG("Using platform: " << platforms[_platformId].getInfo<CL_PLATFORM_NAME>().c_str());

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
		ETHCL_LOG("Using device: " << device.getInfo<CL_DEVICE_NAME>().c_str() << "(" << device_version.c_str() << ")");

		if (strncmp("OpenCL 1.0", device_version.c_str(), 10) == 0)
		{
			ETHCL_LOG("OpenCL 1.0 is not supported.");
			return false;
		}
		if (strncmp("OpenCL 1.1", device_version.c_str(), 10) == 0)
			m_openclOnePointOne = true;

		// create context
		m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
		m_queue = cl::CommandQueue(m_context, device);

		// make sure that global work size is evenly divisible by the local workgroup size
		m_globalWorkSize = s_initialGlobalWorkSize;
		if (m_globalWorkSize % s_workgroupSize != 0)
			m_globalWorkSize = ((m_globalWorkSize / s_workgroupSize) + 1) * s_workgroupSize;
		// remember the device's address bits
		m_deviceBits = device.getInfo<CL_DEVICE_ADDRESS_BITS>();
		// make sure first step of global work size adjustment is large enough
		m_stepWorkSizeAdjust = pow(2, m_deviceBits / 2 + 1);

		// patch source code
		// note: ETHASH_CL_MINER_KERNEL is simply ethash_cl_miner_kernel.cl compiled
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		string code(ETHASH_CL_MINER_KERNEL, ETHASH_CL_MINER_KERNEL + ETHASH_CL_MINER_KERNEL_SIZE);
		addDefinition(code, "GROUP_SIZE", s_workgroupSize);
		addDefinition(code, "DAG_SIZE", (unsigned)(_dagSize / ETHASH_MIX_BYTES));
		addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
		addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
		//debugf("%s", code.c_str());

		// create miner OpenCL program
		cl::Program::Sources sources;
		sources.push_back({ code.c_str(), code.size() });

		cl::Program program(m_context, sources);
		try
		{
			program.build({ device });
			ETHCL_LOG("Printing program log");
			ETHCL_LOG(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
		}
		catch (cl::Error const&)
		{
			ETHCL_LOG(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
			return false;
		}

		// create buffer for dag
		try
		{
			m_dagChunksCount = 1;
			ETHCL_LOG("Creating one big buffer for the DAG");
			m_dagChunks.push_back(cl::Buffer(m_context, CL_MEM_READ_ONLY, _dagSize));
			ETHCL_LOG("Loading single big chunk kernels");
			m_hashKernel = cl::Kernel(program, "ethash_hash");
			m_searchKernel = cl::Kernel(program, "ethash_search");
			ETHCL_LOG("Mapping one big chunk.");
			m_queue.enqueueWriteBuffer(m_dagChunks[0], CL_TRUE, 0, _dagSize, _dag);
		}
		catch (cl::Error const& err)
		{
			ETHCL_LOG("Allocating/mapping single buffer failed with: " << err.what() << "(" << err.err() << "). GPU can't allocate the DAG in a single chunk. Bailing.");
			return false;
#if 0		// Disabling chunking for release since it seems not to work. Never manages to mine a block. TODO: Fix when time is found.
			int errCode = err.err();
			if (errCode != CL_INVALID_BUFFER_SIZE || errCode != CL_MEM_OBJECT_ALLOCATION_FAILURE)
				ETHCL_LOG("Allocating/mapping single buffer failed with: " << err.what() << "(" << errCode << ")");
			cl_ulong result;
			// if we fail midway on the try above make sure we start clean
			m_dagChunks.clear();
			device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &result);
			ETHCL_LOG(
				"Failed to allocate 1 big chunk. Max allocateable memory is "
				<< result << ". Trying to allocate 4 chunks."
			);
			// The OpenCL kernel has a hard coded number of 4 chunks at the moment
			m_dagChunksCount = 4;
			for (unsigned i = 0; i < m_dagChunksCount; i++)
			{
				// TODO Note: If we ever change to _dagChunksNum other than 4, then the size would need recalculation
				ETHCL_LOG("Creating buffer for chunk " << i);
				m_dagChunks.push_back(cl::Buffer(
					m_context,
					CL_MEM_READ_ONLY,
					(i == 3) ? (_dagSize - 3 * ((_dagSize >> 9) << 7)) : (_dagSize >> 9) << 7
				));
			}
			ETHCL_LOG("Loading chunk kernels");
			m_hashKernel = cl::Kernel(program, "ethash_hash_chunks");
			m_searchKernel = cl::Kernel(program, "ethash_search_chunks");
			// TODO Note: If we ever change to _dagChunksNum other than 4, then the size would need recalculation
			void* dag_ptr[4];
			for (unsigned i = 0; i < m_dagChunksCount; i++)
			{
				ETHCL_LOG("Mapping chunk " << i);
				dag_ptr[i] = m_queue.enqueueMapBuffer(m_dagChunks[i], true, m_openclOnePointOne ? CL_MAP_WRITE : CL_MAP_WRITE_INVALIDATE_REGION, 0, (i == 3) ? (_dagSize - 3 * ((_dagSize >> 9) << 7)) : (_dagSize >> 9) << 7);
			}
			for (unsigned i = 0; i < m_dagChunksCount; i++)
			{
				memcpy(dag_ptr[i], (char *)_dag + i*((_dagSize >> 9) << 7), (i == 3) ? (_dagSize - 3 * ((_dagSize >> 9) << 7)) : (_dagSize >> 9) << 7);
				m_queue.enqueueUnmapMemObject(m_dagChunks[i], dag_ptr[i]);
			}
#endif
		}
		// create buffer for header
		ETHCL_LOG("Creating buffer for header.");
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, 32);

		// create mining buffers
		for (unsigned i = 0; i != c_bufferCount; ++i)
		{
			ETHCL_LOG("Creating mining buffer " << i);
			m_hashBuffer[i] = cl::Buffer(m_context, CL_MEM_WRITE_ONLY | (!m_openclOnePointOne ? CL_MEM_HOST_READ_ONLY : 0), 32 * c_hashBatchSize);
			m_searchBuffer[i] = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, (c_maxSearchResults + 1) * sizeof(uint32_t));
		}
	}
	catch (cl::Error const& err)
	{
		ETHCL_LOG(err.what() << "(" << err.err() << ")");
		return false;
	}
	return true;
}

void ethash_cl_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	try
	{
		struct pending_batch
		{
			uint64_t start_nonce;
			unsigned buf;
		};
		queue<pending_batch> pending;

		// this can't be a static because in MacOSX OpenCL implementation a segfault occurs when a static is passed to OpenCL functions
		uint32_t const c_zero = 0;

		// update header constant buffer
		m_queue.enqueueWriteBuffer(m_header, false, 0, 32, header);
		for (unsigned i = 0; i != c_bufferCount; ++i)
			m_queue.enqueueWriteBuffer(m_searchBuffer[i], false, 0, 4, &c_zero);

#if CL_VERSION_1_2 && 0
		cl::Event pre_return_event;
		if (!m_opencl_1_1)
			m_queue.enqueueBarrierWithWaitList(NULL, &pre_return_event);
		else
#endif
			m_queue.finish();

		unsigned argPos = 2;
		m_searchKernel.setArg(1, m_header);
		for (unsigned i = 0; i < m_dagChunksCount; ++i, ++argPos)
			m_searchKernel.setArg(argPos, m_dagChunks[i]);
		// pass these to stop the compiler unrolling the loops
		m_searchKernel.setArg(argPos + 1, target);
		m_searchKernel.setArg(argPos + 2, ~0u);

		unsigned buf = 0;
		random_device engine;
		uint64_t start_nonce = uniform_int_distribution<uint64_t>()(engine);
		for (;; start_nonce += m_globalWorkSize)
		{
			auto t = chrono::high_resolution_clock::now();
			// supply output buffer to kernel
			m_searchKernel.setArg(0, m_searchBuffer[buf]);
			if (m_dagChunksCount == 1)
				m_searchKernel.setArg(3, start_nonce);
			else
				m_searchKernel.setArg(6, start_nonce);

			// execute it!
			m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, m_globalWorkSize, s_workgroupSize);

			pending.push({ start_nonce, buf });
			buf = (buf + 1) % c_bufferCount;

			// read results
			if (pending.size() == c_bufferCount)
			{
				pending_batch const& batch = pending.front();

				// could use pinned host pointer instead
				uint32_t* results = (uint32_t*)m_queue.enqueueMapBuffer(m_searchBuffer[batch.buf], true, CL_MAP_READ, 0, (1 + c_maxSearchResults) * sizeof(uint32_t));
				unsigned num_found = min<unsigned>(results[0], c_maxSearchResults);

				uint64_t nonces[c_maxSearchResults];
				for (unsigned i = 0; i != num_found; ++i)
					nonces[i] = batch.start_nonce + results[i + 1];

				m_queue.enqueueUnmapMemObject(m_searchBuffer[batch.buf], results);
				bool exit = num_found && hook.found(nonces, num_found);
				exit |= hook.searched(batch.start_nonce, m_globalWorkSize); // always report searched before exit
				if (exit)
					break;

				// reset search buffer if we're still going
				if (num_found)
					m_queue.enqueueWriteBuffer(m_searchBuffer[batch.buf], true, 0, 4, &c_zero);

				pending.pop();
			}

			// adjust global work size depending on last search time
			if (s_msPerBatch)
			{
				// Global work size must be:
				//  - less than or equal to 2 ^ DEVICE_BITS - 1
				//  - divisible by lobal work size (workgroup size)
				auto d = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t);
				if (d != chrono::milliseconds(0)) // if duration is zero, we did not get in the actual searh/or search not finished
				{
					if (d > chrono::milliseconds(s_msPerBatch * 10 / 9))
					{
						// Divide the step by 2 when adjustment way change
						if (m_wayWorkSizeAdjust > -1)
							m_stepWorkSizeAdjust = max<unsigned>(1, m_stepWorkSizeAdjust / 2);
						m_wayWorkSizeAdjust = -1;
						// cerr << "m_stepWorkSizeAdjust: " << m_stepWorkSizeAdjust << ", m_wayWorkSizeAdjust: " << m_wayWorkSizeAdjust << endl;

						// cerr << "Batch of " << m_globalWorkSize << " took " << chrono::duration_cast<chrono::milliseconds>(d).count() << " ms, >> " << s_msPerBatch << " ms." << endl;
						m_globalWorkSize = max<unsigned>(128, m_globalWorkSize - m_stepWorkSizeAdjust);
						// cerr << "New global work size" << m_globalWorkSize << endl;
					}
					else if (d < chrono::milliseconds(s_msPerBatch * 9 / 10))
					{
						// Divide the step by 2 when adjustment way change
						if (m_wayWorkSizeAdjust < 1)
							m_stepWorkSizeAdjust = max<unsigned>(1, m_stepWorkSizeAdjust / 2);
						m_wayWorkSizeAdjust = 1;
						// cerr << "m_stepWorkSizeAdjust: " << m_stepWorkSizeAdjust << ", m_wayWorkSizeAdjust: " << m_wayWorkSizeAdjust << endl;

						// cerr << "Batch of " << m_globalWorkSize << " took " << chrono::duration_cast<chrono::milliseconds>(d).count() << " ms, << " << s_msPerBatch << " ms." << endl;
						m_globalWorkSize = min<unsigned>(pow(2, m_deviceBits) - 1, m_globalWorkSize + m_stepWorkSizeAdjust);
						// Global work size should never be less than the workgroup size
						m_globalWorkSize = max<unsigned>(s_workgroupSize,  m_globalWorkSize);
						// cerr << "New global work size" << m_globalWorkSize << endl;
					}
				}
			}
		}

		// not safe to return until this is ready
#if CL_VERSION_1_2 && 0
		if (!m_opencl_1_1)
			pre_return_event.wait();
#endif
	}
	catch (cl::Error const& err)
	{
		ETHCL_LOG(err.what() << "(" << err.err() << ")");
	}
}
