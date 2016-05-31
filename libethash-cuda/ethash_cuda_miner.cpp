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
/** @file ethash_cuda_miner.cpp
* @author Genoil <jw@meneer.net>
* @date 2015
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <queue>
#include <random>
#include <atomic>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include <libethash/internal.h>
#include <cuda_runtime.h>
#include "ethash_cuda_miner.h"
#include "ethash_cuda_miner_kernel_globals.h"


#define ETHASH_BYTES 32

// workaround lame platforms

#undef min
#undef max

using namespace std;

unsigned const ethash_cuda_miner::c_defaultBlockSize = 128;
unsigned const ethash_cuda_miner::c_defaultGridSize = 8192; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const ethash_cuda_miner::c_defaultNumStreams = 2;

#if defined(_WIN32)
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char* lpOutputString);
static std::atomic_flag s_logSpin = ATOMIC_FLAG_INIT;
#define ETHCUDA_LOG(_contents) \
	do \
			{ \
		std::stringstream ss; \
		ss << _contents; \
						while (s_logSpin.test_and_set(std::memory_order_acquire)) {} \
		OutputDebugStringA(ss.str().c_str()); \
		cout << ss.str() << endl << flush; \
		s_logSpin.clear(std::memory_order_release); \
			} while (false)
#else
#define ETHCUDA_LOG(_contents) cout << "[CUDA]:" << _contents << endl
#endif

ethash_cuda_miner::search_hook::~search_hook() {}

ethash_cuda_miner::ethash_cuda_miner()
{
}

std::string ethash_cuda_miner::platform_info(unsigned _deviceId)
{
	int runtime_version;
	int device_count;

	device_count = getNumDevices();

	if (device_count == 0)
		return std::string();

	CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime_version));

	// use selected default device
	int device_num = std::min<int>((int)_deviceId, device_count - 1);
	cudaDeviceProp device_props;

	CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, device_num));

	char platform[5];
	int version_major = runtime_version / 1000;
	int version_minor = (runtime_version - (version_major * 1000)) / 10;
	sprintf(platform, "%d.%d", version_major, version_minor);

	char compute[5];
	sprintf(compute, "%d.%d", device_props.major, device_props.minor);

	return "{ \"platform\": \"CUDA " + std::string(platform) + "\", \"device\": \"" + std::string(device_props.name) + "\", \"version\": \"Compute " + std::string(compute) + "\" }";
}

unsigned ethash_cuda_miner::getNumDevices()
{
	int device_count;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
	return device_count;
}

bool ethash_cuda_miner::configureGPU(
	int *	 _devices,
	unsigned _blockSize,
	unsigned _gridSize,
	unsigned _numStreams,
	unsigned _extraGPUMemory,
	unsigned _scheduleFlag,
	uint64_t _currentBlock
	)
{
	try
	{
		s_blockSize = _blockSize;
		s_gridSize = _gridSize;
		s_extraRequiredGPUMem = _extraGPUMemory;
		s_numStreams = _numStreams;
		s_scheduleFlag = _scheduleFlag;

		ETHCUDA_LOG(
			"Using grid size " << s_gridSize << ", block size " << s_blockSize << endl
			);

		// by default let's only consider the DAG of the first epoch
		uint64_t dagSize = ethash_get_datasize(_currentBlock);
		uint64_t requiredSize = dagSize + _extraGPUMemory;
		unsigned devicesCount = getNumDevices();
		for (unsigned int i = 0; i < devicesCount; i++)
		{
			
			if (_devices[i] != -1)
			{
				int deviceId = min((int)devicesCount - 1, _devices[i]);
				cudaDeviceProp props;
				CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
				if (props.totalGlobalMem >= requiredSize)
				{
					ETHCUDA_LOG(
						"Found suitable CUDA device [" << string(props.name)
						<< "] with " << props.totalGlobalMem << " bytes of GPU memory"
						);
				}
				else
				{
					ETHCUDA_LOG(
						"CUDA device " << string(props.name)
						<< " has insufficient GPU memory." << to_string(props.totalGlobalMem) <<
						" bytes of memory found < " << to_string(requiredSize) << " bytes of memory required"
						);
					return false;
				}
			}
		}
		return true;
	}
	catch (runtime_error)
	{
		return false;
	}
}

unsigned ethash_cuda_miner::s_extraRequiredGPUMem;
unsigned ethash_cuda_miner::s_blockSize = ethash_cuda_miner::c_defaultBlockSize;
unsigned ethash_cuda_miner::s_gridSize = ethash_cuda_miner::c_defaultGridSize;
unsigned ethash_cuda_miner::s_numStreams = ethash_cuda_miner::c_defaultNumStreams;
unsigned ethash_cuda_miner::s_scheduleFlag = 0;

void ethash_cuda_miner::listDevices()
{
	string outString = "\nListing CUDA devices.\nFORMAT: [deviceID] deviceName\n";
	for (unsigned int i = 0; i < getNumDevices(); i++)
	{
		cudaDeviceProp props;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

		outString += "[" + to_string(i) + "] " + string(props.name) + "\n";
		outString += "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) + "\n";
		outString += "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
	}
	ETHCUDA_LOG(outString);
}

void ethash_cuda_miner::finish()
{
	CUDA_SAFE_CALL(cudaDeviceReset());
}

bool ethash_cuda_miner::init(ethash_light_t _light, uint8_t const* _lightData, uint64_t _lightSize, unsigned _deviceId)
{
	try
	{
		int device_count = getNumDevices();

		if (device_count == 0)
			return false;

		// use selected device
		int device_num = std::min<int>((int)_deviceId, device_count - 1);

		cudaDeviceProp device_props;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, device_num));

		cout << "Using device: " << device_props.name << " (Compute " << device_props.major << "." << device_props.minor << ")" << endl;

		CUDA_SAFE_CALL(cudaSetDevice(device_num));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		m_search_buf = new volatile uint32_t *[s_numStreams];
		m_streams = new cudaStream_t[s_numStreams];

		uint64_t dagSize = ethash_get_datasize(_light->block_number);
		uint32_t dagSize128   = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(_lightSize / sizeof(node));

		// create buffer for cache
		hash64_t * light;
		CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), _lightSize));
		// copy dag to CPU.
		CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), _lightData, _lightSize, cudaMemcpyHostToDevice));

		// create buffer for dag
		hash128_t * dag;
		CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));
		// copy dag to CPU.
		//CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(dag), _dag, _dagSize, cudaMemcpyHostToDevice));

		
		// create mining buffers
		for (unsigned i = 0; i != s_numStreams; ++i)
		{
			CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], SEARCH_RESULT_BUFFER_SIZE * sizeof(uint32_t)));
			CUDA_SAFE_CALL(cudaStreamCreate(&m_streams[i]));
		}
		set_constants(dag, dagSize128, light, lightSize64);
		memset(&m_current_header, 0, sizeof(hash32_t));
		m_current_target = 0;
		m_current_nonce = 0;
		m_current_index = 0;

		m_sharedBytes = device_props.major * 100 < SHUFFLE_MIN_VER ? (64 * s_blockSize) / 8 : 0 ;


		cout << "Generating DAG for GPU #" << device_num << endl;
		ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0], device_num);

		return true;
	}
	catch (runtime_error)
	{
		return false;
	}
}

void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	bool initialize = false;
	bool exit = false;
	if (memcmp(&m_current_header, header, sizeof(hash32_t)))
	{
		m_current_header = *reinterpret_cast<hash32_t const *>(header);
		set_header(m_current_header);
		initialize = true;
	}
	if (m_current_target != target)
	{
		m_current_target = target;
		set_target(m_current_target);
		initialize = true;
	}
	if (initialize)
	{
		random_device engine;
		m_current_nonce = uniform_int_distribution<uint64_t>()(engine);
		m_current_index = 0;
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		for (unsigned int i = 0; i < s_numStreams; i++)
			m_search_buf[i][0] = 0;
	}
	uint64_t batch_size = s_gridSize * s_blockSize;
	for (; !exit; m_current_index++, m_current_nonce += batch_size)
	{
		unsigned int stream_index = m_current_index % s_numStreams;
		cudaStream_t stream = m_streams[stream_index];
		volatile uint32_t* buffer = m_search_buf[stream_index];
		uint32_t found_count = 0;
		uint64_t nonces[SEARCH_RESULT_BUFFER_SIZE - 1];
		uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;
		if (m_current_index >= s_numStreams)
		{
			CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
			found_count = buffer[0];
			if (found_count)
				buffer[0] = 0;
			for (unsigned int j = 0; j < found_count; j++)
				nonces[j] = nonce_base + buffer[j + 1];
		}
		run_ethash_search(s_gridSize, s_blockSize, m_sharedBytes, stream, buffer, m_current_nonce);
		if (m_current_index >= s_numStreams)
		{
			exit = found_count && hook.found(nonces, found_count);
			exit |= hook.searched(nonce_base, batch_size);
		}
	}
}

