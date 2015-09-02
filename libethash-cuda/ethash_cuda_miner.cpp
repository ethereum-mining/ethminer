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
#if !CL_VERSION_1_2
#define CL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE
#define CL_MEM_HOST_READ_ONLY 0
#endif

#undef min
#undef max

using namespace std;

unsigned const ethash_cuda_miner::c_defaultBlockSize = 128;
unsigned const ethash_cuda_miner::c_defaultGridSize = 2048; // * CL_DEFAULT_LOCAL_WORK_SIZE
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
		cerr << ss.str() << endl << flush; \
		s_logSpin.clear(std::memory_order_release); \
			} while (false)
#else
#define ETHCUDA_LOG(_contents) cout << "[CUDA]:" << _contents << endl
#endif

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
		}                                                                 \
} while (0)

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
	bool	 _highcpu,
	uint64_t _currentBlock
	)
{
	s_blockSize = _blockSize;
	s_gridSize = _gridSize;
	s_extraRequiredGPUMem = _extraGPUMemory;
	s_numStreams = _numStreams;
	s_highCPU = _highcpu;

	// by default let's only consider the DAG of the first epoch
	uint64_t dagSize = ethash_get_datasize(_currentBlock);
	uint64_t requiredSize = dagSize + _extraGPUMemory;
	for (unsigned int i = 0; i < getNumDevices(); i++)
	{
		if (_devices[i] != -1) 
		{
			cudaDeviceProp props;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, _devices[i]));
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

unsigned ethash_cuda_miner::s_extraRequiredGPUMem;
unsigned ethash_cuda_miner::s_blockSize = ethash_cuda_miner::c_defaultBlockSize;
unsigned ethash_cuda_miner::s_gridSize = ethash_cuda_miner::c_defaultGridSize;
unsigned ethash_cuda_miner::s_numStreams = ethash_cuda_miner::c_defaultNumStreams;
bool ethash_cuda_miner::s_highCPU = false;

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
	for (unsigned i = 0; i != s_numStreams; i++) {
		cudaStreamDestroy(m_streams[i]);
		m_streams[i] = 0;
	}
	cudaDeviceReset();
}

bool ethash_cuda_miner::init(uint8_t const* _dag, uint64_t _dagSize, unsigned _deviceId)
{
	int device_count = getNumDevices();

	if (device_count == 0)
		return false;

	// use selected device
	int device_num = std::min<int>((int)_deviceId, device_count - 1);
	
	cudaDeviceProp device_props;
	if (cudaGetDeviceProperties(&device_props, device_num) == cudaErrorInvalidDevice)
	{
		cout << cudaGetErrorString(cudaErrorInvalidDevice) << endl;
		return false;
	}

	cout << "Using device: " << device_props.name << "(" << device_props.major << "." << device_props.minor << ")" << endl;

	cudaError_t r = cudaSetDevice(device_num);
	if (r != cudaSuccess)
	{
		cout << cudaGetErrorString(r) << endl;
		return false;
	}
	cudaDeviceReset();
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	m_hash_buf	 = new void *[s_numStreams];
	m_search_buf = new uint32_t *[s_numStreams];
	m_streams = new cudaStream_t[s_numStreams];

	// patch source code
	cudaError result;

	uint32_t dagSize128 = (unsigned)(_dagSize / ETHASH_MIX_BYTES);
	unsigned max_outputs = c_max_search_results;

	result = set_constants(&dagSize128, &max_outputs);

	// create buffer for dag
	result = cudaMalloc(&m_dag_ptr, _dagSize);

	// create buffer for header256
	result = cudaMalloc(&m_header, 32);

	// copy dag to CPU.
    result = cudaMemcpy(m_dag_ptr, _dag, _dagSize, cudaMemcpyHostToDevice);
	
	// create mining buffers
	for (unsigned i = 0; i != s_numStreams; ++i)
	{		
		result = cudaMallocHost(&m_hash_buf[i], 32 * c_hash_batch_size);
		result = cudaMallocHost(&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t));
		result = cudaStreamCreate(&m_streams[i]);
	}
	if (result != cudaSuccess)
	{
		cout << cudaGetErrorString(result) << endl;
		return false;
	}
	return true;
}

/**
 * Prevent High CPU usage while waiting for an async task
 */
static unsigned waitStream(cudaStream_t stream)
{
	unsigned wait_ms = 0;
	while (cudaStreamQuery(stream) == cudaErrorNotReady) {
		this_thread::sleep_for(chrono::milliseconds(10));
		wait_ms += 10;
	}
	return wait_ms;
}

void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	struct pending_batch
	{
		uint64_t start_nonce;
		unsigned buf;
	};
	std::queue<pending_batch> pending;

	static uint32_t const c_zero = 0;

	// update header constant buffer
	cudaMemcpy(m_header, header, 32, cudaMemcpyHostToDevice);
	for (unsigned i = 0; i != s_numStreams; ++i)
	{
		cudaMemcpy(m_search_buf[i], &c_zero, 4, cudaMemcpyHostToDevice);
	}
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}

	unsigned buf = 0;
	std::random_device engine;
	uint64_t start_nonce = std::uniform_int_distribution<uint64_t>()(engine);
	for (;; start_nonce += s_gridSize)
	{
		run_ethash_search(s_gridSize, s_blockSize, m_streams[buf], m_search_buf[buf], m_header, m_dag_ptr, start_nonce, target);
		
		pending.push({ start_nonce, buf });
		buf = (buf + 1) % s_numStreams;

		// read results
		if (pending.size() == s_numStreams)
		{
			pending_batch const& batch = pending.front();

			uint32_t results[1 + c_max_search_results];

			if (!s_highCPU)
				waitStream(m_streams[buf]); // 28ms
			cudaMemcpyAsync(results, m_search_buf[batch.buf], (1 + c_max_search_results) * sizeof(uint32_t), cudaMemcpyHostToHost, m_streams[batch.buf]);

			unsigned num_found = std::min<unsigned>(results[0], c_max_search_results);
			uint64_t nonces[c_max_search_results];
			for (unsigned i = 0; i != num_found; ++i)
			{
				nonces[i] = batch.start_nonce + results[i + 1];
				//cout << results[i + 1] << ", ";
			}
			//if (num_found > 0)
			//	cout << endl;
			
			bool exit = num_found && hook.found(nonces, num_found);
			exit |= hook.searched(batch.start_nonce, s_gridSize * s_blockSize); // always report searched before exit
			if (exit)
				break;

			start_nonce += s_gridSize * s_blockSize;
			// reset search buffer if we're still going
			if (num_found)
				cudaMemcpyAsync(m_search_buf[batch.buf], &c_zero, 4, cudaMemcpyHostToDevice, m_streams[batch.buf]);

			cudaError err = cudaGetLastError();
			if (cudaSuccess != err)
			{
				throw std::runtime_error(cudaGetErrorString(err));
			}
			pending.pop();
		}
	}	
}

