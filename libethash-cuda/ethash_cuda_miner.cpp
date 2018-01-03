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
#include <queue>
#include <random>
#include <sstream>
#include <chrono>
#include <thread>
#include <libethash/ethash.h>
#include <libethash/internal.h>
#include <libdevcore/Log.h>
#include <cuda_runtime.h>
#include "ethash_cuda_miner.h"
#include "ethash_cuda_miner_kernel_globals.h"

// workaround lame platforms

#undef min
#undef max

using namespace std;
using namespace dev;

struct CUDAChannel: public LogChannel
{
	static const char* name() { return EthOrange " cu"; }
	static const int verbosity = 2;
	static const bool debug = false;
};
#define cudalog clog(CUDAChannel)
#define ETHCUDA_LOG(_contents) cudalog << _contents


unsigned const ethash_cuda_miner::c_defaultBlockSize = 128;
unsigned const ethash_cuda_miner::c_defaultGridSize = 8192; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const ethash_cuda_miner::c_defaultNumStreams = 2;

ethash_cuda_miner::search_hook::~search_hook() {}

ethash_cuda_miner::ethash_cuda_miner(size_t numDevices) : m_light(numDevices) {}

bool ethash_cuda_miner::configureGPU(
	size_t numDevices,
	const int* _devices,
	unsigned _blockSize,
	unsigned _gridSize,
	unsigned _numStreams,
	unsigned _scheduleFlag,
	uint64_t _currentBlock
	)
{
	try
	{
		s_blockSize = _blockSize;
		s_gridSize = _gridSize;
		s_numStreams = _numStreams;
		s_scheduleFlag = _scheduleFlag;

		cudalog << "Using grid size " << s_gridSize << ", block size " << s_blockSize;

		// by default let's only consider the DAG of the first epoch
		uint64_t dagSize = ethash_get_datasize(_currentBlock);
		int devicesCount = static_cast<int>(numDevices);
		for (int i = 0; i < devicesCount; i++)
		{
			if (_devices[i] != -1)
			{
				int deviceId = min(devicesCount - 1, _devices[i]);
				cudaDeviceProp props;
				CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
				if (props.totalGlobalMem >= dagSize)
				{
					cudalog <<  "Found suitable CUDA device [" << string(props.name) << "] with " << props.totalGlobalMem << " bytes of GPU memory";
				}
				else
				{
					cudalog <<  "CUDA device " << string(props.name) << " has insufficient GPU memory." << props.totalGlobalMem << " bytes of memory found < " << dagSize << " bytes of memory required";
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

void ethash_cuda_miner::setParallelHash(unsigned _parallelHash)
{
  m_parallelHash = _parallelHash;
}

unsigned ethash_cuda_miner::m_parallelHash = 4;
unsigned ethash_cuda_miner::s_blockSize = ethash_cuda_miner::c_defaultBlockSize;
unsigned ethash_cuda_miner::s_gridSize = ethash_cuda_miner::c_defaultGridSize;
unsigned ethash_cuda_miner::s_numStreams = ethash_cuda_miner::c_defaultNumStreams;
unsigned ethash_cuda_miner::s_scheduleFlag = 0;

bool ethash_cuda_miner::init(size_t numDevices, ethash_light_t _light, uint8_t const* _lightData, uint64_t _lightSize, unsigned _deviceId, bool _cpyToHost, uint8_t* &hostDAG, unsigned dagCreateDevice)
{
	try
	{
		if (numDevices == 0)
			return false;

		// use selected device
		m_device_num = _deviceId < numDevices -1 ? _deviceId : numDevices - 1;
		nvmlh = wrap_nvml_create();

		cudaDeviceProp device_props;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));

		cudalog << "Using device: " << device_props.name << " (Compute " + to_string(device_props.major) + "." + to_string(device_props.minor) + ")";

		m_search_buf = new volatile uint32_t *[s_numStreams];
		m_streams = new cudaStream_t[s_numStreams];

		uint64_t dagSize = ethash_get_datasize(_light->block_number);
		uint32_t dagSize128   = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(_lightSize / sizeof(node));

		
		
		CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
		cudalog << "Set Device to current";
		if(dagSize128 != m_dag_size || !m_dag)
		{
			//We need to reset the device and recreate the dag  
			cudalog << "Resetting device";
			CUDA_SAFE_CALL(cudaDeviceReset());
			CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
			CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
			//We need to reset the light and the Dag for the following code to reallocate
			//since cudaDeviceReset() free's all previous allocated memory
			m_light[m_device_num] = nullptr;
			m_dag = nullptr; 
		}
		// create buffer for cache
		hash128_t * dag = m_dag;
		hash64_t * light = m_light[m_device_num];

		if(!light){ 
			cudalog << "Allocating light with size: " << _lightSize;
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), _lightSize));
		}
		// copy lightData to device
		CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), _lightData, _lightSize, cudaMemcpyHostToDevice));
		m_light[m_device_num] = light;
		
		if(dagSize128 != m_dag_size || !dag) // create buffer for dag
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));
			
		set_constants(dag, dagSize128, light, lightSize64); //in ethash_cuda_miner_kernel.cu
		
		if(dagSize128 != m_dag_size || !dag)
		{
			// create mining buffers
			cudalog << "Generating mining buffers"; //TODO whats up with this?
			for (unsigned i = 0; i != s_numStreams; ++i)
			{
				CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], SEARCH_RESULT_BUFFER_SIZE * sizeof(uint32_t)));
				CUDA_SAFE_CALL(cudaStreamCreate(&m_streams[i]));
			}
			
			memset(&m_current_header, 0, sizeof(hash32_t));
			m_current_target = 0;
			m_current_nonce = 0;
			m_current_index = 0;

			m_sharedBytes = device_props.major * 100 < SHUFFLE_MIN_VER ? (64 * s_blockSize) / 8 : 0 ;

			if (!hostDAG)
			{
				if((m_device_num == dagCreateDevice) || !_cpyToHost){ //if !cpyToHost -> All devices shall generate their DAG
					cudalog << "Generating DAG for GPU #" << m_device_num << " with dagSize: " 
							<< dagSize <<" gridSize: " << s_gridSize << " &m_streams[0]: " << &m_streams[0];
					ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0], m_device_num);

					if (_cpyToHost)
					{
						uint8_t* memoryDAG = new uint8_t[dagSize];
						cudalog << "Copying DAG from GPU #" << m_device_num << " to host";
						CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagSize, cudaMemcpyDeviceToHost));

						hostDAG = memoryDAG;
					}
				}else{
					while(!hostDAG)
						this_thread::sleep_for(chrono::milliseconds(100)); 
					goto cpyDag;
				}
			}
			else
			{
cpyDag:
				cudalog << "Copying DAG from host to GPU #" << m_device_num;
				const void* hdag = (const void*)hostDAG;
				CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(dag), hdag, dagSize, cudaMemcpyHostToDevice));
			}
		}
    
		m_dag = dag;
		m_dag_size = dagSize128;
		return true;
	}
	catch (runtime_error const&)
	{
		return false;
	}
}

void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook, bool _ethStratum, uint64_t _startN)
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
	if (_ethStratum)
	{
		if (initialize)
		{
			m_starting_nonce = 0;
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i][0] = 0;
		}
		if (m_starting_nonce != _startN)
		{
			// reset nonce counter
			m_starting_nonce = _startN;
			m_current_nonce = m_starting_nonce;
		}
	}
	else
	{
		if (initialize)
		{
			random_device engine;
			m_current_nonce = uniform_int_distribution<uint64_t>()(engine);
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i][0] = 0;
		}
	}
	uint64_t batch_size = s_gridSize * s_blockSize;
	for (; !exit; m_current_index++, m_current_nonce += batch_size)
	{
		auto stream_index = m_current_index % s_numStreams;
		cudaStream_t stream = m_streams[stream_index];
		volatile uint32_t* buffer = m_search_buf[stream_index];
		uint32_t found_count = 0;
		uint64_t nonces[SEARCH_RESULT_BUFFER_SIZE - 1];
		uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;
		if (m_current_index >= s_numStreams)
		{
			CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
			found_count = buffer[0];
			if (found_count) {
				buffer[0] = 0;
				if (found_count > (SEARCH_RESULT_BUFFER_SIZE - 1))
					found_count = SEARCH_RESULT_BUFFER_SIZE - 1;
				for (unsigned int j = 0; j < found_count; j++)
					nonces[j] = nonce_base + buffer[j + 1];
			}
		}
		run_ethash_search(s_gridSize, s_blockSize, m_sharedBytes, stream, buffer, m_current_nonce, m_parallelHash);
		if (m_current_index >= s_numStreams)
		{
			if (found_count)
				hook.found(nonces, found_count);
			hook.searched(batch_size);
			exit = hook.shouldStop();
		}
	}
}

dev::eth::HwMonitor ethash_cuda_miner::hwmon()
{
	dev::eth::HwMonitor hw;
	if (nvmlh) {
		unsigned int tempC = 0, fanpcnt = 0;
		wrap_nvml_get_tempC(nvmlh, nvmlh->cuda_nvml_device_id[m_device_num], &tempC);
		wrap_nvml_get_fanpcnt(nvmlh, nvmlh->cuda_nvml_device_id[m_device_num], &fanpcnt);
		hw.tempC = tempC;
		hw.fanP = fanpcnt;
	}
	return hw;
}

