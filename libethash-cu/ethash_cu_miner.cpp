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
/** @file ethash_cu_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <queue>
#include <random>
#include <vector>
#include <chrono>
#include <thread>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include "ethash_cu_miner.h"
#include "ethash_cu_miner_kernel_globals.h"


#define ETHASH_BYTES 32

// workaround lame platforms
#if !CL_VERSION_1_2
#define CL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE
#define CL_MEM_HOST_READ_ONLY 0
#endif

#undef min
#undef max

using namespace std;

unsigned const ethash_cu_miner::c_defaultLocalWorkSize = 128;
unsigned const ethash_cu_miner::c_defaultGlobalWorkSizeMultiplier = 2048; // * CL_DEFAULT_LOCAL_WORK_SIZE

ethash_cu_miner::search_hook::~search_hook() {}

ethash_cu_miner::ethash_cu_miner()
{
}

std::string ethash_cu_miner::platform_info(unsigned _deviceId)
{
	int runtime_version;
	int device_count;
	
	device_count = get_num_devices();

	if (device_count == 0)
		return std::string();

	if (cudaRuntimeGetVersion(&runtime_version) == cudaErrorInvalidValue)
	{
		cout << cudaGetErrorString(cudaErrorInvalidValue) << endl;
		return std::string();
	}

	// use selected default device
	int device_num = std::min<int>((int)_deviceId, device_count - 1);

	cudaDeviceProp device_props;
	if (cudaGetDeviceProperties(&device_props, device_num) == cudaErrorInvalidDevice)
	{
		cout << cudaGetErrorString(cudaErrorInvalidDevice) << endl;
		return std::string();
	}

	char platform[5];
	int version_major = runtime_version / 1000;
	int version_minor = (runtime_version - (version_major * 1000)) / 10;
	sprintf(platform, "%d.%d", version_major, version_minor);


	char compute[5];
	sprintf(compute, "%d.%d", device_props.major, device_props.minor);

	return "{ \"platform\": \"CUDA " + std::string(platform) + "\", \"device\": \"" + device_props.name + "\", \"version\": \"Compute " + std::string(compute) + "\" }";

}

int ethash_cu_miner::get_num_devices()
{
	int device_count;

	if (cudaGetDeviceCount(&device_count) == cudaErrorNoDevice)
	{
		cout << cudaGetErrorString(cudaErrorNoDevice) << endl;
		return 0;
	}
	return device_count;
}

void ethash_cu_miner::finish()
{
	for (unsigned i = 0; i != m_num_buffers; i++) {
		cudaStreamDestroy(m_streams[i]);
		m_streams[i] = 0;
	}
	cudaDeviceReset();
}

bool ethash_cu_miner::init(uint8_t const* _dag, uint64_t _dagSize, unsigned num_buffers, unsigned search_batch_size, unsigned workgroup_size, unsigned _deviceId, bool highcpu)
{
	
	int device_count = get_num_devices();

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
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	m_num_buffers = num_buffers;
	m_search_batch_size = search_batch_size;

	m_hash_buf	 = new void *[m_num_buffers];
	m_search_buf = new uint32_t *[m_num_buffers];
	m_streams    = new cudaStream_t[m_num_buffers];

	// use requested workgroup size, but we require multiple of 8
	m_workgroup_size = ((workgroup_size + 7) / 8) * 8;

	m_highcpu = highcpu;

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
	for (unsigned i = 0; i != m_num_buffers; ++i)
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

void ethash_cu_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
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
	for (unsigned i = 0; i != m_num_buffers; ++i)
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
	for (;; start_nonce += m_search_batch_size)
	{
		run_ethash_search(m_search_batch_size / m_workgroup_size, m_workgroup_size, m_streams[buf], m_search_buf[buf], m_header, m_dag_ptr, start_nonce, target);	
		
		pending.push({ start_nonce, buf });
		buf = (buf + 1) % m_num_buffers;

		// read results
		if (pending.size() == m_num_buffers)
		{
			pending_batch const& batch = pending.front();

			uint32_t results[1 + c_max_search_results];

			if (!m_highcpu)
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
			exit |= hook.searched(batch.start_nonce, m_search_batch_size); // always report searched before exit
			if (exit)
				break;

			start_nonce += m_search_batch_size;
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

