/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to sp_, trpuvot, djm34, cbuchner for things i took from ccminer.
*/

#include "ethash_cuda_miner_kernel.h"
#include "ethash_cuda_miner_kernel_globals.h"
#include "cuda_helper.h"

#define SHUFFLE_MIN_VER 300

#if __CUDA_ARCH__ < SHUFFLE_MIN_VER
#include "dagger_shared.cuh"
#define TPB		128
#define BPSM	4
#else
#include "dagger_shuffled.cuh"
#define TPB		896
#define BPSM	1
#endif

__global__ void 
__launch_bounds__(TPB, BPSM)
ethash_search(
	volatile uint32_t* g_output,
	uint64_t start_nonce
	)
{
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;	
	uint64_t hash = compute_hash(start_nonce + gid);
	if (cuda_swab64(hash) > d_target) return;
	uint32_t index = atomicInc(const_cast<uint32_t*>(g_output), SEARCH_RESULT_BUFFER_SIZE - 1) + 1;
	g_output[index] = gid;
	__threadfence_system();
}

void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	volatile uint32_t* g_output,
	uint64_t start_nonce
)
{
	ethash_search << <blocks, threads, (sizeof(compute_hash_share) * threads) / THREADS_PER_HASH, stream >> >(g_output, start_nonce);
	CUDA_SAFE_CALL(cudaGetLastError());
}

void set_constants(
	hash128_t* _dag,
	uint32_t _dag_size
	)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag, &_dag, sizeof(hash128_t *)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag_size, &_dag_size, sizeof(uint32_t)));
}

void set_header(
	hash32_t _header
	)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_header, &_header, sizeof(hash32_t)));
}

void set_target(
	uint64_t _target
	)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &_target, sizeof(uint64_t)));
}
