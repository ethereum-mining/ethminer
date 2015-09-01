/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to sp_, trpuvot, djm34, cbuchner for things i took from ccminer.
*/

#include "ethash_cuda_miner_kernel.h"
#include "ethash_cuda_miner_kernel_globals.h"
#include "cuda_helper.h"

#define SHUFFLE_MIN_VER 350
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
#include "dagger_shuffled.cuh"
#else
#include "dagger_shared.cuh"
#endif

__global__ void 
__launch_bounds__(128, 7)
ethash_search(
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
	)
{
	
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;	
	
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
	uint64_t hash = compute_hash_shuffle((uint2 *)g_header, g_dag, start_nonce + gid);
	if (cuda_swab64(hash) < target)
#else
	hash32_t hash = compute_hash(g_header, g_dag, start_nonce + gid);	
	if (cuda_swab64(hash.uint64s[0]) < target)
#endif
	{
		atomicInc(g_output, d_max_outputs);
		g_output[g_output[0]] = gid;
	}
	
}

void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
)
{
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
	ethash_search <<<blocks, threads, 0, stream >>>(g_output, g_header, g_dag, start_nonce, target);
#else
	ethash_search <<<blocks, threads, (sizeof(compute_hash_share) * threads) / THREADS_PER_HASH, stream>>>(g_output, g_header, g_dag, start_nonce, target);
#endif
}

cudaError set_constants(
	uint32_t * dag_size,
	uint32_t * max_outputs
	)
{
	cudaError result;
	result = cudaMemcpyToSymbol(d_dag_size, dag_size, sizeof(uint32_t));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint32_t));
	return result;
}
