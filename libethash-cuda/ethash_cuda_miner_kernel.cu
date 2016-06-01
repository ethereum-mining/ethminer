/*
* Genoil's CUDA mining kernel for Ethereum
* based on Tim Hughes' opencl kernel.
* thanks to sp_, trpuvot, djm34, cbuchner for things i took from ccminer.
*/

#include "ethash_cuda_miner_kernel.h"
#include "ethash_cuda_miner_kernel_globals.h"
#include "cuda_helper.h"

#include "fnv.cuh"

#define copy(dst, src, count) for (int i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }


#if __CUDA_ARCH__ < SHUFFLE_MIN_VER
#include "keccak_u64.cuh"
#include "dagger_shared.cuh"
#define TPB		128
#define BPSM	4
#else
#include "keccak.cuh"
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
}

void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	uint32_t sharedbytes,
	cudaStream_t stream,
	volatile uint32_t* g_output,
	uint64_t start_nonce
)
{
	ethash_search << <blocks, threads, sharedbytes, stream >> >(g_output, start_nonce);
	CUDA_SAFE_CALL(cudaGetLastError());
}

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64/4)

__global__ void
__launch_bounds__(128, 7)
ethash_calculate_dag_item(uint32_t start)
{
	uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
	if (node_index > d_dag_size * 2) return;

	hash200_t dag_node;
	copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
	dag_node.words[0] ^= node_index;
	SHA3_512(dag_node.uint2s);

	const int thread_id = threadIdx.x & 3;

	for (uint32_t i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
		uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
#if __CUDA_ARCH__ < SHUFFLE_MIN_VER
		for (unsigned w = 0; w != 4; ++w) {
			dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], d_light[parent_index].uint4s[w]);
		}
#else
		for (uint32_t t = 0; t < 4; t++) {
			uint32_t shuffle_index = __shfl(parent_index, t, 4);
			uint4 p4 = d_light[shuffle_index].uint4s[thread_id];

			for (int w = 0; w < 4; w++) {
				uint4 s4 = make_uint4(__shfl(p4.x, w, 4), __shfl(p4.y, w, 4), __shfl(p4.z, w, 4), __shfl(p4.w, w, 4));
				if (t == thread_id) {
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
				}
			}

		}
#endif		
	}
	SHA3_512(dag_node.uint2s);
	hash64_t * dag_nodes = (hash64_t *)d_dag;

#if __CUDA_ARCH__ < SHUFFLE_MIN_VER
	for (uint32_t i = 0; i < 4; i++) {
		dag_nodes[node_index].uint4s[i] =  dag_node.uint4s[i];
	}
#else
	for (uint32_t t = 0; t < 4; t++) {

		uint32_t shuffle_index = __shfl(node_index, t, 4);
		uint4 s[4];
		for (uint32_t w = 0; w < 4; w++) {
			s[w] = make_uint4(__shfl(dag_node.uint4s[w].x, t, 4), __shfl(dag_node.uint4s[w].y, t, 4), __shfl(dag_node.uint4s[w].z, t, 4), __shfl(dag_node.uint4s[w].w, t, 4));
		}
		dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
	}
#endif		 
}

void ethash_generate_dag(
	uint64_t dag_size,
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	int device
	)
{
	uint32_t const work = (uint32_t)(dag_size / sizeof(hash64_t));

	uint32_t fullRuns = work / (blocks * threads);
	uint32_t const restWork = work % (blocks * threads);
	if (restWork > 0) fullRuns++;
	for (uint32_t i = 0; i < fullRuns; i++)
	{
		ethash_calculate_dag_item <<<blocks, threads, 0, stream >>>(i * blocks * threads);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		printf("CUDA#%d: %.0f%%\n",device, 100.0f * (float)i / (float)fullRuns);
	}
	//printf("GPU#%d 100%%\n");
	CUDA_SAFE_CALL(cudaGetLastError());
}

void set_constants(
	hash128_t* _dag,
	uint32_t _dag_size,
	hash64_t * _light,
	uint32_t _light_size
	)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag, &_dag, sizeof(hash128_t *)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag_size, &_dag_size, sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light, &_light, sizeof(hash64_t *)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light_size, &_light_size, sizeof(uint32_t)));
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
