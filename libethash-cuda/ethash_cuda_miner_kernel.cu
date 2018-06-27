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

#include "keccak.cuh"
#include "dagger_shuffled.cuh"

template <uint32_t _PARALLEL_HASH>
__global__ void 
ethash_search(
	volatile search_results* g_output,
	uint64_t start_nonce
	)
{
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint2 mix[4];
        if (compute_hash<_PARALLEL_HASH>(start_nonce + gid, d_target, mix))
		return;
	uint32_t index = atomicInc((uint32_t *)&g_output->count, 0xffffffff);
	if (index >= SEARCH_RESULTS)
		return;
	g_output->result[index].gid = gid;
	g_output->result[index].mix[0] = mix[0].x;
	g_output->result[index].mix[1] = mix[0].y;
	g_output->result[index].mix[2] = mix[1].x;
	g_output->result[index].mix[3] = mix[1].y;
	g_output->result[index].mix[4] = mix[2].x;
	g_output->result[index].mix[5] = mix[2].y;
	g_output->result[index].mix[6] = mix[3].x;
	g_output->result[index].mix[7] = mix[3].y;
}

void run_ethash_search(
	uint32_t gridSize,
	uint32_t blockSize,
	cudaStream_t stream,
	volatile search_results* g_output,
	uint64_t start_nonce,
	uint32_t parallelHash
)
{
	switch (parallelHash)
	{
		case 1: ethash_search <1> <<<gridSize, blockSize, 0, stream >>>(g_output, start_nonce); break;
		case 2: ethash_search <2> <<<gridSize, blockSize, 0, stream >>>(g_output, start_nonce); break;
		case 4: ethash_search <4> <<<gridSize, blockSize, 0, stream >>>(g_output, start_nonce); break;
		case 8: ethash_search <8> <<<gridSize, blockSize, 0, stream >>>(g_output, start_nonce); break;
		default: ethash_search <4> <<<gridSize, blockSize, 0, stream >>>(g_output, start_nonce); break;
	}
	CUDA_SAFE_CALL(cudaGetLastError());
}

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64/4)


__global__ void
ethash_calculate_dag_item(uint32_t start)
{
	uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
	if (((node_index/4)*4) >= d_dag_size * 2) return;

	hash200_t dag_node;
	copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
	dag_node.words[0] ^= node_index;
	SHA3_512(dag_node.uint2s);

	const int thread_id = threadIdx.x & 3;

	for (uint32_t i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
		uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
		for (uint32_t t = 0; t < 4; t++) {

			uint32_t shuffle_index = __shfl_sync(0xFFFFFFFF,parent_index, t, 4);

			uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
			for (int w = 0; w < 4; w++) {

				uint4 s4 = make_uint4(__shfl_sync(0xFFFFFFFF,p4.x, w, 4), __shfl_sync(0xFFFFFFFF,p4.y, w, 4), __shfl_sync(0xFFFFFFFF,p4.z, w, 4), __shfl_sync(0xFFFFFFFF,p4.w, w, 4));
				if (t == thread_id) {
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
				}
			}
		}
	}
	SHA3_512(dag_node.uint2s);
	hash64_t * dag_nodes = (hash64_t *)d_dag;

	for (uint32_t t = 0; t < 4; t++) {
		uint32_t shuffle_index = __shfl_sync(0xFFFFFFFF,node_index, t, 4);
		uint4 s[4];
		for (uint32_t w = 0; w < 4; w++) {
			s[w] = make_uint4(__shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].x, t, 4), __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].y, t, 4), __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].z, t, 4), __shfl_sync(0xFFFFFFFF,dag_node.uint4s[w].w, t, 4));
		}
		if (shuffle_index < d_dag_size * 2) {
		dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
	}
}
}

void ethash_generate_dag(
	uint64_t dag_size,
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream
	)
{
	const uint32_t work = (uint32_t)(dag_size / sizeof(hash64_t));
	const uint32_t run = blocks * threads;

	for (uint32_t base = 0; base < work; base += run)
	{
		ethash_calculate_dag_item <<<blocks, threads, 0, stream>>>(base);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
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
