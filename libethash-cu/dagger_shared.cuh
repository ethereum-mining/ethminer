#include "ethash_cu_miner_kernel_globals.h"
#include "ethash_cu_miner_kernel.h"
#include "keccak.cuh"
#include "dagger.cuh"

typedef union
{
	hash64_t init;
	hash32_t mix;
} compute_hash_share;

__device__ hash64_t init_hash(hash32_t const* header, uint64_t nonce)
{
	hash64_t init;

	// sha3_512(header .. nonce)
	uint64_t state[25];

	copy(state, header->uint64s, 4);
	state[4] = nonce;
	state[5] = 0x0000000000000001;
	state[6] = 0;
	state[7] = 0;
	state[8] = 0x8000000000000000;
	for (uint32_t i = 9; i < 25; i++)
	{
		state[i] = 0;
	}

	keccak_f1600_block((uint2 *)state, 8);
	copy(init.uint64s, state, 8);
	return init;
}

__device__ uint32_t inner_loop(uint4 mix, uint32_t thread_id, uint32_t* share, hash128_t const* g_dag)
{
	// share init0
	if (thread_id == 0)
		*share = mix.x;

	uint32_t init0 = *share;

	uint32_t a = 0;

	do
	{

		bool update_share = thread_id == ((a >> 2) & (THREADS_PER_HASH - 1));

		//#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{

			if (update_share)
			{
				uint32_t m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a + i), m[i]) % d_dag_size;
			}
			__threadfence_block();

#if __CUDA_ARCH__ >= 350
			mix = fnv4(mix, __ldg(&g_dag[*share].uint4s[thread_id]));
#else
			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
#endif

		}

	} while ((a += 4) != ACCESSES);

	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix)
{
	uint64_t state[25];

	hash32_t hash;

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->uint64s, 8);
	copy(state + 8, mix->uint64s, 4);
	state[12] = 0x0000000000000001;
	for (uint32_t i = 13; i < 16; i++)
	{
		state[i] = 0;
	}
	state[16] = 0x8000000000000000;
	for (uint32_t i = 17; i < 25; i++)
	{
		state[i] = 0;
	}

	keccak_f1600_block((uint2 *)state, 1);

	// copy out
	copy(hash.uint64s, state, 4);
	return hash;
}

__device__ hash32_t compute_hash(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	extern __shared__  compute_hash_share share[];

	// Compute one init hash per work item.
	hash64_t init = init_hash(g_header, nonce);

	// Threads work together in this phase in groups of 8.
	uint32_t const thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
	uint32_t const hash_id = threadIdx.x >> 3;

	hash32_t mix;

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;

		uint4 thread_init = share[hash_id].init.uint4s[thread_id & 3];

		uint32_t thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uint32s, g_dag);

		share[hash_id].mix.uint32s[thread_id] = thread_mix;


		if (i == thread_id)
			mix = share[hash_id].mix;
	}

	return final_hash(&init, &mix);
}