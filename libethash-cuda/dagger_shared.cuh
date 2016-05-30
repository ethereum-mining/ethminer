#include "ethash_cuda_miner_kernel_globals.h"
#include "ethash_cuda_miner_kernel.h"

typedef union {
	uint4	 uint4s[4];
	uint64_t ulongs[8];
	uint32_t uints[16];
} compute_hash_share;


__device__ uint64_t compute_hash(
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	uint64_t state[25];
	state[4] = nonce;
	keccak_f1600_init(state);
	
	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x &  (THREADS_PER_HASH - 1);
	const int hash_id = threadIdx.x  >> 3;

	extern __shared__  compute_hash_share share[];
	
	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			copy(share[hash_id].ulongs, state, 8);

		__syncthreads();

		uint4 mix = share[hash_id].uint4s[thread_id & 3];
		__syncthreads();

		uint32_t *share0 = share[hash_id].uints;

		// share init0
		if (thread_id == 0)
			*share0 = mix.x;
		__syncthreads();
		uint32_t init0 = *share0;

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = bfe(a, 2u, 3u);

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t) {
					*share0 = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;
				}
				__syncthreads();

				mix = fnv4(mix, d_dag[*share0].uint4s[thread_id]);
			}
		}

		share[hash_id].uints[thread_id] = fnv_reduce(mix);
		__syncthreads();

		if (i == thread_id)
			copy(state + 8, share[hash_id].ulongs, 4);

		__syncthreads();
	}
	
	// keccak_256(keccak_512(header..nonce) .. mix);
	return keccak_f1600_final(state);
}