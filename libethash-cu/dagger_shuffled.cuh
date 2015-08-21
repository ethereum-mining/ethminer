#include "ethash_cu_miner_kernel_globals.h"
#include "ethash_cu_miner_kernel.h"
#include "keccak.cuh"
#include "dagger.cuh"

__device__ uint64_t compute_hash_shuffle(
	uint2 const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	uint2 state[25];

	state[0] = g_header[0];
	state[1] = g_header[1];
	state[2] = g_header[2];
	state[3] = g_header[3];
	state[4] = vectorize(nonce);
	state[5] = vectorize(0x0000000000000001ULL);
	for (uint32_t i = 6; i < 25; i++)
	{
		state[i] = make_uint2(0, 0);
	}
	state[8] = vectorize(0x8000000000000000ULL);
	keccak_f1600_block(state,8);

	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x &  (THREADS_PER_HASH - 1);
	const int start_lane = threadIdx.x & ~(THREADS_PER_HASH - 1);
	const int mix_idx    = thread_id & 3;

	uint4 mix;
	uint2 shuffle[8];

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init among threads
		for (int j = 0; j < 8; j++) {
			shuffle[j].x = __shfl(state[j].x, start_lane + i);
			shuffle[j].y = __shfl(state[j].y, start_lane + i);
		}

		// ugly but avoids local reads/writes
		if (mix_idx < 2) {
			if (mix_idx == 0)
				mix = vectorize2(shuffle[0], shuffle[1]);
			else
				mix = vectorize2(shuffle[2], shuffle[3]);
		}
		else  {
			if (mix_idx == 2)
				mix = vectorize2(shuffle[4], shuffle[5]);
			else
				mix = vectorize2(shuffle[6], shuffle[7]);
		}
		
		uint32_t init0 = __shfl(shuffle[0].x, start_lane);

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t)
				{	
					shuffle[0].x = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;
				}
				shuffle[0].x = __shfl(shuffle[0].x, start_lane + t);

				mix = fnv4(mix, g_dag[shuffle[0].x].uint4s[thread_id]);
			}
		}

		uint32_t thread_mix = fnv_reduce(mix);

		// update mix accross threads
		shuffle[0].x = __shfl(thread_mix, start_lane + 0);
		shuffle[0].y = __shfl(thread_mix, start_lane + 1);
		shuffle[1].x = __shfl(thread_mix, start_lane + 2);
		shuffle[1].y = __shfl(thread_mix, start_lane + 3);
		shuffle[2].x = __shfl(thread_mix, start_lane + 4);
		shuffle[2].y = __shfl(thread_mix, start_lane + 5);
		shuffle[3].x = __shfl(thread_mix, start_lane + 6);
		shuffle[3].y = __shfl(thread_mix, start_lane + 7);

		if (i == thread_id) {
			//move mix into state:
			state[8] = shuffle[0];
			state[9] = shuffle[1];
			state[10] = shuffle[2];
			state[11] = shuffle[3];
		}
	}

	// keccak_256(keccak_512(header..nonce) .. mix);
	state[12] = vectorize(0x0000000000000001ULL);
	for (uint32_t i = 13; i < 25; i++)
	{
		state[i] = vectorize(0ULL);
	}
	state[16] = vectorize(0x8000000000000000);
	keccak_f1600_block(state, 1);

	return devectorize(state[0]);
}