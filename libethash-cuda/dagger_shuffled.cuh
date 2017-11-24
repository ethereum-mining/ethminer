#include "ethash_cuda_miner_kernel_globals.h"
#include "ethash_cuda_miner_kernel.h"
#include "cuda_helper.h"

template <uint32_t _PARALLEL_HASH>
__device__ __forceinline__ uint64_t compute_hash(
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	uint2 state[12];
	
	state[4] = vectorize(nonce);

	keccak_f1600_init(state);
	
	// Threads work together in this phase in groups of 8.
	const int thread_id  = threadIdx.x &  (THREADS_PER_HASH - 1);
	const int mix_idx    = thread_id & 3;

	for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
	{
		uint4 mix[_PARALLEL_HASH];
		uint32_t offset[_PARALLEL_HASH];
		uint32_t init0[_PARALLEL_HASH];
	
		// share init among threads
		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			uint2 shuffle[8];
			for (int j = 0; j < 8; j++) 
			{
#if CUDA_VERSION < SHUFFLE_DEPRECATED
				shuffle[j].x = __shfl(state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl(state[j].y, i+p, THREADS_PER_HASH);
#else
				shuffle[j].x = __shfl_sync(0xFFFFFFFF,state[j].x, i+p, THREADS_PER_HASH);
				shuffle[j].y = __shfl_sync(0xFFFFFFFF,state[j].y, i+p, THREADS_PER_HASH);
#endif
			}
			switch (mix_idx)
			{
				case 0: mix[p] = vectorize2(shuffle[0], shuffle[1]); break;
				case 1: mix[p] = vectorize2(shuffle[2], shuffle[3]); break;
				case 2: mix[p] = vectorize2(shuffle[4], shuffle[5]); break;
				case 3: mix[p] = vectorize2(shuffle[6], shuffle[7]); break;
			}
#if CUDA_VERSION < SHUFFLE_DEPRECATED
			init0[p] = __shfl(shuffle[0].x, 0, THREADS_PER_HASH);
#else
			init0[p] = __shfl_sync(0xFFFFFFFF,shuffle[0].x, 0, THREADS_PER_HASH);
#endif
		}

		for (uint32_t a = 0; a < ACCESSES; a += 4)
		{
			int t = bfe(a, 2u, 3u);

			for (uint32_t b = 0; b < 4; b++)
			{
				for (int p = 0; p < _PARALLEL_HASH; p++)
				{
					offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t *)&mix[p])[b]) % d_dag_size;
#if CUDA_VERSION < SHUFFLE_DEPRECATED
					offset[p] = __shfl(offset[p], t, THREADS_PER_HASH);
#else
					offset[p] = __shfl_sync(0xFFFFFFFF,offset[p], t, THREADS_PER_HASH);
#endif
				}
				#pragma unroll
				for (int p = 0; p < _PARALLEL_HASH; p++)
				{
                                        //if(blockIdx.x == 0 && threadIdx.x==0 && offset[p] > (d_dag_size>>1)) //larger than half
                                        //    printf("d_dag_size = %d offset[p] = %d\n", d_dag_size, offset[p]);
					mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
				}

                                
			}
		}

		for (int p = 0; p < _PARALLEL_HASH; p++)
		{
			uint2 shuffle[4];
			uint32_t thread_mix = fnv_reduce(mix[p]);

			// update mix accross threads
#if CUDA_VERSION < SHUFFLE_DEPRECATED
			shuffle[0].x = __shfl(thread_mix, 0, THREADS_PER_HASH);
			shuffle[0].y = __shfl(thread_mix, 1, THREADS_PER_HASH);
			shuffle[1].x = __shfl(thread_mix, 2, THREADS_PER_HASH);
			shuffle[1].y = __shfl(thread_mix, 3, THREADS_PER_HASH);
			shuffle[2].x = __shfl(thread_mix, 4, THREADS_PER_HASH);
			shuffle[2].y = __shfl(thread_mix, 5, THREADS_PER_HASH);
			shuffle[3].x = __shfl(thread_mix, 6, THREADS_PER_HASH);
			shuffle[3].y = __shfl(thread_mix, 7, THREADS_PER_HASH);
#else
			shuffle[0].x = __shfl_sync(0xFFFFFFFF,thread_mix, 0, THREADS_PER_HASH);
			shuffle[0].y = __shfl_sync(0xFFFFFFFF,thread_mix, 1, THREADS_PER_HASH);
			shuffle[1].x = __shfl_sync(0xFFFFFFFF,thread_mix, 2, THREADS_PER_HASH);
			shuffle[1].y = __shfl_sync(0xFFFFFFFF,thread_mix, 3, THREADS_PER_HASH);
			shuffle[2].x = __shfl_sync(0xFFFFFFFF,thread_mix, 4, THREADS_PER_HASH);
			shuffle[2].y = __shfl_sync(0xFFFFFFFF,thread_mix, 5, THREADS_PER_HASH);
			shuffle[3].x = __shfl_sync(0xFFFFFFFF,thread_mix, 6, THREADS_PER_HASH);
			shuffle[3].y = __shfl_sync(0xFFFFFFFF,thread_mix, 7, THREADS_PER_HASH);
#endif
			if ((i+p) == thread_id) {
				//move mix into state:
				state[8] = shuffle[0];
				state[9] = shuffle[1];
				state[10] = shuffle[2];
				state[11] = shuffle[3];
			}
		}
	}
	
	// keccak_256(keccak_512(header..nonce) .. mix);
	return keccak_f1600_final(state);
}
