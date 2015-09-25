#ifndef _ETHASH_CUDA_MINER_KERNEL_H_
#define _ETHASH_CUDA_MINER_KERNEL_H_

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define SEARCH_RESULT_BUFFER_SIZE 64

typedef union
{
	uint64_t uint64s[16 / sizeof(uint64_t)];
	uint32_t uint32s[16 / sizeof(uint32_t)];
} hash16_t;

typedef union
{
	uint32_t uint32s[32 / sizeof(uint32_t)];
	uint64_t uint64s[32 / sizeof(uint64_t)];
	uint2 uint2s[32 / sizeof(uint2)];
	uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;


typedef union
{
	uint32_t uint32s[64 / sizeof(uint32_t)];
	uint64_t uint64s[64 / sizeof(uint64_t)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;


typedef union
{
	uint32_t uint32s[128 / sizeof(uint32_t)];
	uint4	 uint4s[128 / sizeof(uint4)];
} hash128_t;


void set_constants(
	hash128_t* _dag,
	uint32_t _dag_size
	);

void set_header(
	hash32_t _header
	);

void set_target(
	uint64_t _target
	);

void run_ethash_search(
	uint32_t search_batch_size,
	uint32_t workgroup_size,
	cudaStream_t stream,
	volatile uint32_t* g_output,
	uint64_t start_nonce
	);


#define CUDA_SAFE_CALL(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		const char * errorString = cudaGetErrorString(err);	\
		fprintf(stderr,										\
			"CUDA error in func '%s' at line %i : %s.\n",	\
			__FUNCTION__, __LINE__, errorString);			\
		throw std::runtime_error(errorString);				\
	}														\
} while (0)

#endif
