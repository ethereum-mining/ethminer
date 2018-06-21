#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cuda_runtime.h>

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
#define SEARCH_RESULTS 4

typedef struct {
	uint32_t count;
	struct {
		// One word for gid and 8 for mix hash
		uint32_t gid;
		uint32_t mix[8];
		uint32_t pad[7]; // pad to size power of 2
	} result[SEARCH_RESULTS];
} search_results;

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

typedef struct
{
	uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef struct
{
	uint4	 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union {
	uint32_t words[64 / sizeof(uint32_t)];
	uint2	 uint2s[64 / sizeof(uint2)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
	uint32_t words[200 / sizeof(uint32_t)];
	uint2	 uint2s[200 / sizeof(uint2)];
	uint4	 uint4s[200 / sizeof(uint4)];
} hash200_t;

void set_constants(
	hash128_t* _dag,
	uint32_t _dag_size,
	hash64_t * _light,
	uint32_t _light_size
	);

void set_header(
	hash32_t _header
	);

void set_target(
	uint64_t _target
	);

void run_ethash_search(
	uint32_t gridSize,
	uint32_t blockSize,
	cudaStream_t stream,
	volatile search_results* g_output,
	uint64_t start_nonce,
	uint32_t parallelHash
	);

void ethash_generate_dag(
	uint64_t dag_size,
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream
	);

struct cuda_runtime_error : public virtual std::runtime_error
{
	cuda_runtime_error( const std::string &msg ) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)				\
do {							\
	cudaError_t err = call;				\
	if (cudaSuccess != err) {			\
		std::stringstream ss;			\
		ss << "CUDA error in func " 		\
			<< __FUNCTION__ 		\
			<< " at line "			\
			<< __LINE__			\
			<< ' '				\
			<< cudaGetErrorString(err);	\
		throw cuda_runtime_error(ss.str());	\
	}						\
} while (0)

