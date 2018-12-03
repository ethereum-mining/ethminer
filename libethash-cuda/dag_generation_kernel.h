#pragma once

#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
#define MAX_SEARCH_RESULTS 4U

struct Search_Result
{
    // One word for gid and 8 for mix hash
    uint32_t gid;
    uint32_t mix[8];
    uint32_t pad[7];  // pad to size power of 2
};

struct Search_results
{
    Search_Result result[MAX_SEARCH_RESULTS];
    uint32_t count = 0;
};

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

typedef struct
{
    uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef struct
{
    uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union
{
    uint32_t words[64 / sizeof(uint32_t)];
    uint2 uint2s[64 / sizeof(uint2)];
    uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
    uint32_t words[200 / sizeof(uint32_t)];
    uint2 uint2s[200 / sizeof(uint2)];
    uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;

void ethash_generate_dag(uint64_t dag_size, uint32_t blocks, uint32_t threads, cudaStream_t stream);
