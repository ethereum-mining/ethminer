/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"

#define ETHASH_ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
#define MAX_SEARCH_RESULTS 4U

typedef struct
{
    uint32_t count;
    struct
    {
        // One word for gid and 8 for mix hash
        uint32_t gid;
        uint32_t mix[8];
        uint32_t pad[7];  // pad to size power of 2 (keep this so the struct is same for ethash
    } result[MAX_SEARCH_RESULTS];
} search_results;

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

void set_constants(hash128_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size);
void get_constants(hash128_t** _dag, uint32_t* _dag_size, hash64_t** _light, uint32_t* _light_size);

void set_header(hash32_t _header);

void set_target(uint64_t _target);

void run_ethash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile search_results* g_output, uint64_t start_nonce, uint32_t parallelHash);

void ethash_generate_dag(uint64_t dag_size, uint32_t blocks, uint32_t threads, cudaStream_t stream);

struct cuda_runtime_error : public virtual std::runtime_error
{
    cuda_runtime_error(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)                                                              \
    do                                                                                    \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (cudaSuccess != err)                                                           \
        {                                                                                 \
            std::stringstream ss;                                                         \
            ss << "CUDA error in func " << __FUNCTION__ << " at line " << __LINE__ << ' ' \
               << cudaGetErrorString(err);                                                \
            throw cuda_runtime_error(ss.str());                                           \
        }                                                                                 \
    } while (0)

#define CU_SAFE_CALL(call)                                                         \
    do                                                                             \
    {                                                                              \
        CUresult result = call;                                                    \
        if (result != CUDA_SUCCESS)                                                \
        {                                                                          \
            std::stringstream ss;                                                  \
            const char* msg;                                                       \
            cuGetErrorName(result, &msg);                                          \
            ss << "CUDA error in func " << __FUNCTION__ << " at line " << __LINE__ \
               << " calling " #call " failed with error " << msg;                  \
            throw cuda_runtime_error(ss.str());                                    \
        }                                                                          \
    } while (0)

#define NVRTC_SAFE_CALL(call)                                                                     \
    do                                                                                            \
    {                                                                                             \
        nvrtcResult result = call;                                                                \
        if (result != NVRTC_SUCCESS)                                                              \
        {                                                                                         \
            std::stringstream ss;                                                                 \
            ss << "CUDA NVRTC error in func " << __FUNCTION__ << " at line " << __LINE__          \
               << " calling " #call " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            throw cuda_runtime_error(ss.str());                                                   \
        }                                                                                         \
    } while (0)
