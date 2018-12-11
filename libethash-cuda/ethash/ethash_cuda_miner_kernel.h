#pragma once

#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include "../cuda_helper.h"
#include "../dag_generation_kernel.h"


void set_constants(hash128_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size);

void set_header(hash32_t _header);

void set_target(uint64_t _target);

void run_ethash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile Search_results* g_output, uint64_t start_nonce, uint32_t parallelHash);

