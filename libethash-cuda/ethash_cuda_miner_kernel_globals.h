#ifndef _ETHASH_CUDA_MINER_KERNEL_GLOBALS_H_
#define _ETHASH_CUDA_MINER_KERNEL_GLOBALS_H_

#define SHUFFLE_MIN_VER 300

//#include "cuda_helper.h"

__constant__ uint32_t d_dag_size;
__constant__ hash128_t* d_dag;
__constant__ uint32_t d_light_size;
__constant__ hash64_t* d_light;
__constant__ hash32_t d_header;
__constant__ uint64_t d_target;

#endif