#pragma once

__constant__ uint32_t d_dag_size;
__constant__ hash128_t* d_dag;
__constant__ uint32_t d_light_size;
__constant__ hash64_t* d_light;
__constant__ hash32_t d_header;
__constant__ uint64_t d_target;

#if (__CUDACC_VER_MAJOR__ > 8)
#define SHFL(w, x,y,z)               __shfl_sync((w),(x),(y),(z))
#else
#define SHFL(w, x,y,z)               __shfl((x),(y),(z))
#endif

