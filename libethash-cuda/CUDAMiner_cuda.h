#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cuda_runtime.h>

#if (__CUDACC_VER_MAJOR__ > 8)
#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))
#else
#define SHFL(x, y, z) __shfl((x), (y), (z))
#endif

#if (__CUDA_ARCH__ >= 320)
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
#define MAX_SEARCH_RESULTS 4U

typedef struct {
	uint32_t count;
	struct {
		// One word for gid and 8 for mix hash
		uint32_t gid;
		uint32_t mix[8];
	} result[MAX_SEARCH_RESULTS];
} Search_results;

typedef struct
{
	uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef struct
{
	uint64_t uint64s[256 / sizeof(uint64_t)];
} hash256_t;

typedef union {
	uint32_t words[64 / sizeof(uint32_t)];
	uint2	 uint2s[64 / sizeof(uint2)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
	uint32_t words[200 / sizeof(uint32_t)];
	uint64_t uint64s[200 / sizeof(uint64_t)];
	uint2	 uint2s[200 / sizeof(uint2)];
	uint4	 uint4s[200 / sizeof(uint4)];
} hash200_t;

void set_constants(hash64_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size);
void get_constants(hash64_t** _dag, uint32_t* _dag_size, hash64_t** _light, uint32_t* _light_size);

void set_header(hash32_t _header);

void set_target(uint64_t _target);

void ethash_generate_dag(
	hash64_t* dag,
	uint64_t dag_bytes,
	hash64_t * light,
	uint32_t light_words,
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	int device
	);

struct cuda_runtime_error : public virtual std::runtime_error
{
	cuda_runtime_error( std::string msg ) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)				\
do {							\
	cudaError_t result = call;				\
	if (cudaSuccess != result) {			\
		std::stringstream ss;			\
		ss << "CUDA error in func " 		\
            << __FUNCTION__ 		\
			<< " at line "			\
			<< __LINE__			\
			<< " calling " #call " failed with error "     \
			<< cudaGetErrorString(result);	\
		throw cuda_runtime_error(ss.str());	\
	}						\
} while (0)

#define CU_SAFE_CALL(call)								\
do {													\
	CUresult result = call;								\
	if (result != CUDA_SUCCESS) {						\
		std::stringstream ss;							\
		const char *msg;								\
		cuGetErrorName(result, &msg);                   \
		ss << "CUDA error in func " 					\
			<< __FUNCTION__ 							\
			<< " at line "								\
			<< __LINE__									\
			<< " calling " #call " failed with error "  \
			<< msg;										\
		throw cuda_runtime_error(ss.str());				\
	}													\
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
