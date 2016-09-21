#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#define asm("a" : "=l"(result) : "l"(a))
#define __CUDA_ARCH__ 520 // highlight shuffle code by default.

uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
uint32_t __shfl(uint32_t x, uint32_t y, uint32_t z);
uint32_t atomicExch(uint32_t *x, uint32_t y);
uint32_t atomicAdd(uint32_t *x, uint32_t y);
void __syncthreads(void);
void __threadfence(void);
void __threadfence_block(void);

uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
uint32_t __shfl(uint32_t x, uint32_t y, uint32_t z);
uint32_t atomicExch(uint32_t *x, uint32_t y);
uint32_t atomicAdd(uint32_t *x, uint32_t y);
void __syncthreads(void);
void __threadfence(void);
#endif

#include <stdint.h>

#ifndef MAX_GPUS
#define MAX_GPUS 32
#endif

extern "C" int device_map[MAX_GPUS];
extern "C"  long device_sm[MAX_GPUS];
extern cudaStream_t gpustream[MAX_GPUS];

// common functions
extern void cuda_check_cpu_init(int thr_id, uint32_t threads);
extern void cuda_check_cpu_setTarget(const void *ptarget);
extern void cuda_check_cpu_setTarget_mod(const void *ptarget, const void *ptarget2);
extern uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash);
extern uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint32_t foundnonce);
extern void cudaReportHardwareFailure(int thr_id, cudaError_t error, const char* func);

#ifndef __CUDA_ARCH__
// define blockDim and threadIdx for host
extern const dim3 blockDim;
extern const uint3 threadIdx;
#endif

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);


#ifndef SPH_C32
#define SPH_C32(x) ((x ## U))
// #define SPH_C32(x) ((uint32_t)(x ## U))
#endif

#ifndef SPH_C64
#define SPH_C64(x) ((x ## ULL))
// #define SPH_C64(x) ((uint64_t)(x ## ULL))
#endif

#ifndef SPH_T32
#define SPH_T32(x) (x)
// #define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))
#endif
#ifndef SPH_T64
#define SPH_T64(x) (x)
// #define SPH_T64(x) ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#endif

#define ROTL32c(x, n) ((x) << (n)) | ((x) >> (32 - (n)))

#if __CUDA_ARCH__ < 320
// Kepler (Compute 3.0)
#define ROTL32(x, n) ((x) << (n)) | ((x) >> (32 - (n)))
#else
// Kepler (Compute 3.5, 5.0)
__device__ __forceinline__ uint32_t ROTL32(const uint32_t x, const uint32_t n)
{
	return(__funnelshift_l((x), (x), (n)));
}
#endif
#if __CUDA_ARCH__ < 320
// Kepler (Compute 3.0)
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
__device__ __forceinline__ uint32_t ROTR32(const uint32_t x, const uint32_t n)
{
	return(__funnelshift_r((x), (x), (n)));
}
#endif




__device__ __forceinline__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
	uint64_t result;
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(LO), "r"(HI));
	return result;
}

__device__ __forceinline__ uint64_t REPLACE_HIWORD(const uint64_t x, const uint32_t y) 
{
	uint64_t result;
	asm(
		"{\n\t"
		".reg .u32 t,t2; \n\t"
		"mov.b64 {t2,t},%1; \n\t"
		"mov.b64 %0,{t2,%2}; \n\t"
		"}" : "=l"(result) : "l"(x), "r"(y)
		);
	return result;

}
__device__ __forceinline__ uint64_t REPLACE_LOWORD(const uint64_t x, const uint32_t y) 
{
	uint64_t result;
	asm(
		"{\n\t"
		".reg .u32 t,t2; \n\t"
		"mov.b64 {t2,t},%1; \n\t"
		"mov.b64 %0,{%2,t}; \n\t"
		"}" : "=l"(result) : "l"(x) , "r"(y)
		);
	return result;
}

// Endian Drehung für 32 Bit Typen
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t cuda_swab32(const uint32_t x)
{
	/* device */
	return __byte_perm(x, x, 0x0123);
}
#else
	/* host */
	#define cuda_swab32(x) \
	((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
		(((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif


static __device__ __forceinline__ uint32_t _HIWORD(const uint64_t x)
{
	uint32_t result;
	asm(
		"{\n\t"
		".reg .u32 xl; \n\t"
		"mov.b64 {xl,%0},%1; \n\t"
		"}" : "=r"(result) : "l"(x)
		);
	return result;
}

static __device__ __forceinline__ uint32_t _LOWORD(const uint64_t x)
{
	uint32_t result;
	asm(
		"{\n\t"
		".reg .u32 xh; \n\t"
		"mov.b64 {%0,xh},%1; \n\t"
		"}" : "=r"(result) : "l"(x)
		);
	return result;
}

// Input:       77665544 33221100
// Output:      00112233 44556677
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint64_t cuda_swab64(const uint64_t x)
{
	uint64_t result;
	uint2 t;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(t.x), "=r"(t.y) : "l"(x));
	t.x=__byte_perm(t.x, 0, 0x0123);
	t.y=__byte_perm(t.y, 0, 0x0123);
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(t.y), "r"(t.x));
	return result;
}
#else
	/* host */
	#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
#endif

/*********************************************************************/
// Macros to catch CUDA errors in CUDA runtime calls

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET(call) do {                                   \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return;                                                       \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET_X(call, ret) do {                            \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return ret;                                                   \
	}                                                                 \
} while (0)

/*********************************************************************/
#ifdef _WIN64
#define USE_XOR_ASM_OPTS 0
#else
#define USE_XOR_ASM_OPTS 1
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor1(const uint64_t a, const uint64_t b)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
	return result;
}
#else
#define xor1(a,b) (a ^ b)
#endif

/*
#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor3(const uint64_t a, const uint64_t b, const uint64_t c)
{
	uint64_t result;
	asm("xor.b64 %0, %2, %3;\n\t"
	    "xor.b64 %0, %0, %1;\n\t"
		//output : input registers 
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}
#else
#define xor3(a,b,c) (a ^ b ^ c)
#endif
*/

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor8(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d, const uint64_t e, const uint64_t f, const uint64_t g, const  uint64_t h)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(g) ,"l"(h));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}
#else
#define xor8(a,b,c,d,e,f,g,h) ((a^b)^(c^d)^(e^f)^(g^h))
#endif

// device asm for x17
__device__ __forceinline__
uint64_t xandx(const uint64_t a, const uint64_t b, const uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 n;\n\t"
		"xor.b64 %0, %2, %3;\n\t"
		"and.b64 n, %0, %1;\n\t"
		"xor.b64 %0, n, %3;"
	"}\n"
	: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t andor(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"and.b64 m,  %1, %2;\n\t"
		" or.b64 n,  %1, %2;\n\t"
		"and.b64 %0, n,  %3;\n\t"
		" or.b64 %0, %0, m ;\n\t"
	"}\n"
	: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t shr_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("shr.b64 %0,%1,%2;\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}

// device asm for ?
__device__ __forceinline__
uint64_t shl_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("shl.b64 %0,%1,%2;\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}

#ifndef USE_ROT_ASM_OPT
#define USE_ROT_ASM_OPT 2
#endif

// 64-bit ROTATE RIGHT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
__device__ __forceinline__
uint64_t ROTR64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTR64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shr.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shl.b64 %0, %1, roff;\n\t"
		"add.u64 %0, %0, lhs;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#else
/* host */
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		".reg .pred p;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"setp.lt.u32 p, %2, 32;\n\t"
		"@!p mov.b64 %0, {vl,vh};\n\t"
		"@p  mov.b64 %0, {vh,vl};\n\t"
	"}"
		: "=l"(res) : "l"(x) , "r"(offset)
	);
	return res;
}
#else
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif

__device__ __forceinline__
uint64_t SWAPDWORDS(uint64_t value)
{
#if __CUDA_ARCH__ >= 320
	uint2 temp;
	asm("mov.b64 {%0, %1}, %2; ": "=r"(temp.x), "=r"(temp.y) : "l"(value));
	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(temp.y), "r"(temp.x));
	return value;
#else
	return ROTL64(value, 32);
#endif
}

/* lyra2 - int2 operators */

__device__ __forceinline__
void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x) {
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(lo), "=r"(hi) : "l"(x));
}

__device__ __forceinline__ uint64_t devectorize(uint2 x)
{
	uint64_t result;
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(result) : "r"(x.x), "r"(x.y));
	return result;
}


__device__ __forceinline__ uint2 vectorize(const uint64_t x)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(x));
	return result;
}
__device__ __forceinline__ void devectorize2(uint4 inn, uint2 &x, uint2 &y)
{
	x.x = inn.x;
	x.y = inn.y;
	y.x = inn.z;
	y.y = inn.w;
}


__device__ __forceinline__ uint4 vectorize2(uint2 x, uint2 y)
{
	uint4 result;
	result.x = x.x;
	result.y = x.y;
	result.z = y.x;
	result.w = y.y;

	return result;
}

__device__ __forceinline__ uint4 vectorize2(uint2 x)
{
	uint4 result;
	result.x = x.x;
	result.y = x.y;
	result.z = x.x;
	result.w = x.y;
	return result;
}


__device__ __forceinline__ uint4 vectorize4(uint64_t x, uint64_t y)
{
	uint4 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(x));
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.z), "=r"(result.w) : "l"(y));
	return result;
}
__device__ __forceinline__ void devectorize4(uint4 inn, uint64_t &x, uint64_t &y)
{
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(x) : "r"(inn.x), "r"(inn.y));
	asm("mov.b64 %0,{%1,%2}; \n\t"
		: "=l"(y) : "r"(inn.z), "r"(inn.w));
}



static __device__ __forceinline__ uint2 vectorizelow(uint32_t v) {
	uint2 result;
	result.x = v;
	result.y = 0;
	return result;
}
static __device__ __forceinline__ uint2 vectorizehigh(uint32_t v) {
	uint2 result;
	result.x = 0;
	result.y = v;
	return result;
}

static __device__ __forceinline__ uint2 operator^ (uint2 a, uint32_t b) { return make_uint2(a.x^ b, a.y); }
static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __device__ __forceinline__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __device__ __forceinline__ uint2 operator| (uint2 a, uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
static __device__ __forceinline__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
static __device__ __forceinline__ void operator^= (uint2 &a, uint2 b) { a = a ^ b; }
static __device__ __forceinline__ uint2 operator+ (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

static __device__ __forceinline__ uint2 operator+ (uint2 a, uint32_t b)
{
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
}


static __device__ __forceinline__ uint2 operator- (uint2 a, uint32_t b)
{
	uint2 result;
	asm("{\n\t"
		"sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
}


static __device__ __forceinline__ uint2 operator- (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"sub.cc.u32 %0,%2,%4; \n\t"
		"subc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}



static __device__ __forceinline__ uint4 operator^ (uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __device__ __forceinline__ uint4 operator& (uint4 a, uint4 b) { return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w); }
static __device__ __forceinline__ uint4 operator| (uint4 a, uint4 b) { return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w); }
static __device__ __forceinline__ uint4 operator~ (uint4 a) { return make_uint4(~a.x, ~a.y, ~a.z, ~a.w); }
static __device__ __forceinline__ void operator^= (uint4 &a, uint4 b) { a = a ^ b; }
static __device__ __forceinline__ uint4 operator^ (uint4 a, uint2 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.x, a.w ^ b.y); }


static __device__ __forceinline__ void operator+= (uint2 &a, uint2 b) { a = a + b; }

/**
 * basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
 * (what does uint64 "*" operator)
 */
static __device__ __forceinline__ uint2 operator* (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
	"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}

// uint2 method
#if  __CUDA_ARCH__ >= 350
__device__ __inline__ uint2 ROR2(const uint2 a, const int offset) 
{
	uint2 result;
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else
__device__ __inline__ uint2 ROR2(const uint2 v, const int n) 
{
	uint2 result;
	if (n <= 32) 
	{
		result.y = ((v.y >> (n)) | (v.x << (32 - n)));
		result.x = ((v.x >> (n)) | (v.y << (32 - n)));
	}
	else 
	{
		result.y = ((v.x >> (n - 32)) | (v.y << (64 - n)));
		result.x = ((v.y >> (n - 32)) | (v.x << (64 - n)));
	}
	return result;
}
#endif



__device__ __inline__ uint32_t ROL8(const uint32_t x)
{
	return __byte_perm(x, x, 0x2103);
}
__device__ __inline__ uint32_t ROL16(const uint32_t x)
{
	return __byte_perm(x, x, 0x1032);
}
__device__ __inline__ uint32_t ROL24(const uint32_t x)
{
	return __byte_perm(x, x, 0x0321);
}

__device__ __inline__ uint2 ROR8(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x0765);
	result.y = __byte_perm(a.y, a.x, 0x4321);

	return result;
}

__device__ __inline__ uint2 ROR16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x5432);

	return result;
}

__device__ __inline__ uint2 ROR24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x6543);

	return result;
}

__device__ __inline__ uint2 ROL8(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x6543);
	result.y = __byte_perm(a.y, a.x, 0x2107);

	return result;
}

__device__ __inline__ uint2 ROL16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x5432);
	result.y = __byte_perm(a.y, a.x, 0x1076);

	return result;
}

__device__ __inline__ uint2 ROL24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x4321);
	result.y = __byte_perm(a.y, a.x, 0x0765);

	return result;
}



#if  __CUDA_ARCH__ >= 350
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset) {
	uint2 result;
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n)
{
		uint2 result;
		if (n <= 32) 
		{
			result.y = ((v.y << (n)) | (v.x >> (32 - n)));
			result.x = ((v.x << (n)) | (v.y >> (32 - n)));
		}
		else 
		{
			result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
			result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));

		}
		return result;
}
#endif

__device__ __forceinline__
uint64_t ROTR16(uint64_t x)
{
#if __CUDA_ARCH__ > 500
	short4 temp;
	asm("mov.b64 { %0,  %1, %2, %3 }, %4; ": "=h"(temp.x), "=h"(temp.y), "=h"(temp.z), "=h"(temp.w) : "l"(x));
	asm("mov.b64 %0, {%1, %2, %3 , %4}; ":  "=l"(x) : "h"(temp.y), "h"(temp.z), "h"(temp.w), "h"(temp.x));
	return x;
#else
	return ROTR64(x, 16);
#endif
}
__device__ __forceinline__
uint64_t ROTL16(uint64_t x)
{
#if __CUDA_ARCH__ > 500
	short4 temp;
	asm("mov.b64 { %0,  %1, %2, %3 }, %4; ": "=h"(temp.x), "=h"(temp.y), "=h"(temp.z), "=h"(temp.w) : "l"(x));
	asm("mov.b64 %0, {%1, %2, %3 , %4}; ":  "=l"(x) : "h"(temp.w), "h"(temp.x), "h"(temp.y), "h"(temp.z));
	return x;
#else
	return ROTL64(x, 16);
#endif
}

__device__ __forceinline__
uint2 SWAPINT2(uint2 x)
{
	return(make_uint2(x.y, x.x));
}
__device__ __forceinline__ bool cuda_hashisbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	if (hash[7] > target[7])
		return false;
	if (hash[7] < target[7])
		return true;
	if (hash[6] > target[6])
		return false;
	if (hash[6] < target[6])
		return true;
	if (hash[5] > target[5])
		return false;
	if (hash[5] < target[5])
		return true;
	if (hash[4] > target[4])
		return false;
	if (hash[4] < target[4])
		return true;
	if (hash[3] > target[3])
		return false;
	if (hash[3] < target[3])
		return true;
	if (hash[2] > target[2])
		return false;
	if (hash[2] < target[2])
		return true;
	if (hash[1] > target[1])
		return false;
	if (hash[1] < target[1])
		return true;
	if (hash[0] > target[0])
		return false;
	return true;
}

__device__ __forceinline__
uint2 SWAPDWORDS2(uint2 value)
{
	return make_uint2(value.y, value.x);
}
__device__ __forceinline__
uint4 SWAPDWORDS2(uint4 value)
{
	return make_uint4(value.y, value.x, value.w ,value.z);
}

static __forceinline__ __device__ uint2 SHL2(uint2 a, int offset)
{
#if __CUDA_ARCH__ > 300
	uint2 result;
	if (offset<32) 
	{
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else {
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
#else
	if (offset<=32) 
	{
		a.y = (a.y << offset) | (a.x >> (32 - offset));
		a.x = (a.x << offset);
	}
	else
	{
		a.y = (a.x << (offset-32));
		a.x = 0;
	}
	return a;
#endif
}
static __forceinline__ __device__ uint2 SHR2(uint2 a, int offset)
{
	#if __CUDA_ARCH__ > 300
	uint2 result;
	if (offset<32) {
		asm("{\n\t"
			"shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
			"shr.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else {
		asm("{\n\t"
			"shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
			"shl.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
	#else
	if (offset<=32) 
	{
		a.x = (a.x >> offset) | (a.y << (32 - offset));
		a.y = (a.y >> offset);
	}
	else
	{
		a.x = (a.y >> (offset - 32));
		a.y = 0;
	}
	return a;
	#endif
}

static __device__ __forceinline__ uint64_t devectorizeswap(uint2 v) { return MAKE_ULONGLONG(cuda_swab32(v.y), cuda_swab32(v.x)); }
static __device__ __forceinline__ uint2 vectorizeswap(uint64_t v) {
	uint2 result;
	LOHI(result.y, result.x, v);
	result.x = cuda_swab32(result.x);
	result.y = cuda_swab32(result.y);
	return result;
}


__device__ __forceinline__ uint32_t devectorize16(ushort2 x)
{
	uint32_t result;
	asm("mov.b32 %0,{%1,%2}; \n\t"
		: "=r"(result) : "h"(x.x) , "h"(x.y));
	return result;
}


__device__ __forceinline__ ushort2 vectorize16(uint32_t x)
{
	ushort2 result;
	asm("mov.b32 {%0,%1},%2; \n\t"
		: "=h"(result.x), "=h"(result.y) : "r"(x));
	return result;
}




static __device__ __forceinline__ uint4 mul4(uint4 a)
{
	uint4 result;
	asm("{\n\t"
		 "mul.lo.u32        %0,%4,%5;  \n\t"
		 "mul.hi.u32        %1,%4,%5;  \n\t"
		 "mul.lo.u32        %2,%6,%7;  \n\t"
		 "mul.hi.u32        %3,%6,%7;  \n\t"
		 "}\n\t"
		 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w));
	return result;
}
static __device__ __forceinline__ uint4 add4(uint4 a, uint4 b)
 {
	uint4 result;
	asm("{\n\t"
		 "add.cc.u32           %0,%4,%8;  \n\t"
		 "addc.u32             %1,%5,%9;  \n\t"
		 "add.cc.u32           %2,%6,%10;  \n\t"
		 "addc.u32             %3,%7,%11;  \n\t"
		 "}\n\t"
		 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
	return result;
	}

static __device__ __forceinline__ uint4 madd4(uint4 a, uint4 b)
 {
	uint4 result;
	asm("{\n\t"
		 "mad.lo.cc.u32        %0,%4,%5,%8;  \n\t"
		 "madc.hi.u32          %1,%4,%5,%9;  \n\t"
		 "mad.lo.cc.u32        %2,%6,%7,%10;  \n\t"
		 "madc.hi.u32          %3,%6,%7,%11;  \n\t"
		 "}\n\t"
		 : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
	return result;
	}

static __device__ __forceinline__ ulonglong2 madd4long(ulonglong2 a, ulonglong2 b)
 {
	ulonglong2 result;
	asm("{\n\t"
		 ".reg .u32 a0,a1,a2,a3,b0,b1,b2,b3;\n\t"
		 "mov.b64 {a0,a1}, %2;\n\t"
		 "mov.b64 {a2,a3}, %3;\n\t"
		 "mov.b64 {b0,b1}, %4;\n\t"
		 "mov.b64 {b2,b3}, %5;\n\t"
		 "mad.lo.cc.u32        b0,a0,a1,b0;  \n\t"
		 "madc.hi.u32          b1,a0,a1,b1;  \n\t"
		 "mad.lo.cc.u32        b2,a2,a3,b2;  \n\t"
		 "madc.hi.u32          b3,a2,a3,b3;  \n\t"
		 "mov.b64 %0, {b0,b1};\n\t"
		 "mov.b64 %1, {b2,b3};\n\t"
		 "}\n\t"
		 : "=l"(result.x), "=l"(result.y) : "l"(a.x), "l"(a.y), "l"(b.x), "l"(b.y));
	return result;
	}
static __device__ __forceinline__ void madd4long2(ulonglong2 &a, ulonglong2 b)
 {
	
		asm("{\n\t"
		 ".reg .u32 a0,a1,a2,a3,b0,b1,b2,b3;\n\t"
		 "mov.b64 {a0,a1}, %0;\n\t"
		 "mov.b64 {a2,a3}, %1;\n\t"
		 "mov.b64 {b0,b1}, %2;\n\t"
		 "mov.b64 {b2,b3}, %3;\n\t"
		 "mad.lo.cc.u32        b0,a0,a1,b0;  \n\t"
		 "madc.hi.u32          b1,a0,a1,b1;  \n\t"
		 "mad.lo.cc.u32        b2,a2,a3,b2;  \n\t"
		 "madc.hi.u32          b3,a2,a3,b3;  \n\t"
		 "mov.b64 %0, {b0,b1};\n\t"
		 "mov.b64 %1, {b2,b3};\n\t"
		 "}\n\t"
		 : "+l"(a.x), "+l"(a.y) : "l"(b.x), "l"(b.y));
	
}

__device__ __forceinline__
uint32_t xor3b(uint32_t a, uint32_t b, uint32_t c) {
	uint32_t result;
	asm("{ .reg .u32 t1;\n\t"
		"xor.b32 t1, %2, %3;\n\t"
		"xor.b32 %0, %1, t1;\n\t"
		"}"
		: "=r"(result) : "r"(a), "r"(b), "r"(c));
	return result;
}

__device__ __forceinline__
uint32_t shr_t32(uint32_t x, uint32_t n) {
	uint32_t result;
	asm("shr.b32 %0,%1,%2;"	: "=r"(result) : "r"(x), "r"(n));
	return result;
}

__device__ __forceinline__
uint32_t shl_t32(uint32_t x, uint32_t n) {
	uint32_t result;
	asm("shl.b32 %0,%1,%2;" : "=r"(result) : "r"(x), "r"(n));
	return result;
}

// device asm 32 for pluck
__device__ __forceinline__
uint32_t andor32(uint32_t a, uint32_t b, uint32_t c) {
	uint32_t result;
	asm("{ .reg .u32 m,n,o;\n\t"
		"and.b32 m,  %1, %2;\n\t"
		" or.b32 n,  %1, %2;\n\t"
		"and.b32 o,   n, %3;\n\t"
		" or.b32 %0,  m, o ;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(a), "r"(b), "r"(c));
	return result;
}

__device__ __forceinline__
uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits) {
	uint32_t ret;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
	return ret;

}

__device__ __forceinline__
uint32_t bfi(uint32_t x, uint32_t a, uint32_t bit, uint32_t numBits) {
	uint32_t ret;
	asm("bfi.b32 %0, %1, %2, %3,%4;" : "=r"(ret) : "r"(x), "r"(a), "r"(bit), "r"(numBits));
	return ret;
}



#endif // #ifndef CUDA_HELPER_H


