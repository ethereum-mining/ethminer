#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>


// 64-bit ROTATE LEFT

#if __CUDA_ARCH__ >= 320

__device__ uint64_t ROTL64H(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"mov.b64 %0, {vl,vh};\n\t"
		"}"
		: "=l"(res) : "l"(x), "r"(offset)
		);
	return res;
}

__device__ uint64_t ROTL64L(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"mov.b64 %0, {vh,vl};\n\t"
		"}"
		: "=l"(res) : "l"(x), "r"(offset)
		);
	return res;
}
#elif __CUDA_ARCH__ >= 120

#define ROTL64H(x, n) ROTL64(x,n)
#define ROTL64L(x, n) ROTL64(x,n)

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
#else
#define ROTL64H(x, n) ROTL64(x,n)
#define ROTL64L(x, n) ROTL64(x,n)
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif