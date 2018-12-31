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

#include "ethash_miner_kernel.h"
#include "cuda_helper.h"

__constant__ uint32_t d_dag_size;
__constant__ hash128_t* d_dag;
__constant__ uint32_t d_light_size;
__constant__ hash64_t* d_light;
__constant__ hash32_t d_header;
__constant__ uint64_t d_target;

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

#define FNV_PRIME 0x01000193
#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64 / 4)

#define fnv(x, y) ((x)*FNV_PRIME ^ (y))

DEV_INLINE uint4 fnv4(uint4 a, uint4 b)
{
    uint4 c;
    c.x = a.x * FNV_PRIME ^ b.x;
    c.y = a.y * FNV_PRIME ^ b.y;
    c.z = a.z * FNV_PRIME ^ b.z;
    c.w = a.w * FNV_PRIME ^ b.w;
    return c;
}

DEV_INLINE uint32_t fnv_reduce(uint4 v)
{
    return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }

__device__ __constant__ uint2 const keccak_round_constants[24] = {{0x00000001, 0x00000000},
    {0x00008082, 0x00000000}, {0x0000808a, 0x80000000}, {0x80008000, 0x80000000},
    {0x0000808b, 0x00000000}, {0x80000001, 0x00000000}, {0x80008081, 0x80000000},
    {0x00008009, 0x80000000}, {0x0000008a, 0x00000000}, {0x00000088, 0x00000000},
    {0x80008009, 0x00000000}, {0x8000000a, 0x00000000}, {0x8000808b, 0x00000000},
    {0x0000008b, 0x80000000}, {0x00008089, 0x80000000}, {0x00008003, 0x80000000},
    {0x00008002, 0x80000000}, {0x00000080, 0x80000000}, {0x0000800a, 0x00000000},
    {0x8000000a, 0x80000000}, {0x80008081, 0x80000000}, {0x00008080, 0x80000000},
    {0x80000001, 0x00000000}, {0x80008008, 0x80000000}};

DEV_INLINE uint2 xor5(const uint2 a, const uint2 b, const uint2 c, const uint2 d, const uint2 e)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
    uint2 result;
    asm(
        "// xor5\n\t"
        "lop3.b32 %0, %2, %3, %4, 0x96;\n\t"
        "lop3.b32 %0, %0, %5, %6, 0x96;\n\t"
        "lop3.b32 %1, %7, %8, %9, 0x96;\n\t"
        "lop3.b32 %1, %1, %10, %11, 0x96;"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(b.x), "r"(c.x), "r"(d.x), "r"(e.x), "r"(a.y), "r"(b.y), "r"(c.y), "r"(d.y),
        "r"(e.y));
    return result;
#else
    return a ^ b ^ c ^ d ^ e;
#endif
}

DEV_INLINE uint2 xor3(const uint2 a, const uint2 b, const uint2 c)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
    uint2 result;
    asm(
        "// xor3\n\t"
        "lop3.b32 %0, %2, %3, %4, 0x96;\n\t"
        "lop3.b32 %1, %5, %6, %7, 0x96;"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(b.x), "r"(c.x), "r"(a.y), "r"(b.y), "r"(c.y));
    return result;
#else
    return a ^ b ^ c;
#endif
}

DEV_INLINE uint2 chi(const uint2 a, const uint2 b, const uint2 c)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
    uint2 result;
    asm(
        "// chi\n\t"
        "lop3.b32 %0, %2, %3, %4, 0xD2;\n\t"
        "lop3.b32 %1, %5, %6, %7, 0xD2;"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(b.x), "r"(c.x),  // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
        "r"(a.y), "r"(b.y), "r"(c.y));   // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
    return result;
#else
    return a ^ (~b) & c;
#endif
}

DEV_INLINE void keccak_f1600_init(uint2* state)
{
    uint2 s[25];
    uint2 t[5], u, v;
    const uint2 u2zero = make_uint2(0, 0);

    devectorize2(d_header.uint4s[0], s[0], s[1]);
    devectorize2(d_header.uint4s[1], s[2], s[3]);
    s[4] = state[4];
    s[5] = make_uint2(1, 0);
    s[6] = u2zero;
    s[7] = u2zero;
    s[8] = make_uint2(0, 0x80000000);

    #pragma unroll
    for (int i = 9; i < 25; i++)
        s[i] = u2zero;

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0].x = s[0].x ^ s[5].x;
    t[0].y = s[0].y;
    t[1] = s[1];
    t[2] = s[2];
    t[3].x = s[3].x;
    t[3].y = s[3].y ^ s[8].y;
    t[4] = s[4];

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2] = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4] = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8] = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7] = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u = s[5];
    v = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u = s[10];
    v = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u = s[15];
    v = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u = s[20];
    v = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[0];

    for (int i = 1; i < 23; i++)
    {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[10] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[6] ^= u;
    s[16] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[12] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[18] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[23];

    #pragma unroll
    for (int i = 0; i < 12; ++i)
        state[i] = s[i];
}

DEV_INLINE uint64_t keccak_f1600_final(uint2* state)
{
    uint2 s[25];
    uint2 t[5], u, v;
    const uint2 u2zero = make_uint2(0, 0);

    #pragma unroll
    for (int i = 0; i < 12; ++i)
        s[i] = state[i];

    s[12] = make_uint2(1, 0);
    s[13] = u2zero;
    s[14] = u2zero;
    s[15] = u2zero;
    s[16] = make_uint2(0, 0x80000000);
    for (uint32_t i = 17; i < 25; i++)
        s[i] = u2zero;

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor3(s[0], s[5], s[10]);
    t[1] = xor3(s[1], s[6], s[11]) ^ s[16];
    t[2] = xor3(s[2], s[7], s[12]);
    t[3] = s[3] ^ s[8];
    t[4] = s[4] ^ s[9];

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2] = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4] = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8] = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7] = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u = s[5];
    v = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u = s[10];
    v = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u = s[15];
    v = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u = s[20];
    v = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[0];

    for (int i = 1; i < 23; i++)
    {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }

    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    s[0] = xor3(s[0], t[4], ROL2(t[1], 1));
    s[6] = xor3(s[6], t[0], ROL2(t[2], 1));
    s[12] = xor3(s[12], t[1], ROL2(t[3], 1));

    s[1] = ROL2(s[6], 44);
    s[2] = ROL2(s[12], 43);

    s[0] = chi(s[0], s[1], s[2]);

    /* iota: a[0,0] ^= round constant */
    // s[0] ^= vectorize(keccak_round_constants[23]);
    return devectorize(s[0] ^ keccak_round_constants[23]);
}

DEV_INLINE void SHA3_512(uint2* s)
{
    uint2 t[5], u, v;
    const uint2 u2zero = make_uint2(0, 0);

    s[8] = make_uint2(1, 0x80000000);

    #pragma unroll
    for (int i = 9; i < 25; i++)
    {
        s[i] = u2zero;
    }

    for (int i = 0; i < 23; i++)
    {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = t[4] ^ ROL2(t[1], 1);
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = t[0] ^ ROL2(t[2], 1);
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = t[1] ^ ROL2(t[3], 1);
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = t[2] ^ ROL2(t[4], 1);
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = t[3] ^ ROL2(t[0], 1);
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1] = ROL2(s[6], 44);
        s[6] = ROL2(s[9], 20);
        s[9] = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2] = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL2(s[19], 8);
        s[19] = ROL2(s[23], 56);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4] = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8] = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5] = ROL2(s[3], 28);
        s[3] = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7] = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u = s[0];
        v = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u = s[5];
        v = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u = s[10];
        v = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u = s[15];
        v = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u = s[20];
        v = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= LDG(keccak_round_constants[i]);
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = t[4] ^ ROL2(t[1], 1);
    s[0] ^= u;
    s[10] ^= u;

    u = t[0] ^ ROL2(t[2], 1);
    s[6] ^= u;
    s[16] ^= u;

    u = t[1] ^ ROL2(t[3], 1);
    s[12] ^= u;
    s[22] ^= u;

    u = t[2] ^ ROL2(t[4], 1);
    s[3] ^= u;
    s[18] ^= u;

    u = t[3] ^ ROL2(t[0], 1);
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u = s[0];
    v = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= LDG(keccak_round_constants[23]);
}


template <uint32_t _PARALLEL_HASH>
DEV_INLINE bool compute_hash(uint64_t nonce, uint2* mix_hash)
{
    // sha3_512(header .. nonce)
    uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state);

    // Threads work together in this phase in groups of 8.
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    const int mix_idx = thread_id & 3;

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            int ip = i + p;

            #pragma unroll
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = SHFL(state[j].x, ip, THREADS_PER_HASH);
                shuffle[j].y = SHFL(state[j].y, ip, THREADS_PER_HASH);
            }
            switch (mix_idx)
            {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        for (uint32_t a = 0; a < ETHASH_ACCESSES; a += 4)
        {
            int t = bfe(a, 2u, 3u);

            for (uint32_t b = 0; b < 4; b++)
            {
                for (int p = 0; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
                    offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    // keccak_256(keccak_512(header..nonce) .. mix);
    if (cuda_swab64(keccak_f1600_final(state)) > d_target)
        return true;

    mix_hash[0] = state[8];
    mix_hash[1] = state[9];
    mix_hash[2] = state[10];
    mix_hash[3] = state[11];

    return false;
}

template <uint32_t _PARALLEL_HASH>
__global__ void ethash_search(volatile search_results* g_output, uint64_t start_nonce)
{
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 mix[4];
    if (compute_hash<_PARALLEL_HASH>(start_nonce + gid, mix))
        return;
    uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
    if (index >= MAX_SEARCH_RESULTS)
        return;
    g_output->result[index].gid = gid;
    g_output->result[index].mix[0] = mix[0].x;
    g_output->result[index].mix[1] = mix[0].y;
    g_output->result[index].mix[2] = mix[1].x;
    g_output->result[index].mix[3] = mix[1].y;
    g_output->result[index].mix[4] = mix[2].x;
    g_output->result[index].mix[5] = mix[2].y;
    g_output->result[index].mix[6] = mix[3].x;
    g_output->result[index].mix[7] = mix[3].y;
}

void run_ethash_search(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile search_results* g_output, uint64_t start_nonce, uint32_t parallelHash)
{
    switch (parallelHash)
    {
    case 1:
        ethash_search<1><<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        break;
    case 2:
        ethash_search<2><<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        break;
    case 4:
        ethash_search<4><<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        break;
    case 8:
        ethash_search<8><<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        break;
    default:
        ethash_search<4><<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
        break;
    }
    CUDA_SAFE_CALL(cudaGetLastError());
}

__global__ void ethash_calculate_dag_item(uint32_t start)
{
    uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (((node_index >> 1) & (~1)) >= d_dag_size)
        return;

    hash200_t dag_node;
    copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
    dag_node.words[0] ^= node_index;
    SHA3_512(dag_node.uint2s);

    const int thread_id = threadIdx.x & 3;

    for (uint32_t i = 0; i != ETHASH_DATASET_PARENTS; ++i)
    {
        uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
        for (uint32_t t = 0; t < 4; t++)
        {
            uint32_t shuffle_index = SHFL(parent_index, t, 4);

            uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
            for (int w = 0; w < 4; w++)
            {
                uint4 s4 = make_uint4(SHFL(p4.x, w, 4), SHFL(p4.y, w, 4), SHFL(p4.z, w, 4), SHFL(p4.w, w, 4));
                if (t == thread_id)
                {
                    dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
                }
            }
        }
    }
    SHA3_512(dag_node.uint2s);
    hash64_t* dag_nodes = (hash64_t*)d_dag;

    for (uint32_t t = 0; t < 4; t++)
    {
        uint32_t shuffle_index = SHFL(node_index, t, 4);
        uint4 s[4];

        #pragma unroll
        for (int w = 0; w < 4; w++)
        {
            s[w] = make_uint4(SHFL(dag_node.uint4s[w].x, t, 4), SHFL(dag_node.uint4s[w].y, t, 4),
                              SHFL(dag_node.uint4s[w].z, t, 4), SHFL(dag_node.uint4s[w].w, t, 4));
        }
        if (shuffle_index < d_dag_size * 2)
        {
            dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
        }
    }
}

void ethash_generate_dag(
    uint64_t dag_size, uint32_t gridSize, uint32_t blockSize, cudaStream_t stream)
{
    const uint32_t work = (uint32_t)(dag_size / sizeof(hash64_t));
    const uint32_t run = gridSize * blockSize;

    uint32_t base;
    for (base = 0; base <= work - run; base += run)
    {
        ethash_calculate_dag_item<<<gridSize, blockSize, 0, stream>>>(base);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    if (base < work)
    {
        uint32_t lastGrid = work - base;
        lastGrid = (lastGrid + blockSize - 1) / blockSize;
        ethash_calculate_dag_item<<<lastGrid, blockSize, 0, stream>>>(base);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    CUDA_SAFE_CALL(cudaGetLastError());
}

void set_constants(hash128_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag, &_dag, sizeof(hash128_t*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_dag_size, &_dag_size, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light, &_light, sizeof(hash64_t*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_light_size, &_light_size, sizeof(uint32_t)));
}

void get_constants(hash128_t** _dag, uint32_t* _dag_size, hash64_t** _light, uint32_t* _light_size)
{
    /*
       Using the direct address of the targets did not work.
       So I've to read first into local variables when using cudaMemcpyFromSymbol()
    */
    if (_dag)
    {
        hash128_t* _d;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&_d, d_dag, sizeof(hash128_t*)));
        *_dag = _d;
    }
    if (_dag_size)
    {
        uint32_t _ds;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&_ds, d_dag_size, sizeof(uint32_t)));
        *_dag_size = _ds;
    }
    if (_light)
    {
        hash64_t* _l;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&_l, d_light, sizeof(hash64_t*)));
        *_light = _l;
    }
    if (_light_size)
    {
        uint32_t _ls;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&_ls, d_light_size, sizeof(uint32_t)));
        *_light_size = _ls;
    }
}

void set_header(hash32_t _header)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_header, &_header, sizeof(hash32_t)));
}

void set_target(uint64_t _target)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &_target, sizeof(uint64_t)));
}
