#include "cuda_helper.h"

__device__ __constant__ uint2 const keccak_round_constants[24] = {
    { 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
    { 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
    { 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
    { 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
    { 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
    { 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

DEV_INLINE uint2 xor5(
    const uint2 a, const uint2 b, const uint2 c, const uint2 d, const uint2 e)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
    uint2 result;
    asm volatile (
        "// xor5\n\t"
        "lop3.b32 %0, %2, %3, %4, 0x96;\n\t"
        "lop3.b32 %0, %0, %5, %6, 0x96;\n\t"
        "lop3.b32 %1, %7, %8, %9, 0x96;\n\t"
        "lop3.b32 %1, %1, %10, %11, 0x96;"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(b.x), "r"(c.x),"r"(d.x),"r"(e.x),
          "r"(a.y), "r"(b.y), "r"(c.y),"r"(d.y),"r"(e.y));
    return result;
#else
    return a ^ b ^ c ^ d ^ e;
#endif
}

DEV_INLINE uint2 xor3(const uint2 a, const uint2 b, const uint2 c)
{
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
    uint2 result;
    asm volatile (
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
    asm volatile (
        "// chi\n\t"
        "lop3.b32 %0, %2, %3, %4, 0xD2;\n\t"
        "lop3.b32 %1, %5, %6, %7, 0xD2;"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(b.x), "r"(c.x),  // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
          "r"(a.y), "r"(b.y), "r"(c.y)); // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
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
    for (uint32_t i = 9; i < 25; i++)
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

    for (int i = 0; i < 12; ++i)
        state[i] = s[i];
}

DEV_INLINE uint64_t keccak_f1600_final(uint2* state)
{
    uint2 s[25];
    uint2 t[5], u, v;
    const uint2 u2zero = make_uint2(0, 0);

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

    for (uint32_t i = 8; i < 25; i++)
    {
        s[i] = make_uint2(0, 0);
    }
    s[8].x = 1;
    s[8].y = 0x80000000;

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
