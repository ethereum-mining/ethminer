#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA 1
#define OPENCL_PLATFORM_AMD 2
#define OPENCL_PLATFORM_CLOVER 3

#ifndef MAX_OUTPUTS
#define MAX_OUTPUTS 63U
#endif

#ifndef PLATFORM
#define PLATFORM OPENCL_PLATFORM_AMD
#endif

#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#define HASHES_PER_GROUP (GROUP_SIZE / PROGPOW_LANES)

typedef struct
{
    uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

// Implementation based on:
// https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c

__constant const uint32_t keccakf_rndc[24] = {0x00000001, 0x00008082, 0x0000808a, 0x80008000,
    0x0000808b, 0x80000001, 0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080, 0x0000800a, 0x8000000a,
    0x80008081, 0x00008080, 0x80000001, 0x80008008};

// Implementation of the Keccakf transformation with a width of 800
void keccak_f800_round(uint32_t st[25], const int r)
{
    const uint32_t keccakf_rotc[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const uint32_t keccakf_piln[24] = {
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

    uint32_t t, bc[5];
    // Theta
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (int i = 0; i < 5; i++)
    {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1u);
        for (uint32_t j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (int i = 0; i < 24; i++)
    {
        uint32_t j = keccakf_piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, keccakf_rotc[i]);
        t = bc[0];
    }

    //  Chi
    for (uint32_t j = 0; j < 25; j += 5)
    {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_rndc[r];
}

// Keccak - implemented as a variant of SHAKE
// The width is 800, with a bitrate of 576, a capacity of 224, and no padding
// Only need 64 bits of output for mining
uint64_t keccak_f800(__constant hash32_t const* g_header, uint64_t seed, hash32_t digest)
{
    uint32_t st[25];

    for (int i = 0; i < 25; i++)
        st[i] = 0;
    for (int i = 0; i < 8; i++)
        st[i] = g_header->uint32s[i];
    st[8] = seed;
    st[9] = seed >> 32;
    for (int i = 0; i < 8; i++)
        st[10 + i] = digest.uint32s[i];

    for (int r = 0; r < 21; r++)
    {
        keccak_f800_round(st, r);
    }
    // last round can be simplified due to partial output
    keccak_f800_round(st, 21);

    // Byte swap so byte 0 of hash is MSB of result
    uint64_t res = (uint64_t)st[1] << 32 | st[0];
    return as_ulong(as_uchar8(res).s76543210);
}

#define fnv1a(h, d) (h = (h ^ d) * 0x1000193)

typedef struct
{
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
uint32_t kiss99(kiss99_t* st)
{
    st->z = 36969 * (st->z & 65535) + (st->z >> 16);
    st->w = 18000 * (st->w & 65535) + (st->w >> 16);
    uint32_t MWC = ((st->z << 16) + st->w);
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    st->jcong = 69069 * st->jcong + 1234567;
    return ((MWC ^ st->jcong) + st->jsr);
}

void fill_mix(uint64_t seed, uint32_t lane_id, uint32_t mix[PROGPOW_REGS])
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = 0x811c9dc5;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, seed);
    st.w = fnv1a(fnv_hash, seed >> 32);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
#pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(&st);
}

typedef struct
{
    uint32_t uint32s[PROGPOW_LANES];
    uint64_t uint64s[PROGPOW_LANES / 2];
} shuffle_t;

// NOTE: This struct must match the one defined in CLMiner.cpp
struct SearchResults
{
    struct
    {
        uint gid;
        uint mix[8];
        uint pad[7];  // pad to 16 words for easy indexing
    } rslt[MAX_OUTPUTS];
    uint count;
    uint hashCount;
    uint abort;
};


#if PLATFORM != OPENCL_PLATFORM_NVIDIA  // use maxrregs on nv
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
#endif
__kernel void
ethash_search(__global struct SearchResults* restrict g_output, __constant hash32_t const* g_header,
    __global dag_t const* g_dag, ulong start_nonce, ulong target, uint hack_false)
{
    if (g_output->abort)
        return;

    __local shuffle_t share[HASHES_PER_GROUP];
    __local uint32_t c_dag[PROGPOW_CACHE_WORDS];

    uint32_t const lid = get_local_id(0);
    uint32_t const gid = get_global_id(0);
    uint64_t const nonce = start_nonce + gid;

    const uint32_t lane_id = lid & (PROGPOW_LANES - 1);
    const uint32_t group_id = lid / PROGPOW_LANES;

    // Load the first portion of the DAG into the cache
    for (uint32_t word = lid * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS;
         word += GROUP_SIZE * PROGPOW_DAG_LOADS)
    {
        dag_t load = g_dag[word / PROGPOW_DAG_LOADS];
        for (int i = 0; i < PROGPOW_DAG_LOADS; i++)
            c_dag[word + i] = load.s[i];
    }

    hash32_t digest;
    for (int i = 0; i < 8; i++)
        digest.uint32s[i] = 0;
    // keccak(header..nonce)
    uint64_t seed = keccak_f800(g_header, start_nonce + gid, digest);

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++)
    {
        uint32_t mix[PROGPOW_REGS];

        // share the hash's seed across all lanes
        if (lane_id == h)
            share[group_id].uint64s[0] = seed;
        barrier(CLK_LOCAL_MEM_FENCE);
        uint64_t hash_seed = share[group_id].uint64s[0];

        // initialize mix for all lanes
        fill_mix(hash_seed, lane_id, mix);

#pragma unroll 1
        for (uint32_t l = 0; l < PROGPOW_CNT_DAG; l++)
            progPowLoop(l, mix, g_dag, c_dag, share[0].uint64s, hack_false);

        // Reduce mix data to a per-lane 32-bit digest
        uint32_t mix_hash = 0x811c9dc5;
#pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(mix_hash, mix[i]);

        // Reduce all lanes to a single 256-bit digest
        hash32_t digest_temp;
        for (int i = 0; i < 8; i++)
            digest_temp.uint32s[i] = 0x811c9dc5;
        share[group_id].uint32s[lane_id] = mix_hash;
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int i = 0; i < PROGPOW_LANES; i++)
            fnv1a(digest_temp.uint32s[i % 8], share[group_id].uint32s[i]);
        if (h == lane_id)
            digest = digest_temp;
    }

    if (lid == 0)
        atomic_inc(&g_output->hashCount);

    // keccak(header .. keccak(header..nonce) .. digest);
    if (keccak_f800(g_header, seed, digest) <= target)
    {
        uint slot = atomic_inc(&g_output->count);
        if (slot < MAX_OUTPUTS)
        {
            g_output->rslt[slot].gid = gid;
            for (int i = 0; i < 8; i++)
                g_output->rslt[slot].mix[i] = digest.uint32s[i];
        }
        atomic_inc(&g_output->abort);
    }
}


//
// DAG calculation logic
//


#ifndef LIGHT_WORDS
#define LIGHT_WORDS 262139
#endif

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64 / 4)

#define FNV_PRIME 0x01000193

__constant uint2 const Keccak_f1600_RC[24] = {
    (uint2)(0x00000001, 0x00000000),
    (uint2)(0x00008082, 0x00000000),
    (uint2)(0x0000808a, 0x80000000),
    (uint2)(0x80008000, 0x80000000),
    (uint2)(0x0000808b, 0x00000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008009, 0x80000000),
    (uint2)(0x0000008a, 0x00000000),
    (uint2)(0x00000088, 0x00000000),
    (uint2)(0x80008009, 0x00000000),
    (uint2)(0x8000000a, 0x00000000),
    (uint2)(0x8000808b, 0x00000000),
    (uint2)(0x0000008b, 0x80000000),
    (uint2)(0x00008089, 0x80000000),
    (uint2)(0x00008003, 0x80000000),
    (uint2)(0x00008002, 0x80000000),
    (uint2)(0x00000080, 0x80000000),
    (uint2)(0x0000800a, 0x00000000),
    (uint2)(0x8000000a, 0x80000000),
    (uint2)(0x80008081, 0x80000000),
    (uint2)(0x00008080, 0x80000000),
    (uint2)(0x80000001, 0x00000000),
    (uint2)(0x80008008, 0x80000000),
};

#if PLATFORM == OPENCL_PLATFORM_NVIDIA && COMPUTE >= 35
static uint2 ROL2(const uint2 a, const int offset)
{
    uint2 result;
    if (offset >= 32)
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else
    {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}
#elif PLATFORM == OPENCL_PLATFORM_AMD
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
static uint2 ROL2(const uint2 vv, const int r)
{
    if (r <= 32)
    {
        return amd_bitalign((vv).xy, (vv).yx, 32 - r);
    }
    else
    {
        return amd_bitalign((vv).yx, (vv).xy, 64 - r);
    }
}
#else
static uint2 ROL2(const uint2 v, const int n)
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

static void chi(uint2* a, const uint n, const uint2* t)
{
    a[n + 0] = bitselect(t[n + 0] ^ t[n + 2], t[n + 0], t[n + 1]);
    a[n + 1] = bitselect(t[n + 1] ^ t[n + 3], t[n + 1], t[n + 2]);
    a[n + 2] = bitselect(t[n + 2] ^ t[n + 4], t[n + 2], t[n + 3]);
    a[n + 3] = bitselect(t[n + 3] ^ t[n + 0], t[n + 3], t[n + 4]);
    a[n + 4] = bitselect(t[n + 4] ^ t[n + 1], t[n + 4], t[n + 0]);
}

static void keccak_f1600_round(uint2* a, uint r)
{
    uint2 t[25];
    uint2 u;

    // Theta
    t[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    t[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    t[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    t[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    t[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    u = t[4] ^ ROL2(t[1], 1);
    a[0] ^= u;
    a[5] ^= u;
    a[10] ^= u;
    a[15] ^= u;
    a[20] ^= u;
    u = t[0] ^ ROL2(t[2], 1);
    a[1] ^= u;
    a[6] ^= u;
    a[11] ^= u;
    a[16] ^= u;
    a[21] ^= u;
    u = t[1] ^ ROL2(t[3], 1);
    a[2] ^= u;
    a[7] ^= u;
    a[12] ^= u;
    a[17] ^= u;
    a[22] ^= u;
    u = t[2] ^ ROL2(t[4], 1);
    a[3] ^= u;
    a[8] ^= u;
    a[13] ^= u;
    a[18] ^= u;
    a[23] ^= u;
    u = t[3] ^ ROL2(t[0], 1);
    a[4] ^= u;
    a[9] ^= u;
    a[14] ^= u;
    a[19] ^= u;
    a[24] ^= u;

    // Rho Pi

    t[0] = a[0];
    t[10] = ROL2(a[1], 1);
    t[20] = ROL2(a[2], 62);
    t[5] = ROL2(a[3], 28);
    t[15] = ROL2(a[4], 27);

    t[16] = ROL2(a[5], 36);
    t[1] = ROL2(a[6], 44);
    t[11] = ROL2(a[7], 6);
    t[21] = ROL2(a[8], 55);
    t[6] = ROL2(a[9], 20);

    t[7] = ROL2(a[10], 3);
    t[17] = ROL2(a[11], 10);
    t[2] = ROL2(a[12], 43);
    t[12] = ROL2(a[13], 25);
    t[22] = ROL2(a[14], 39);

    t[23] = ROL2(a[15], 41);
    t[8] = ROL2(a[16], 45);
    t[18] = ROL2(a[17], 15);
    t[3] = ROL2(a[18], 21);
    t[13] = ROL2(a[19], 8);

    t[14] = ROL2(a[20], 18);
    t[24] = ROL2(a[21], 2);
    t[9] = ROL2(a[22], 61);
    t[19] = ROL2(a[23], 56);
    t[4] = ROL2(a[24], 14);

    // Chi
    chi(a, 0, t);

    // Iota
    a[0] ^= Keccak_f1600_RC[r];

    chi(a, 5, t);
    chi(a, 10, t);
    chi(a, 15, t);
    chi(a, 20, t);
}

static void keccak_f1600_no_absorb(uint2* a, uint out_size, uint isolate)
{
    // Originally I unrolled the first and last rounds to interface
    // better with surrounding code, however I haven't done this
    // without causing the AMD compiler to blow up the VGPR usage.


    // uint o = 25;
    for (uint r = 0; r < 24;)
    {
        // This dynamic branch stops the AMD compiler unrolling the loop
        // and additionally saves about 33% of the VGPRs, enough to gain another
        // wavefront. Ideally we'd get 4 in flight, but 3 is the best I can
        // massage out of the compiler. It doesn't really seem to matter how
        // much we try and help the compiler save VGPRs because it seems to throw
        // that information away, hence the implementation of keccak here
        // doesn't bother.
        if (isolate)
        {
            keccak_f1600_round(a, r++);
            // if (r == 23) o = out_size;
        }
    }


    // final round optimised for digest size
    // keccak_f1600_round(a, 23, out_size);
}

#define copy(dst, src, count)         \
    for (uint i = 0; i != count; ++i) \
    {                                 \
        (dst)[i] = (src)[i];          \
    }

static uint fnv(uint x, uint y)
{
    return x * FNV_PRIME ^ y;
}

static uint4 fnv4(uint4 x, uint4 y)
{
    return x * FNV_PRIME ^ y;
}

typedef union
{
    uint words[64 / sizeof(uint)];
    uint2 uint2s[64 / sizeof(uint2)];
    uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
    uint words[200 / sizeof(uint)];
    uint2 uint2s[200 / sizeof(uint2)];
    uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;

typedef struct
{
    uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

static void SHA3_512(uint2* s, uint isolate)
{
    for (uint i = 8; i != 25; ++i)
    {
        s[i] = (uint2){0, 0};
    }
    s[8].x = 0x00000001;
    s[8].y = 0x80000000;
    keccak_f1600_no_absorb(s, 8, isolate);
}

__kernel void ethash_calculate_dag_item(
    uint start, __global hash64_t const* g_light, __global hash64_t* g_dag, uint isolate)
{
    uint const node_index = start + get_global_id(0);
    if (node_index * sizeof(hash64_t) >= PROGPOW_DAG_BYTES)
        return;

    hash200_t dag_node;
    copy(dag_node.uint4s, g_light[node_index % LIGHT_WORDS].uint4s, 4);
    dag_node.words[0] ^= node_index;
    SHA3_512(dag_node.uint2s, isolate);

    for (uint i = 0; i != ETHASH_DATASET_PARENTS; ++i)
    {
        uint parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % LIGHT_WORDS;

        for (uint w = 0; w != 4; ++w)
        {
            dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], g_light[parent_index].uint4s[w]);
        }
    }
    SHA3_512(dag_node.uint2s, isolate);
    copy(g_dag[node_index].uint4s, dag_node.uint4s, 4);
}
