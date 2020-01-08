// Copyright 2017 Yurio Miyazawa (a.k.a zawawa) <me@yurio.net>
//
// This file is part of Gateless Gate Sharp.
//
// Gateless Gate Sharp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Gateless Gate Sharp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Gateless Gate Sharp.  If not, see <http://www.gnu.org/licenses/>.



#if (defined(__Tahiti__) || defined(__Pitcairn__) || defined(__Capeverde__) || defined(__Oland__) || defined(__Hainan__))
#define LEGACY
#endif

#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#if defined(cl_amd_media_ops)
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#elif defined(cl_nv_pragma_unroll)
uint amd_bitalign(uint src0, uint src1, uint src2)
{
    uint dest;
    asm("shf.r.wrap.b32 %0, %2, %1, %3;" : "=r"(dest) : "r"(src0), "r"(src1), "r"(src2));
    return dest;
}
#else
#define amd_bitalign(src0, src1, src2) ((uint) (((((ulong)(src0)) << 32) | (ulong)(src1)) >> ((src2) & 31)))
#endif

#if WORKSIZE % 4 != 0
#error "WORKSIZE has to be a multiple of 4"
#endif

#define FNV_PRIME 0x01000193U

static __constant uint2 const Keccak_f1600_RC[24] = {
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

#ifdef cl_amd_media_ops

#define barrier(x) mem_fence(x)

#define ROTL64_1(x, y) amd_bitalign((x), (x).s10, 32 - (y))
#define ROTL64_2(x, y) amd_bitalign((x).s10, (x), 32 - (y))

#else

#define ROTL64_1(x, y) as_uint2(rotate(as_ulong(x), (ulong)(y)))
#define ROTL64_2(x, y) ROTL64_1(x, (y) + 32)

#endif


#define KECCAKF_1600_RND(a, i, outsz) do { \
    const uint2 m0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20] ^ ROTL64_1(a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22], 1);\
    const uint2 m1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21] ^ ROTL64_1(a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23], 1);\
    const uint2 m2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22] ^ ROTL64_1(a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24], 1);\
    const uint2 m3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23] ^ ROTL64_1(a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20], 1);\
    const uint2 m4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24] ^ ROTL64_1(a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21], 1);\
    \
    const uint2 tmp = a[1]^m0;\
    \
    a[0] ^= m4;\
    a[5] ^= m4; \
    a[10] ^= m4; \
    a[15] ^= m4; \
    a[20] ^= m4; \
    \
    a[6] ^= m0; \
    a[11] ^= m0; \
    a[16] ^= m0; \
    a[21] ^= m0; \
    \
    a[2] ^= m1; \
    a[7] ^= m1; \
    a[12] ^= m1; \
    a[17] ^= m1; \
    a[22] ^= m1; \
    \
    a[3] ^= m2; \
    a[8] ^= m2; \
    a[13] ^= m2; \
    a[18] ^= m2; \
    a[23] ^= m2; \
    \
    a[4] ^= m3; \
    a[9] ^= m3; \
    a[14] ^= m3; \
    a[19] ^= m3; \
    a[24] ^= m3; \
    \
    a[1] = ROTL64_2(a[6], 12);\
    a[6] = ROTL64_1(a[9], 20);\
    a[9] = ROTL64_2(a[22], 29);\
    a[22] = ROTL64_2(a[14], 7);\
    a[14] = ROTL64_1(a[20], 18);\
    a[20] = ROTL64_2(a[2], 30);\
    a[2] = ROTL64_2(a[12], 11);\
    a[12] = ROTL64_1(a[13], 25);\
    a[13] = ROTL64_1(a[19],  8);\
    a[19] = ROTL64_2(a[23], 24);\
    a[23] = ROTL64_2(a[15], 9);\
    a[15] = ROTL64_1(a[4], 27);\
    a[4] = ROTL64_1(a[24], 14);\
    a[24] = ROTL64_1(a[21],  2);\
    a[21] = ROTL64_2(a[8], 23);\
    a[8] = ROTL64_2(a[16], 13);\
    a[16] = ROTL64_2(a[5], 4);\
    a[5] = ROTL64_1(a[3], 28);\
    a[3] = ROTL64_1(a[18], 21);\
    a[18] = ROTL64_1(a[17], 15);\
    a[17] = ROTL64_1(a[11], 10);\
    a[11] = ROTL64_1(a[7],  6);\
    a[7] = ROTL64_1(a[10],  3);\
    a[10] = ROTL64_1(tmp,  1);\
    \
    uint2 m5 = a[0]; uint2 m6 = a[1]; a[0] = bitselect(a[0]^a[2],a[0],a[1]); \
    a[0] ^= as_uint2(Keccak_f1600_RC[i]); \
    if (outsz > 1) { \
        a[1] = bitselect(a[1]^a[3],a[1],a[2]); a[2] = bitselect(a[2]^a[4],a[2],a[3]); a[3] = bitselect(a[3]^m5,a[3],a[4]); a[4] = bitselect(a[4]^m6,a[4],m5);\
        if (outsz > 4) { \
            m5 = a[5]; m6 = a[6]; a[5] = bitselect(a[5]^a[7],a[5],a[6]); a[6] = bitselect(a[6]^a[8],a[6],a[7]); a[7] = bitselect(a[7]^a[9],a[7],a[8]); a[8] = bitselect(a[8]^m5,a[8],a[9]); a[9] = bitselect(a[9]^m6,a[9],m5);\
            if (outsz > 8) { \
                m5 = a[10]; m6 = a[11]; a[10] = bitselect(a[10]^a[12],a[10],a[11]); a[11] = bitselect(a[11]^a[13],a[11],a[12]); a[12] = bitselect(a[12]^a[14],a[12],a[13]); a[13] = bitselect(a[13]^m5,a[13],a[14]); a[14] = bitselect(a[14]^m6,a[14],m5);\
                m5 = a[15]; m6 = a[16]; a[15] = bitselect(a[15]^a[17],a[15],a[16]); a[16] = bitselect(a[16]^a[18],a[16],a[17]); a[17] = bitselect(a[17]^a[19],a[17],a[18]); a[18] = bitselect(a[18]^m5,a[18],a[19]); a[19] = bitselect(a[19]^m6,a[19],m5);\
                m5 = a[20]; m6 = a[21]; a[20] = bitselect(a[20]^a[22],a[20],a[21]); a[21] = bitselect(a[21]^a[23],a[21],a[22]); a[22] = bitselect(a[22]^a[24],a[22],a[23]); a[23] = bitselect(a[23]^m5,a[23],a[24]); a[24] = bitselect(a[24]^m6,a[24],m5);\
            } \
        } \
    } \
 } while(0)


#define KECCAK_PROCESS(st, in_size, out_size)    do { \
    for (int r = 0; r < 24; ++r) { \
        int os = (r < 23 ? 25 : (out_size));\
        KECCAKF_1600_RND(st, r, os); \
    } \
} while(0)


#define fnv(x, y)        ((x) * FNV_PRIME ^ (y))
#define fnv_reduce(v)    fnv(fnv(fnv(v.x, v.y), v.z), v.w)

typedef union {
    uint uints[128 / sizeof(uint)];
    ulong ulongs[128 / sizeof(ulong)];
    uint2 uint2s[128 / sizeof(uint2)];
    uint4 uint4s[128 / sizeof(uint4)];
    uint8 uint8s[128 / sizeof(uint8)];
    uint16 uint16s[128 / sizeof(uint16)];
    ulong8 ulong8s[128 / sizeof(ulong8)];
} hash128_t;


typedef union {
    ulong8 ulong8s[1];
    ulong4 ulong4s[2];
    uint2 uint2s[8];
    uint4 uint4s[4];
    uint8 uint8s[2];
    uint16 uint16s[1];
    ulong ulongs[8];
    uint  uints[16];
} compute_hash_share;


#ifdef LEGACY

#define MIX(x) \
do { \
    if (get_local_id(0) == lane_idx) { \
        uint s = mix.s0; \
        s = select(mix.s1, s, (x) != 1); \
        s = select(mix.s2, s, (x) != 2); \
        s = select(mix.s3, s, (x) != 3); \
        s = select(mix.s4, s, (x) != 4); \
        s = select(mix.s5, s, (x) != 5); \
        s = select(mix.s6, s, (x) != 6); \
        s = select(mix.s7, s, (x) != 7); \
        buffer[hash_id] = fnv(init0 ^ (a + x), s) % dag_size; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    mix = fnv(mix, g_dag[buffer[hash_id]].uint8s[thread_id]); \
} while(0)

#else

#define MIX(x) \
do { \
    uint s = mix.s0; \
    s = select(mix.s1, s, (x) != 1); \
    s = select(mix.s2, s, (x) != 2); \
    s = select(mix.s3, s, (x) != 3); \
    s = select(mix.s4, s, (x) != 4); \
    s = select(mix.s5, s, (x) != 5); \
    s = select(mix.s6, s, (x) != 6); \
    s = select(mix.s7, s, (x) != 7); \
    buffer[get_local_id(0)] = fnv(init0 ^ (a + x), s) % dag_size; \
    mix = fnv(mix, g_dag[buffer[lane_idx]].uint8s[thread_id]); \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
} while(0)

#endif

// NOTE: This struct must match the one defined in CLMiner.cpp
struct SearchResults {
    struct {
        uint gid;
        uint mix[8];
        uint pad[7]; // pad to 16 words for easy indexing
    } rslt[MAX_OUTPUTS];
    uint count;
    uint hashCount;
    uint abort;
};

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
    __global struct SearchResults* restrict g_output,
    __constant uint2 const* g_header,
    __global ulong8 const* _g_dag,
    uint dag_size,
    ulong start_nonce,
    ulong target
)
{
#ifdef FAST_EXIT
    if (g_output->abort)
        return;
#endif

    __global hash128_t const* g_dag = (__global hash128_t const*) _g_dag;

    const uint thread_id = get_local_id(0) % 4;
    const uint hash_id = get_local_id(0) / 4;
    const uint gid = get_global_id(0);

    __local compute_hash_share sharebuf[WORKSIZE / 4];
#ifdef LEGACY
    __local uint buffer[WORKSIZE / 4];
#else
    __local uint buffer[WORKSIZE];
#endif
    __local compute_hash_share * const share = sharebuf + hash_id;

    // sha3_512(header .. nonce)
    uint2 state[25];
    state[0] = g_header[0];
    state[1] = g_header[1];
    state[2] = g_header[2];
    state[3] = g_header[3];
    state[4] = as_uint2(start_nonce + gid);
    state[5] = as_uint2(0x0000000000000001UL);
    state[6] = (uint2)(0);
    state[7] = (uint2)(0);
    state[8] = as_uint2(0x8000000000000000UL);
    state[9] = (uint2)(0);
    state[10] = (uint2)(0);
    state[11] = (uint2)(0);
    state[12] = (uint2)(0);
    state[13] = (uint2)(0);
    state[14] = (uint2)(0);
    state[15] = (uint2)(0);
    state[16] = (uint2)(0);
    state[17] = (uint2)(0);
    state[18] = (uint2)(0);
    state[19] = (uint2)(0);
    state[20] = (uint2)(0);
    state[21] = (uint2)(0);
    state[22] = (uint2)(0);
    state[23] = (uint2)(0);
    state[24] = (uint2)(0);

    uint2 mixhash[4];

    for (int pass = 0; pass < 2; ++pass) {
        KECCAK_PROCESS(state, select(5, 12, pass != 0), select(8, 1, pass != 0));
        if (pass > 0)
            break;

        uint init0;
        uint8 mix;

#pragma unroll 1
        for (uint tid = 0; tid < 4; tid++) {
            if (tid == thread_id) {
                share->uint2s[0] = state[0];
                share->uint2s[1] = state[1];
                share->uint2s[2] = state[2];
                share->uint2s[3] = state[3];
                share->uint2s[4] = state[4];
                share->uint2s[5] = state[5];
                share->uint2s[6] = state[6];
                share->uint2s[7] = state[7];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            mix = share->uint8s[thread_id & 1];
            init0 = share->uints[0];

            barrier(CLK_LOCAL_MEM_FENCE);

#ifndef LEGACY
#pragma unroll 1
#endif
            for (uint a = 0; a < ACCESSES; a += 8) {
                const uint lane_idx = 4 * hash_id + a / 8 % 4;
                for (uint x = 0; x < 8; ++x)
                    MIX(x);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            share->uint2s[thread_id] = (uint2)(fnv_reduce(mix.lo), fnv_reduce(mix.hi));

            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid == thread_id) {
                state[8] = share->uint2s[0];
                state[9] = share->uint2s[1];
                state[10] = share->uint2s[2];
                state[11] = share->uint2s[3];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        mixhash[0] = state[8];
        mixhash[1] = state[9];
        mixhash[2] = state[10];
        mixhash[3] = state[11];

        state[12] = as_uint2(0x0000000000000001UL);
        state[13] = (uint2)(0);
        state[14] = (uint2)(0);
        state[15] = (uint2)(0);
        state[16] = as_uint2(0x8000000000000000UL);
        state[17] = (uint2)(0);
        state[18] = (uint2)(0);
        state[19] = (uint2)(0);
        state[20] = (uint2)(0);
        state[21] = (uint2)(0);
        state[22] = (uint2)(0);
        state[23] = (uint2)(0);
        state[24] = (uint2)(0);
    }

#ifdef FAST_EXIT
    if (get_local_id(0) == 0)
        atomic_inc(&g_output->hashCount);
#endif

    if (as_ulong(as_uchar8(state[0]).s76543210) <= target) {
#ifdef FAST_EXIT
        atomic_inc(&g_output->abort);
#endif
        uint slot = min(MAX_OUTPUTS - 1u, atomic_inc(&g_output->count));
        g_output->rslt[slot].gid = gid;
        g_output->rslt[slot].mix[0] = mixhash[0].s0;
        g_output->rslt[slot].mix[1] = mixhash[0].s1;
        g_output->rslt[slot].mix[2] = mixhash[1].s0;
        g_output->rslt[slot].mix[3] = mixhash[1].s1;
        g_output->rslt[slot].mix[4] = mixhash[2].s0;
        g_output->rslt[slot].mix[5] = mixhash[2].s1;
        g_output->rslt[slot].mix[6] = mixhash[3].s0;
        g_output->rslt[slot].mix[7] = mixhash[3].s1;
    }
}

typedef union _Node {
    uint dwords[16];
    uint2 qwords[8];
    uint4 dqwords[4];
} Node;

static void SHA3_512(uint2 *s)
{
    uint2 st[25];

    for (uint i = 0; i < 8; ++i)
        st[i] = s[i];

    st[8] = (uint2)(0x00000001, 0x80000000);

    for (uint i = 9; i != 25; ++i)
        st[i] = (uint2)(0);

    KECCAK_PROCESS(st, 8, 8);

    for (uint i = 0; i < 8; ++i)
        s[i] = st[i];
}

__kernel void GenerateDAG(uint start, __global const uint16 *_Cache, __global uint16 *_DAG, uint light_size)
{
    __global const Node *Cache = (__global const Node *) _Cache;
    __global Node *DAG = (__global Node *) _DAG;
    uint NodeIdx = start + get_global_id(0);

    Node DAGNode = Cache[NodeIdx % light_size];

    DAGNode.dwords[0] ^= NodeIdx;
    SHA3_512(DAGNode.qwords);

    for (uint i = 0; i < 256; ++i) {
        uint ParentIdx = fnv(NodeIdx ^ i, DAGNode.dwords[i & 15]) % light_size;
        __global const Node *ParentNode = Cache + ParentIdx;

#pragma unroll
        for (uint x = 0; x < 4; ++x) {
                DAGNode.dqwords[x] *= (uint4)(FNV_PRIME);
                DAGNode.dqwords[x] ^= ParentNode->dqwords[x];
        }
    }

    SHA3_512(DAGNode.qwords);

    //if (NodeIdx < DAG_SIZE)
    DAG[NodeIdx] = DAGNode;
}
