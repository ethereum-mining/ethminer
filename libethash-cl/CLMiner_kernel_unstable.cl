#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD     2
#define OPENCL_PLATFORM_CLOVER  3

#ifndef ACCESSES
    #define ACCESSES 64
#endif

#ifndef GROUP_SIZE
    #define GROUP_SIZE 128
#endif

#ifndef MAX_OUTPUTS
    #define MAX_OUTPUTS 63U
#endif

#ifndef PLATFORM
    #define PLATFORM OPENCL_PLATFORM_AMD
#endif

#ifndef DAG_SIZE
    #define DAG_SIZE 8388593
#endif

#ifndef LIGHT_SIZE
    #define LIGHT_SIZE 262139
#endif

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64/4)

// Either 8, 4, 2, or 1
#ifndef THREADS_PER_HASH
    #define THREADS_PER_HASH 1
#endif
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

// Check for valid THREADS_PER_HASH param
#if THREADS_PER_HASH == 1
    #define LN_THREAD_PER_HASH  0
    #define ACCESS_INCREMENT    0
    #define ACCESS_SHIFT        0
#elif THREADS_PER_HASH == 2
    #define LN_THREAD_PER_HASH  1
    #define ACCESS_INCREMENT    16
    #define ACCESS_SHIFT        4
#elif THREADS_PER_HASH == 4
    #define LN_THREAD_PER_HASH  2
    #define ACCESS_INCREMENT    8
    #define ACCESS_SHIFT        3
#elif THREADS_PER_HASH == 8
    #define LN_THREAD_PER_HASH  3
    #define ACCESS_INCREMENT    4
    #define ACCESS_SHIFT        2
#else
	#error "Invalid THREADS_PER_HASH, it needs to be 1, 2, 4 or 8"
#endif

#define FNV_PRIME	0x01000193

#define ROTL64_1(x, y)   as_ulong(rotate(as_ulong(x), (ulong)(y)))
#define ROTL64_2(x, y)   ROTL64_1(x, (y) + 32)

#define FNV(x, y)        ((x) * FNV_PRIME ^ (y))
#define FNV_REDUCE(v)    FNV(FNV(FNV(v.x, v.y), v.z), v.w)

#define mem_fence(x) barrier(x)


__constant ulong const Keccak_f1600_RC[24] = {
    (0x0000000000000001UL),
    (0x0000000000008082UL),
    (0x800000000000808AUL),
    (0x8000000080008000UL),
    (0x000000000000808BUL),
    (0x0000000080000001UL),
    (0x8000000080008081UL),
    (0x8000000000008009UL),
    (0x000000000000008AUL),
    (0x0000000000000088UL),
    (0x0000000080008009UL),
    (0x000000008000000AUL),
    (0x000000008000808BUL),
    (0x800000000000008BUL),
    (0x8000000000008089UL),
    (0x8000000000008003UL),
    (0x8000000000008002UL),
    (0x8000000000000080UL),
    (0x000000000000800AUL),
    (0x800000008000000AUL),
    (0x8000000080008081UL),
    (0x8000000000008080UL),
    (0x0000000080000001UL),
    (0x8000000080008008UL),
};
 
static void keccak_f1600_round(ulong* a, const uint r, const uint outsz) {
    const ulong m0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20] ^ ROTL64_1(a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22], 1);
    const ulong m1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21] ^ ROTL64_1(a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23], 1);
    const ulong m2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22] ^ ROTL64_1(a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24], 1);
    const ulong m3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23] ^ ROTL64_1(a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20], 1);
    const ulong m4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24] ^ ROTL64_1(a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21], 1);
 
    const ulong tmp = a[1]^m0;
 
    a[0] ^= m4;
    a[5] ^= m4;
    a[10] ^= m4;
    a[15] ^= m4;
    a[20] ^= m4;
 
    a[6] ^= m0;
    a[11] ^= m0;
    a[16] ^= m0;
    a[21] ^= m0;
 
    a[2] ^= m1;
    a[7] ^= m1;
    a[12] ^= m1;
    a[17] ^= m1;
    a[22] ^= m1;
 
    a[3] ^= m2;
    a[8] ^= m2;
    a[13] ^= m2;
    a[18] ^= m2;
    a[23] ^= m2;
 
    a[4] ^= m3;
    a[9] ^= m3;
    a[14] ^= m3;
    a[19] ^= m3;
    a[24] ^= m3;
 
    a[1] = ROTL64_2(a[6], 12);
    a[6] = ROTL64_1(a[9], 20);
    a[9] = ROTL64_2(a[22], 29);
    a[22] = ROTL64_2(a[14], 7);
    a[14] = ROTL64_1(a[20], 18);
    a[20] = ROTL64_2(a[2], 30);
    a[2] = ROTL64_2(a[12], 11);
    a[12] = ROTL64_1(a[13], 25);
    a[13] = ROTL64_1(a[19],  8);
    a[19] = ROTL64_2(a[23], 24);
    a[23] = ROTL64_2(a[15], 9);
    a[15] = ROTL64_1(a[4], 27);
    a[4] = ROTL64_1(a[24], 14);
    a[24] = ROTL64_1(a[21],  2);
    a[21] = ROTL64_2(a[8], 23);
    a[8] = ROTL64_2(a[16], 13);
    a[16] = ROTL64_2(a[5], 4);
    a[5] = ROTL64_1(a[3], 28);
    a[3] = ROTL64_1(a[18], 21);
    a[18] = ROTL64_1(a[17], 15);
    a[17] = ROTL64_1(a[11], 10);
    a[11] = ROTL64_1(a[7],  6);
    a[7] = ROTL64_1(a[10],  3);
    a[10] = ROTL64_1(tmp,  1);
 
    ulong m5 = a[0];
    ulong m6 = a[1];
 
    a[0] = bitselect(a[0]^a[2],a[0],a[1]);
 
    a[0] ^= Keccak_f1600_RC[r];
 
    if (outsz > 1) {
        a[1] = bitselect(a[1]^a[3],a[1],a[2]);
        a[2] = bitselect(a[2]^a[4],a[2],a[3]);
        a[3] = bitselect(a[3]^m5,a[3],a[4]);
        a[4] = bitselect(a[4]^m6,a[4],m5);
 
        if (outsz > 4) {
            m5 = a[5];
            m6 = a[6];
            a[5] = bitselect(a[5]^a[7],a[5],a[6]);
            a[6] = bitselect(a[6]^a[8],a[6],a[7]);
            a[7] = bitselect(a[7]^a[9],a[7],a[8]);
            a[8] = bitselect(a[8]^m5,a[8],a[9]);
            a[9] = bitselect(a[9]^m6,a[9],m5);
 
            if (outsz > 8) {
                m5 = a[10];
                m6 = a[11];
                a[10] = bitselect(a[10]^a[12],a[10],a[11]);
                a[11] = bitselect(a[11]^a[13],a[11],a[12]);
                a[12] = bitselect(a[12]^a[14],a[12],a[13]);
                a[13] = bitselect(a[13]^m5,a[13],a[14]);
                a[14] = bitselect(a[14]^m6,a[14],m5);
 
                m5 = a[15];
                m6 = a[16];
                a[15] = bitselect(a[15]^a[17],a[15],a[16]);
                a[16] = bitselect(a[16]^a[18],a[16],a[17]);
                a[17] = bitselect(a[17]^a[19],a[17],a[18]);
                a[18] = bitselect(a[18]^m5,a[18],a[19]);
                a[19] = bitselect(a[19]^m6,a[19],m5);
 
                m5 = a[20];
                m6 = a[21];
                a[20] = bitselect(a[20]^a[22],a[20],a[21]);
                a[21] = bitselect(a[21]^a[23],a[21],a[22]);
                a[22] = bitselect(a[22]^a[24],a[22],a[23]);
                a[23] = bitselect(a[23]^m5,a[23],a[24]);
                a[24] = bitselect(a[24]^m6,a[24],m5);
            }
        }
    }
}
 
#define SHA3_512(s) { \
    for (uint i = 8; i != 25; ++i){ \
        s[i] = 0UL; \
    } \
    s[8] = 0x8000000000000001UL; \
    keccak_f1600(s, 8); \
}

#define keccak_f1600(a, outsz) { \
    for (uint r = 0; r < 23;) {\
        keccak_f1600_round(a, r++, 25); \
    }\
    keccak_f1600_round(a, 23, outsz);\
}

typedef struct
{
	ulong ulongs[32 / sizeof(ulong)];
} hash32_t;

typedef union {
    ulong   ulongs[64 / sizeof(ulong)];
    ulong4  ulong4s[64 / sizeof(ulong4)];
 
    uint    uints[64 / sizeof(uint)];
    uint2   uint2s[64 / sizeof(uint2)];
    uint4   uint4s[64 / sizeof(uint4)];
    uint8   uint8s[64 / sizeof(uint8)];
    uint16  uint16s[1];
} hash64_t;
 
typedef union
{
    ulong   ulongs[128 / sizeof(ulong)];
    ulong4  ulong4s[128 / sizeof(ulong4)];
    ulong8  ulong8s[128 / sizeof(ulong8)];
 
    uint    uints[128 / sizeof(uint)];
    uint2   uint2s[128 / sizeof(uint2)];
    uint4   uint4s[128 / sizeof(uint4)];
    uint8   uint8s[128 / sizeof(uint8)];
    uint16  uint16s[128 / sizeof(uint16)];
} hash128_t;
 
typedef union {
    ulong   ulongs[200 / sizeof(ulong)];
 
    uint    uints[200 / sizeof(uint)];
    uint2   uint2s[200 / sizeof(uint2)];
    uint8   uint8s[200 / sizeof(uint8)];
    uint16  uint16s[200 / sizeof(uint16)];
} hash200_t;


#if PLATFORM != OPENCL_PLATFORM_NVIDIA // use maxrregs on nv
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
#endif
__kernel void ethash_search(
    __global volatile uint* restrict g_output,
    __constant hash32_t const* g_header,
    __global hash128_t const* g_dag,
    ulong start_nonce,
    ulong target,
    uint isolate
    )
{

    uint const gid = get_global_id(0);
    uint const thread_id = gid & (THREADS_PER_HASH - 1);
    uint const hash_id = (gid % GROUP_SIZE) >> LN_THREAD_PER_HASH;
 
    __local hash64_t sharebuf[HASHES_PER_LOOP];
    __local hash64_t * const share = sharebuf + hash_id;
 
    ulong state[25];
 
    ((ulong4*)state)[0] = ((__constant ulong4*)g_header)[0];
 
    state[4] = start_nonce + gid;
 
    for (uint i = 6; i != 25; ++i) {
        state[i] = 0;
    }
 
    state[5] = 0x0000000000000001UL;
    state[8] = 0x8000000000000000UL;
 
    keccak_f1600(state, 8);

#if THREADS_PER_HASH == 1
    hash128_t mix;
    share[hash_id].uint16s[0] = ((uint16 *)state)[0];

    #pragma unroll
    for(uint i = 0; i < 4; i++) {
        mix.uint4s[i  ] = share[hash_id].uint4s[i];
        mix.uint4s[i+4] = share[hash_id].uint4s[i];
    }

    uint init0 = share[hash_id].uints[0];

    #pragma unroll
    for(uint i = 0; i < ACCESSES; i++) {
        uint p = FNV(i ^ init0, mix.uints[i & 31]) % DAG_SIZE;
  
        #pragma unroll
        for(uint j = 0; j < 8; j++) 
            mix.uint4s[j] = FNV(mix.uint4s[j], g_dag[p].uint4s[j]);
    }

    #pragma unroll
    for(uint i = 0; i < 8; i++) {
        share[hash_id].uints[i] = FNV_REDUCE(mix.uint4s[i]);
    }

    #pragma unroll
    for(uint i = 0; i < 4; i++) {
        (state + 8)[i] = share[hash_id].ulongs[i];
    }

#else
    #pragma unroll 1
    for (uint tid = 0; tid < THREADS_PER_HASH; tid++) {
        if (tid == thread_id) {
            share->uint16s[0] = ((uint16*)state)[0];
        }
        mem_fence(CLK_LOCAL_MEM_FENCE);
 
#if THREADS_PER_HASH == 2
        uint16 mix = share->uint16s[0];
#elif THREADS_PER_HASH == 4
        uint8 mix = share->uint8s[thread_id & 1];
#elif THREADS_PER_HASH == 8
        uint4 mix = share->uint4s[thread_id & 3];
#elif THREADS_PER_HASH == 16
        uint8 mix = share->uints[thread_id & 7];
#endif
        mem_fence(CLK_LOCAL_MEM_FENCE);
 
        uint init0 = share->uints[0];
        mem_fence(CLK_LOCAL_MEM_FENCE);
 
        #pragma unroll 1
        for (uint a = 0; a < 64; a += ACCESS_INCREMENT) {
            bool update_share = thread_id == ((a >> ACCESS_SHIFT) & (THREADS_PER_HASH - 1));
 
            #pragma unroll
            for (uint i = 0; i != ACCESS_INCREMENT; ++i) {
                if (update_share) {
                    share->uints[0] = FNV(init0 ^ (a + i), ((uint *)&mix)[i]) % DAG_SIZE;
                }
                mem_fence(CLK_LOCAL_MEM_FENCE);

#if THREADS_PER_HASH == 2
                #pragma unroll
                for(uint i = 0; i < 16; i++) 
                	((uint *)&mix)[i] = FNV(((uint *)&mix)[i], g_dag[share->uints[0]].uints[16*thread_id + i]);
#elif THREADS_PER_HASH == 4
                mix = FNV(mix, g_dag[share->uints[0]].uint8s[thread_id]);
#elif THREADS_PER_HASH == 8
                mix = FNV(mix, g_dag[share->uints[0]].uint4s[thread_id]);
#endif
                mem_fence(CLK_LOCAL_MEM_FENCE);
            }
        }
 
#if THREADS_PER_HASH == 2
        share->uint4s[thread_id] = (uint4)(
            FNV_REDUCE(mix.s0123),
            FNV_REDUCE(mix.s4567),
            FNV_REDUCE(mix.s89ab),
            FNV_REDUCE(mix.scdef)
        );
#elif THREADS_PER_HASH == 4
		    share->uint2s[thread_id] = (uint2)(FNV_REDUCE(mix.lo), FNV_REDUCE(mix.hi));
#elif THREADS_PER_HASH == 8
        share->uints[thread_id] = FNV_REDUCE(mix);
#endif
        mem_fence(CLK_LOCAL_MEM_FENCE);
 
        if (tid == thread_id) {
            ((ulong4*)state)[2] = share->ulong4s[0];
        }
        mem_fence(CLK_LOCAL_MEM_FENCE);
    }
 #endif

    for (uint i = 13; i != 25; ++i) {
        state[i] = 0;
    }
 
    state[12] = 0x0000000000000001UL;
    state[16] = 0x8000000000000000UL;
 
    keccak_f1600(state, 1);

    if (as_ulong(as_uchar8(state[0]).s76543210) < target)
    {
        uint slot = min(MAX_OUTPUTS, atomic_inc(&g_output[0]) + 1);
        g_output[slot] = gid;
    }
}

__kernel void ethash_calculate_dag_item(uint start, __global hash64_t const* g_light, __global hash64_t * g_dag, uint isolate)
{
    uint const node_index = start + get_global_id(0);
    if (node_index > DAG_SIZE * 2) return;

    hash200_t dag_node;
    dag_node.uint16s[0] = g_light[node_index % LIGHT_SIZE].uint16s[0];
    dag_node.uints[0] ^= node_index;
    SHA3_512(dag_node.ulongs);


    for (uint i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
        uint parent_index = FNV(node_index ^ i, dag_node.uints[i % NODE_WORDS]) % LIGHT_SIZE;
        dag_node.uint16s[0] = FNV(dag_node.uint16s[0], g_light[parent_index].uint16s[0]);
    }

    SHA3_512(dag_node.ulongs);
    g_dag[node_index].uint16s[0] = dag_node.uint16s[0];
}
