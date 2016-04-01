#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA  1
#define OPENCL_PLATFORM_AMD		2


#define THREADS_PER_HASH (128 / 16)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

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
static uint2 ROL2(const uint2 a, const int offset) {
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

static void keccak_f1600_round(uint2* a, uint r)
{
	uint2 t[25];
	uint2 u, v;

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
	u = a[1];
	t[0] = a[0];
	t[1] = ROL2(a[6], 44);
	t[6] = ROL2(a[9], 20);
	t[9] = ROL2(a[22], 61);
	t[22] = ROL2(a[14], 39);
	t[14] = ROL2(a[20], 18);
	t[20] = ROL2(a[2], 62);
	t[2] = ROL2(a[12], 43);
	t[12] = ROL2(a[13], 25);
	t[13] = ROL2(a[19], 8);
	t[19] = ROL2(a[23], 56);
	t[23] = ROL2(a[15], 41);
	t[15] = ROL2(a[4], 27);
	t[4] = ROL2(a[24], 14);
	t[24] = ROL2(a[21], 2);
	t[21] = ROL2(a[8], 55);
	t[8] = ROL2(a[16], 45);
	t[16] = ROL2(a[5], 36);
	t[5] = ROL2(a[3], 28);
	t[3] = ROL2(a[18], 21);
	t[18] = ROL2(a[17], 15);
	t[17] = ROL2(a[11], 10);
	t[11] = ROL2(a[7], 6);
	t[7] = ROL2(a[10], 3);
	t[10] = ROL2(u, 1);

	// Chi
	a[0] = bitselect(t[0] ^ t[2], t[0], t[1]);
	a[1] = bitselect(t[1] ^ t[3], t[1], t[2]);
	a[2] = bitselect(t[2] ^ t[4], t[2], t[3]);
	a[3] = bitselect(t[3] ^ t[0], t[3], t[4]);
	a[4] = bitselect(t[4] ^ t[1], t[4], t[0]);

	// Iota
	a[0] ^= Keccak_f1600_RC[r];

	a[5] = bitselect(t[5] ^ t[7], t[5], t[6]);
	a[6] = bitselect(t[6] ^ t[8], t[6], t[7]);
	a[7] = bitselect(t[7] ^ t[9], t[7], t[8]);
	a[8] = bitselect(t[8] ^ t[5], t[8], t[9]);
	a[9] = bitselect(t[9] ^ t[6], t[9], t[5]);

	a[10] = bitselect(t[10] ^ t[12], t[10], t[11]);
	a[11] = bitselect(t[11] ^ t[13], t[11], t[12]);
	a[12] = bitselect(t[12] ^ t[14], t[12], t[13]);
	a[13] = bitselect(t[13] ^ t[10], t[13], t[14]);
	a[14] = bitselect(t[14] ^ t[11], t[14], t[10]);

	a[15] = bitselect(t[15] ^ t[17], t[15], t[16]);
	a[16] = bitselect(t[16] ^ t[18], t[16], t[17]);
	a[17] = bitselect(t[17] ^ t[19], t[17], t[18]);
	a[18] = bitselect(t[18] ^ t[15], t[18], t[19]);
	a[19] = bitselect(t[19] ^ t[16], t[19], t[15]);

	a[20] = bitselect(t[20] ^ t[22], t[20], t[21]);
	a[21] = bitselect(t[21] ^ t[23], t[21], t[22]);
	a[22] = bitselect(t[22] ^ t[24], t[22], t[23]);
	a[23] = bitselect(t[23] ^ t[20], t[23], t[24]);
	a[24] = bitselect(t[24] ^ t[21], t[24], t[20]);
}

static void keccak_f1600_no_absorb(uint2* a, uint out_size, uint isolate)
{


	// Originally I unrolled the first and last rounds to interface
	// better with surrounding code, however I haven't done this
	// without causing the AMD compiler to blow up the VGPR usage.

	uint r = 0;
	uint o = 25;
	do
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
			if (r == 23) o = out_size;
		}
	} 
	while (r < 24);

	// final round optimised for digest size
	//keccak_f1600_round(a, 23, out_size);
}

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

static uint fnv(uint x, uint y)
{
	return x * FNV_PRIME ^ y;
}

static uint4 fnv4(uint4 x, uint4 y)
{
	return x * FNV_PRIME ^ y;
}

static uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

typedef struct
{
	ulong ulongs[32 / sizeof(ulong)];
} hash32_t;

typedef struct
{
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union {
	uint4 uint4s[4];
	ulong ulongs[8];
	uint  uints[16];
} compute_hash_share;

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
	__local compute_hash_share share[HASHES_PER_LOOP];

	uint const gid = get_global_id(0);

	// Compute one init hash per work item.

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, g_header->ulongs, 4);
	state[4] = start_nonce + gid;

	for (uint i = 6; i != 25; ++i)
	{
		state[i] = 0;
	}
	state[5] = 0x0000000000000001;
	state[8] = 0x8000000000000000;

	keccak_f1600_no_absorb((uint2*)state, 8, isolate);
	
	// Threads work together in this phase in groups of 8.
	uint const thread_id = gid & 7;
	uint const hash_id = (gid % GROUP_SIZE) >> 3;

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			copy(share[hash_id].ulongs, state, 8);

		barrier(CLK_LOCAL_MEM_FENCE);

		uint4 mix = share[hash_id].uint4s[thread_id & 3];
		barrier(CLK_LOCAL_MEM_FENCE);

		__local uint *share0 = share[hash_id].uints;

		// share init0
		if (thread_id == 0)
			*share0 = mix.x;
		barrier(CLK_LOCAL_MEM_FENCE);
		uint init0 = *share0;

		for (uint a = 0; a < ACCESSES; a += 4)
		{
			bool update_share = thread_id == ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint i = 0; i != 4; ++i)
			{
				if (update_share)
				{
					*share0 = fnv(init0 ^ (a + i), ((uint *)&mix)[i]) % DAG_SIZE;
				}
				barrier(CLK_LOCAL_MEM_FENCE);

				mix = fnv4(mix, g_dag[*share0].uint4s[thread_id]);
			}
		}

		share[hash_id].uints[thread_id] = fnv_reduce(mix);
		barrier(CLK_LOCAL_MEM_FENCE);

		if (i == thread_id)
			copy(state + 8, share[hash_id].ulongs, 4);

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (uint i = 13; i != 25; ++i)
	{
		state[i] = 0;
	}
	state[12] = 0x0000000000000001;
	state[16] = 0x8000000000000000;

	// keccak_256(keccak_512(header..nonce) .. mix);
	keccak_f1600_no_absorb((uint2*)state, 1, isolate);

	if (as_ulong(as_uchar8(state[0]).s76543210) < target)
	{
		uint slot = min(MAX_OUTPUTS, atomic_inc(&g_output[0]) + 1);
		g_output[slot] = gid;
	}
}
