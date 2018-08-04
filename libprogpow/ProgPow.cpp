#include "ProgPow.h"

#include <sstream>

#define rnd() (kiss99(rnd_state))
#define mix_src() ("mix[" + std::to_string(rnd() % PROGPOW_REGS) + "]")
#define mix_dst() ("mix[" + std::to_string(mix_seq[(mix_seq_cnt++)%PROGPOW_REGS]) + "]")

void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

std::string ProgPow::getKern(uint64_t prog_seed, kernel_t kern)
{
	std::stringstream ret;

    uint32_t seed0 = (uint32_t)prog_seed;
    uint32_t seed1 = prog_seed >> 32;
    uint32_t fnv_hash = 0x811c9dc5;
    kiss99_t rnd_state;
    rnd_state.z = fnv1a(fnv_hash, seed0);
    rnd_state.w = fnv1a(fnv_hash, seed1);
    rnd_state.jsr = fnv1a(fnv_hash, seed0);
    rnd_state.jcong = fnv1a(fnv_hash, seed1);

    // Create a random sequence of mix destinations
    // Merge is a read-modify-write, guaranteeing every mix element is modified every loop
    int mix_seq[PROGPOW_REGS];
    int mix_seq_cnt = 0;
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix_seq[i] = i;
    for (int i = PROGPOW_REGS - 1; i > 0; i--)
    {
        int j = rnd() % (i + 1);
        swap(mix_seq[i], mix_seq[j]);
    }

	if (kern == KERNEL_CUDA)
    {
        ret << "typedef unsigned int       uint32_t;\n";
        ret << "typedef unsigned long long uint64_t;\n";
        ret << "#define ROTL32(x,n) __funnelshift_l((x), (x), (n))\n";
        ret << "#define ROTR32(x,n) __funnelshift_r((x), (x), (n))\n";
        ret << "#define min(a,b) ((a<b) ? a : b)\n";
        ret << "#define mul_hi(a, b) __umulhi(a, b)\n";
        ret << "#define clz(a) __clz(a)\n";
        ret << "#define popcount(a) __popc(a)\n";
        ret << "\n";
    }
    else
	{
		ret << "#ifndef GROUP_SIZE\n";
		ret << "#define GROUP_SIZE 128\n";
		ret << "#endif\n";
		ret << "#define GROUP_SHARE (GROUP_SIZE / " << PROGPOW_LANES << ")\n";
        ret << "\n";
        ret << "typedef unsigned int       uint32_t;\n";
        ret << "typedef unsigned long      uint64_t;\n";
        ret << "#define ROTL32(x, n) rotate((x), (uint32_t)(n))\n";
        ret << "#define ROTR32(x, n) rotate((x), (uint32_t)(32-n))\n";
        ret << "\n";
	}

    ret << "#define PROGPOW_LANES			" << PROGPOW_LANES << "\n";
	ret << "#define PROGPOW_REGS			" << PROGPOW_REGS << "\n";
    ret << "#define PROGPOW_CNT_MEM			" << PROGPOW_CNT_MEM << "\n";
    ret << "#define PROGPOW_CNT_MATH		" << PROGPOW_CNT_MATH << "\n";
    ret << "#define PROGPOW_CACHE_WORDS  " << PROGPOW_CACHE_BYTES / sizeof(uint32_t) << "\n";
    ret << "\n";

	if (kern == KERNEL_CUDA)
	{
		ret << "__device__ __forceinline__ void progPowLoop(const uint32_t loop,\n";
		ret << "        uint32_t mix[PROGPOW_REGS],\n";
		ret << "        const uint64_t *g_dag,\n";
		ret << "        const uint32_t c_dag[PROGPOW_CACHE_WORDS])\n";
	}
	else
	{
		ret << "void progPowLoop(const uint32_t loop,\n";
		ret << "        uint32_t mix[PROGPOW_REGS],\n";
		ret << "        __global const uint64_t *g_dag,\n";
		ret << "        __local const uint32_t c_dag[PROGPOW_CACHE_WORDS],\n";
		ret << "        __local uint64_t share[GROUP_SHARE])\n";
	}
	ret << "{\n";

	ret << "uint32_t offset;\n";
	ret << "uint64_t data64;\n";
	ret << "uint32_t data32;\n";

	if (kern == KERNEL_CUDA)
		ret << "const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES-1);\n";
	else
	{
		ret << "const uint32_t lane_id = get_local_id(0) & (PROGPOW_LANES-1);\n";
		ret << "const uint32_t group_id = get_local_id(0) / PROGPOW_LANES;\n";
	}

	// Global memory access
	// lanes access sequential locations
	// Hard code mix[0] to guarantee the address for the global load depends on the result of the load
	ret << "// global load\n";
	if (kern == KERNEL_CUDA)
		ret << "offset = __shfl_sync(0xFFFFFFFF, mix[0], loop%PROGPOW_LANES, PROGPOW_LANES);\n";
	else
	{
		ret << "if(lane_id == (loop % PROGPOW_LANES))\n";
		ret << "    share[group_id] = mix[0];\n";
		ret << "barrier(CLK_LOCAL_MEM_FENCE);\n";
		ret << "offset = share[group_id];\n";
	}
	ret << "offset %= PROGPOW_DAG_WORDS;\n";
	ret << "offset = offset * PROGPOW_LANES + lane_id;\n";
	ret << "data64 = g_dag[offset];\n";

	for (int i = 0; (i < PROGPOW_CNT_CACHE) || (i < PROGPOW_CNT_MATH); i++)
	{
		if (i < PROGPOW_CNT_CACHE)
		{
			// Cached memory access
			// lanes access random locations
			std::string src = mix_src();
			std::string dest = mix_dst();
			uint32_t r = rnd();
			ret << "// cache load\n";
			ret << "offset = " << src << " % PROGPOW_CACHE_WORDS;\n";
			ret << "data32 = c_dag[offset];\n";
			ret << merge(dest, "data32", r);
		}
		if (i < PROGPOW_CNT_MATH)
		{
			// Random Math
			// A tree combining random input registers together
			// reduced to a single result
			std::string src1 = mix_src();
			std::string src2 = mix_src();
			uint32_t r1 = rnd();
			uint32_t r2 = rnd();
			std::string dest = mix_dst();
			ret << "// random math\n";
			ret << math("data32", src1, src2, r1);
			ret << merge(dest, "data32", r2);
		}
	}
	// Consume the global load data at the very end of the loop, to allow fully latency hiding
	ret << "// consume global load data\n";
	ret << merge("mix[0]", "data64", rnd());
	ret << merge(mix_dst(), "(data64>>32)", rnd());
	ret << "}\n";
	ret << "\n";

	return ret.str();
}

// Merge new data from b into the value in a
// Assuming A has high entropy only do ops that retain entropy, even if B is low entropy
// (IE don't do A&B)
std::string ProgPow::merge(std::string a, std::string b, uint32_t r)
{
	switch (r % 4)
	{
	case 0: return a + " = (" + a + " * 33) + " + b + ";\n";
	case 1: return a + " = (" + a + " ^ " + b + ") * 33;\n";
	case 2: return a + " = ROTL32(" + a + ", " + std::to_string((r >> 16) % 32) + ") ^ " + b + ";\n";
	case 3: return a + " = ROTR32(" + a + ", " + std::to_string((r >> 16) % 32) + ") ^ " + b + ";\n";
	}
    return "#error\n";
}

// Random math between two input values
std::string ProgPow::math(std::string d, std::string a, std::string b, uint32_t r)
{
	switch (r % 11)
	{
	case 0: return d + " = " + a + " + " + b + ";\n";
	case 1: return d + " = " + a + " * " + b + ";\n";
	case 2: return d + " = mul_hi(" + a + ", " + b + ");\n";
	case 3: return d + " = min(" + a + ", " + b + ");\n";
	case 4: return d + " = ROTL32(" + a + ", " + b + ");\n";
	case 5: return d + " = ROTR32(" + a + ", " + b + ");\n";
	case 6: return d + " = " + a + " & " + b + ";\n";
	case 7: return d + " = " + a + " | " + b + ";\n";
	case 8: return d + " = " + a + " ^ " + b + ";\n";
	case 9: return d + " = clz(" + a + ") + clz(" + b + ");\n";
	case 10: return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
	}
    return "#error\n";
}

uint32_t ProgPow::fnv1a(uint32_t &h, uint32_t d)
{
	return h = (h ^ d) * 0x1000193;
}

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
uint32_t ProgPow::kiss99(kiss99_t &st)
{
	uint32_t znew = (st.z = 36969 * (st.z & 65535) + (st.z >> 16));
	uint32_t wnew = (st.w = 18000 * (st.w & 65535) + (st.w >> 16));
	uint32_t MWC = ((znew << 16) + wnew);
	uint32_t SHR3 = (st.jsr ^= (st.jsr << 17), st.jsr ^= (st.jsr >> 13), st.jsr ^= (st.jsr << 5));
	uint32_t CONG = (st.jcong = 69069 * st.jcong + 1234567);
	return ((MWC^CONG) + SHR3);
}
