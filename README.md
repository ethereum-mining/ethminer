### ProgPoW - A Programmatic Proof of Work

ProgPoW is a proof-of-work algorithm designed to close the efficency gap available to specialized ASICs. It utilizes almost all parts of commodity hardware (GPUs), and comes pre-tuned for the most common hardware utilized in the Ethereum network. 

Ever since the first bitcoin mining ASIC was released, many new Proof of Work algorithms have been created with the intention of being “ASIC-resistant”. The goal of “ASIC-resistance” is to resist the centralization of PoW mining power such that these coins couldn’t be so easily manipulated by a few players. 

This document presents an overview of the algorithm and examines what it means to be “ASIC-resistant.” Next, we compare existing PoW designs by analyzing how each algorithm executes in hardware. Finally, we present the detailed implementation by walking through the code.

## ProgPoW Overview
The design goal of ProgPoW is to have the algorithm’s requirements match what is available on commodity GPUs:  If the algorithm were to be implemented on a custom ASIC there should be little opportunity for efficiency gains compared to a commodity GPU.

The main elements of the algorithm are:
* Changes keccak_f1600 (with 64-bit words) to keccak_f800 (with 32-bit words) to reduce impact on total power
* Increases mix state.
* Adds a random sequence of math in the main loop.
* Adds reads from a small, low-latency cache that supports random addresses.
* Increases the DRAM read from 128 bytes to 256 bytes.

While a custom ASIC to implement this algorithm is still possible, the efficiency gains available are minimal.  The majority of a commodity GPU is required to support the above elements. The only optimizations available are:
*Remove the graphics pipeline (displays, geometry engines, texturing, etc)
*Remove floating point math

These would result in minimal, roughly 1.1-1.2x, efficiency gains.  This is much less than the 2x for Ethash or 50x for Cryptonight.

## Rationale for PoW on Commodity Hardware
With the growth of large mining pools, the control of hashing power has been delegated to the top few pools to provide a steadier economic return for small miners. While some have made the argument that large centralized pools defeats the purpose of “ASIC resistance,” it’s important to note that ASIC based coins are even more centralized for several reasons.

1. No natural distribution: There isn’t an economic purpose for ultra-specialized hardware outside of mining and thus no reason for most people to have it. 
2. No reserve group: Thus, there’s no reserve pool of hardware or reserve pool of interested parties to jump in when coin price is volatile and attractive for manipulation. 
3. High barrier to entry: Initial miners are those rich enough to invest capital and ecological resources on the unknown experiment a new coin may be. Thus, initial coin distribution through mining will be very limited causing centralized economic bias. 
4. Delegated centralization vs implementation centralization: While pool centralization is delegated, hardware monoculture is not: only the limiter buyers of this hardware can participate so there isn’t even the possibility of divesting control on short notice.
5. No obvious decentralization of control even with decentralized mining: Once large custom ASIC makers get into the game, designing back-doored hardware is trivial. ASIC makers have no incentive to be transparent or fair in market participation.

While the goal of “ASIC resistance” is valuable, the entire concept of “ASIC resistance” is a bit of a fallacy.  CPUs and GPUs are themselves ASICs.  Any algorithm that can run on a commodity ASIC (a CPU or GPU) by definition can have a customized ASIC created for it with slightly less functionality. Some algorithms are intentionally made to be  “ASIC friendly” - where an ASIC implementation is drastically more efficient than the same algorithm running on general purpose hardware. The protection that this offers when the coin is unknown also makes it an attractive target for a dedicate mining ASIC company as soon as it becomes useful.

Therefore, ASIC resistance is: the efficiency difference of specilized hardware versus hardware that has a wider adoption and applicability.  A smaller efficiency difference between custom vs general hardware mean higher resistance and a better algorithm. This efficiency difference is the proper metric to use when comparing the quality of PoW algorithms.  Efficiency could mean absolute performance, performance per watt, or performance per dollar - they are all highly correlated.  If a single entity creates and controls an ASIC that is drastically more efficient, they can gain 51% of the network hashrate and possibly stage an attack.

## Review of Existing PoW Algorithms

# SHA256
* Potential ASIC efficiency gain ~ 1000X

The SHA algorithm is a sequence of simple math operations - additions, logical ops, and rotates.

To process a single op on a CPU or GPU requires fetching and decoding an instruction, reading data from a register file, executing the instruction, and then writing the result back to a register file.  This takes significant time and power.

A single op implemented in an ASIC takes a handful of transistors and wires.  This means every individual op takes negligible power, area, or time.  A hashing core is built by laying out the sequence of required ops.

The hashing core can execute the required sequence of ops in much less time, and using less power or area, than doing the same sequence on a CPU or GPU.  A bitcoin ASIC consists of a number of identical hashing cores and some minimal off-chip communication.

# Scrypt and NeoScrypt
* Potential ASIC efficiency gain ~ 1000X

Scrypt and NeoScrypt are similar to SHA in the arithmetic and bitwise operations used. Unfortunately, popular coins such as Litecoin only use a scratchpad size between 32kb and 128kb for their PoW mining algorithm. This scratch pad is small enough to trivially fit on an ASIC next to the math core. The implementation of the math core would be very similar to SHA, with similar efficiency gains.

# X11 and X16R
* Potential ASIC efficiency gain ~ 1000X

X11 (and similar X##) require an ASIC that has 11 unique hashing cores pipelined in a fixed sequence.  Each individual hashing core would have similar efficiency to an individual SHA core, so the overall design will have the same efficiency gains.

X16R requires the multiple hashing cores to interact through a simple sequencing state machine. Each individual core will have similar efficiency gains and the sequencing logic will take minimal power, area, or time.

The Baikal BK-X is an existing ASIC with multiple hashing cores and a programmable sequencer.  It has been upgraded to enable new algorithms that sequence the hashes in different orders.

# Equihash
* Potential ASIC efficiency gain ~ 100X

The ~150mb of state is large but possible on an ASIC. The binning, sorting, and comparing of bit strings could be implemented on an ASIC at extremely high speed.

# Cuckoo Cycle
* Potential ASIC efficiency gain ~ 100X

The amount of state required on-chip is not clear as there are Time/Memory Tradeoff attacks. A specialized graph traversal core would have similar efficiency gains to a SHA compute core.

# CryptoNight
* Potential ASIC efficiency gain ~ 50X

Compared to Scrypt, CryptoNight does much less compute and requires a full 2mb of scratch pad (there is no known Time/Memory Tradeoff attack).  The large scratch pad will dominate the ASIC implementation and limit the number of hashing cores, limiting the absolute performance of the ASIC.  An ASIC will consist almost entirely of just on-die SRAM.

# Ethash
* Potential ASIC efficiency gain ~ 2X

Ethash requires external memory due to the large size of the DAG.  However that is all that it requires - there is minimal compute that is done on the result loaded from memory.  As a result a custom ASIC could remove most of the complexity, and power, of a GPU and be just a memory interface connected to a small compute engine.

## ProgPoW Algorithm Walkthrough

The DAG is generated exactly as in ethash.  The only difference is that an additional PROGPOW_SIZE_CACHE worth of data is generated that will live in the L1 cache instead of the framebuffer.

ProgPoW can be tuned using the following parameters.  The proposed settings have been tuned for a range of existing, commodity GPUs:

* `PROGPOW_LANES:` The number of parallel lanes that coordinate to calculate a single hash instance; default is `32.`
* `PROGPOW_REGS:` The register file usage size; default is `16.` 
* `PROGPOW_CACHE_BYTES:` The size of the cache; default is `16 x 1024.`
* `PROGPOW_CNT_MEM:` The number of frame buffer accesses, defined as the outer loop of the algorithm; default is `64` (same as Ethash).
* `PROGPOW_CNT_CACHE:` The number of cache accesses per loop; default is `8.`
* `PROGPOW_CNT_MATH:` The number of math operations per loop; default is `8.`

ProgPoW uses **FNV1a** for merging data. The existing Ethash uses FNV1 for merging, but FNV1a provides better distribution properties.

ProgPow uses [KISS99](https://en.wikipedia.org/wiki/KISS_(algorithm)) for random number generation. This is the simplest (fewest instruction) random generator that passes the TestU01 statistical test suite.  A more complex random number generator like Mersenne Twister can be efficiently implemented on a specialized ASIC, providing an opportunity for efficiency gains.

```cpp

uint32_t fnv1a(uint32_t &h, uint32_t d)
{
    return h = (h ^ d) * 0x1000193;
}

typedef struct {
    uint32_t z, w, jsr, jcong;
} kiss99_t;

// KISS99 is simple, fast, and passes the TestU01 suite
// https://en.wikipedia.org/wiki/KISS_(algorithm)
// http://www.cse.yorku.ca/~oz/marsaglia-rng.html
uint32_t kiss99(kiss99_t &st)
{
    uint32_t znew = (st.z = 36969 * (st.z & 65535) + (st.z >> 16));
    uint32_t wnew = (st.w = 18000 * (st.w & 65535) + (st.w >> 16));
    uint32_t MWC = ((znew << 16) + wnew);
    uint32_t SHR3 = (st.jsr ^= (st.jsr << 17), st.jsr ^= (st.jsr >> 13), st.jsr ^= (st.jsr << 5));
    uint32_t CONG = (st.jcong = 69069 * st.jcong + 1234567);
    return ((MWC^CONG) + SHR3);
}
```

The `LANES*REGS` of mix data is initialized from the hash’s seed.

```cpp
void fill_mix(
    uint64_t hash_seed,
    uint32_t lane_id,
    uint32_t mix[PROGPOW_REGS]
)
{
    // Use FNV to expand the per-warp seed to per-lane
    // Use KISS to expand the per-lane seed to fill mix
    uint32_t fnv_hash = 0x811c9dc5;
    kiss99_t st;
    st.z = fnv1a(fnv_hash, seed);
    st.w = fnv1a(fnv_hash, seed >> 32);
    st.jsr = fnv1a(fnv_hash, lane_id);
    st.jcong = fnv1a(fnv_hash, lane_id);
    for (int i = 0; i < PROGPOW_REGS; i++)
            mix[i] = kiss99(st);
}
```

The main search algorithm uses the Keccak sponge function (a width of 800 bits, with a bitrate of 448, and a capacity of 352) to generate a seed, expands the seed, does a sequence of loads and random math on the mix data, and then compresses the result into a final Keccak permutation (with the same parameters as the first) for target comparison.

```cpp

bool progpow_search(
    const uint64_t prog_seed,
    const uint64_t nonce,
    const hash32_t header,
    const uint64_t target,
    const uint64_t *g_dag, // gigabyte DAG located in framebuffer
    const uint64_t *c_dag  // kilobyte DAG located in l1 cache
)
{
    uint32_t mix[PROGPOW_LANES][PROGPOW_REGS];
    uint32_t result[4];
    for (int i = 0; i < 4; i++)
        result[i] = 0;

    // keccak(header..nonce)
    uint64_t seed = keccak_f800(header, nonce, result);

    // initialize mix for all lanes
    for (int l = 0; l < PROGPOW_LANES; l++)
        fill_mix(seed, l, mix);

    // execute the randomly generated inner loop
    for (int i = 0; i < PROGPOW_CNT_MEM; i++)
        progPowLoop(prog_seed, i, mix, g_dag, c_dag);

    // Reduce mix data to a single per-lane result
    uint32_t lane_hash[PROGPOW_LANES];
    for (int l = 0; l < PROGPOW_LANES; l++)
    {
        lane_hash[l] = 0x811c9dc5
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(lane_hash[l], mix[l][i]);
    }
    // Reduce all lanes to a single 128-bit result
    for (int i = 0; i < 4; i++)
        result[i] = 0x811c9dc5;
    for (int l = 0; l < PROGPOW_LANES; l++)
        fnv1a(result[l%4], lane_hash[l])

    // keccak(header .. keccak(header..nonce) .. result);
    return (keccak_f800(header, seed, result) <= target);
}
```

The inner loop uses FNV and KISS99 to generate a random sequence from the `prog_seed`.  This random sequence determines which mix state is accessed and what random math is performed. Since the `prog_seed` changes relatively infrequently it is expected that `progPowLoop` will be compiled while mining instead of interpreted on the fly.

```cpp

kiss99_t progPowInit(uint64_t prog_seed, int mix_seq[PROGPOW_REGS])
{
    kiss99_t prog_rnd;
    uint32_t fnv_hash = 0x811c9dc5;
    prog_rnd.z = fnv1a(fnv_hash, prog_seed);
    prog_rnd.w = fnv1a(fnv_hash, prog_seed >> 32);
    prog_rnd.jsr = fnv1a(fnv_hash, prog_seed);
    prog_rnd.jcong = fnv1a(fnv_hash, prog_seed >> 32);
    // Create a random sequence of mix destinations for merge()
    // guaranteeing every location is touched once
    // Uses Fisher–Yates shuffle
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix_seq[i] = i;
    for (int i = PROGPOW_REGS - 1; i > 0; i--)
    {
        int j = kiss99(prog_rnd) % (i + 1);
        swap(mix_seq[i], mix_seq[j]);
    }
    return prog_rnd;
}
```

The math operations that merges values into the mix data are ones chosen to maintain entropy.

```cpp
// Merge new data from b into the value in a
// Assuming A has high entropy only do ops that retain entropy
// even if B is low entropy
// (IE don't do A&B)
void merge(uint32_t &a, uint32_t b, uint32_t r)
{
    switch (r % 4)
    {
    case 0: a = (a * 33) + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = ROTL32(a, ((r >> 16) % 32)) ^ b; break;
    case 3: a = ROTR32(a, ((r >> 16) % 32)) ^ b; break;
    }
}
```

The math operations chosen for the random math are ones that are easy to implement in CUDA and OpenCL, the two main programming languages for commodity GPUs.

```cpp
// Random math between two input values
uint32_t math(uint32_t a, uint32_t b, uint32_t r)
{
    switch (r % 11)
    {
    case 0: return a + b;
    case 1: return a * b;
    case 2: return mul_hi(a, b);
    case 3: return min(a, b);
    case 4: return ROTL32(a, b);
    case 5: return ROTR32(a, b);
    case 6: return a & b;
    case 7: return a | b;
    case 8: return a ^ b;
    case 9: return clz(a) + clz(b);
    case 10: return popcount(a) + popcount(b);
    }
}
```

The main loop:

```cpp
// Helper to get the next value in the per-program random sequence
#define rnd()    (kiss99(prog_rnd))
// Helper to pick a random mix location
#define mix_src() (rnd() % PROGPOW_REGS)
// Helper to access the sequence of mix destinations
#define mix_dst() (mix_seq[(mix_seq_cnt++)%PROGPOW_REGS])

void progPowLoop(
    const uint64_t prog_seed,
    const uint32_t loop,
    uint32_t mix[PROGPOW_LANES][PROGPOW_REGS],
    const uint64_t *g_dag,
    const uint32_t *c_dag)
{
    // All lanes share a base address for the global load
    // Global offset uses mix[0] to guarantee it depends on the load result
    uint32_t offset_g = mix[loop%PROGPOW_LANES][0] % DAG_SIZE;
    // Lanes can execute in parallel and will be convergent
    for (int l = 0; l < PROGPOW_LANES; l++)
    {
        // global load to sequential locations
        uint64_t data64 = g_dag[offset_g + l];

        // initialize the seed and mix destination sequence
        int mix_seq[PROGPOW_REGS];
        int mix_seq_cnt = 0;
        kiss99_t prog_rnd = progPowInit(prog_seed, mix_seq);

        uint32_t offset, data32;
        int max_i = max(PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH);
        for (int i = 0; i < max_i; i++)
        {
            if (i < PROGPOW_CNT_CACHE)
            {
                // Cached memory access
                // lanes access random location
                offset = mix[l][mix_src()] % PROGPOW_CACHE_WORDS;
                data32 = c_dag[offset];
                merge(mix[l][mix_dst()], data32, rnd());
            }
            if (i < PROGPOW_CNT_MATH)
            {
                // Random Math
                data32 = math(mix[l][mix_src()], mix[l][mix_src()], rnd());
                merge(mix[l][mix_dst()], data32, rnd());
            }
        }
        // Consume the global load data at the very end of the loop
        // Allows full latency hiding
        merge(mix[l][0], data64, rnd());
        merge(mix[l][mix_dst()], data64>>32, rnd());
    }
}
```



