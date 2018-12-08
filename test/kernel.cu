// Inner loop for prog_seed 600
__device__ __forceinline__ void progPowLoop(const uint32_t loop,
    uint32_t mix[PROGPOW_REGS],
    const dag_t *g_dag,
    const uint32_t c_dag[PROGPOW_CACHE_WORDS],
    const bool hack_false)
{
    dag_t data_dag;
    uint32_t offset, data;
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);
    // global load
    offset = __shfl_sync(0xFFFFFFFF, mix[0], loop%PROGPOW_LANES, PROGPOW_LANES);
    offset %= PROGPOW_DAG_ELEMENTS;
    offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;
    data_dag = g_dag[offset];
    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();
    // cache load 0
    offset = mix[26] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[0] = (mix[0] ^ data) * 33;
    // random math 0
    data = mix[10] ^ mix[16];
    mix[4] = ROTL32(mix[4], 27) ^ data;
    // cache load 1
    offset = mix[30] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[27] = ROTR32(mix[27], 7) ^ data;
    // random math 1
    data = mix[24] & mix[14];
    mix[26] = (mix[26] * 33) + data;
    // cache load 2
    offset = mix[1] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[13] = (mix[13] * 33) + data;
    // random math 2
    data = mix[17] & mix[16];
    mix[15] = ROTR32(mix[15], 12) ^ data;
    // cache load 3
    offset = mix[19] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[17] = (mix[17] ^ data) * 33;
    // random math 3
    data = mul_hi(mix[31], mix[5]);
    mix[7] = (mix[7] ^ data) * 33;
    // cache load 4
    offset = mix[11] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[14] = (mix[14] ^ data) * 33;
    // random math 4
    data = mix[23] * mix[19];
    mix[8] = (mix[8] * 33) + data;
    // cache load 5
    offset = mix[21] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[9] = (mix[9] ^ data) * 33;
    // random math 5
    data = clz(mix[30]) + clz(mix[15]);
    mix[12] = ROTR32(mix[12], 16) ^ data;
    // cache load 6
    offset = mix[15] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[3] = ROTR32(mix[3], 27) ^ data;
    // random math 6
    data = clz(mix[12]) + clz(mix[5]);
    mix[10] = (mix[10] * 33) + data;
    // cache load 7
    offset = mix[18] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[1] = ROTR32(mix[1], 6) ^ data;
    // random math 7
    data = min(mix[4], mix[25]);
    mix[11] = ROTR32(mix[11], 27) ^ data;
    // cache load 8
    offset = mix[3] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[6] = (mix[6] ^ data) * 33;
    // random math 8
    data = mul_hi(mix[18], mix[16]);
    mix[16] = (mix[16] ^ data) * 33;
    // cache load 9
    offset = mix[17] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[28] = ROTL32(mix[28], 17) ^ data;
    // random math 9
    data = ROTL32(mix[15], mix[23]);
    mix[31] = (mix[31] * 33) + data;
    // cache load 10
    offset = mix[31] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[2] = (mix[2] * 33) + data;
    // random math 10
    data = mix[11] | mix[17];
    mix[19] = ROTL32(mix[19], 28) ^ data;
    // cache load 11
    offset = mix[16] % PROGPOW_CACHE_WORDS;
    data = c_dag[offset];
    mix[30] = ROTR32(mix[30], 18) ^ data;
    // random math 11
    data = mix[22] * mix[7];
    mix[22] = ROTR32(mix[22], 30) ^ data;
    // random math 12
    data = mix[27] & mix[16];
    mix[29] = ROTR32(mix[29], 25) ^ data;
    // random math 13
    data = ROTL32(mix[11], mix[0]);
    mix[5] = (mix[5] ^ data) * 33;
    // random math 14
    data = ROTR32(mix[15], mix[25]);
    mix[24] = ROTL32(mix[24], 13) ^ data;
    // random math 15
    data = mix[14] & mix[26];
    mix[18] = (mix[18] * 33) + data;
    // random math 16
    data = mix[28] * mix[16];
    mix[25] = (mix[25] ^ data) * 33;
    // random math 17
    data = mix[11] * mix[0];
    mix[23] = (mix[23] ^ data) * 33;
    // random math 18
    data = mix[2] + mix[24];
    mix[21] = ROTR32(mix[21], 20) ^ data;
    // random math 19
    data = mix[25] + mix[4];
    mix[20] = ROTL32(mix[20], 22) ^ data;
    // consume global load data
    // hack to prevent compiler from reordering LD and usage
    if (hack_false) __threadfence_block();
    mix[0] = (mix[0] ^ data_dag.s[0]) * 33;
    mix[0] = ROTR32(mix[0], 21) ^ data_dag.s[1];
    mix[4] = (mix[4] * 33) + data_dag.s[2];
    mix[27] = (mix[27] ^ data_dag.s[3]) * 33;
}