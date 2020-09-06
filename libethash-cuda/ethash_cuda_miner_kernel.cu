#include "ethash_cuda_miner_kernel.h"

#include "ethash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#include "fnv.cuh"

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }

#include "keccak.cuh"

#include "dagger_shuffled.cuh"

__global__ void ethash_search(volatile Search_results* g_output, uint64_t start_nonce)
{
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 mix[4];
    if (compute_hash(start_nonce + gid, mix))
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
    volatile Search_results* g_output, uint64_t start_nonce)
{
    ethash_search<<<gridSize, blockSize, 0, stream>>>(g_output, start_nonce);
    CUDA_SAFE_CALL(cudaGetLastError());
}

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS (64 / 4)


__global__ void ethash_calculate_dag_item(uint32_t start)
{
    uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (((node_index >> 1) & (~1)) >= d_dag_size)
        return;

    hash128_t dag_node;
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
    copy(dag_nodes[node_index].uint4s, dag_node.uint4s, 4);
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
