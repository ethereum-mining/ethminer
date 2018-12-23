/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#pragma once

#include <fstream>

#include "ethash_miner_kernel.h"
#include "progpow_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include <libprogpow/ProgPow.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/lexical_cast.hpp>

#define ETHASH_REVISION 23
#define ETHASH_DATASET_BYTES_INIT 1073741824U  // 2**30
#define ETHASH_DATASET_BYTES_GROWTH 8388608U   // 2**23
#define ETHASH_CACHE_BYTES_INIT 1073741824U    // 2**24
#define ETHASH_CACHE_BYTES_GROWTH 131072U      // 2**17
#define ETHASH_MIX_BYTES 256
#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 256
#define ETHASH_CACHE_ROUNDS 3
#define ETHASH_ACCESSES 64

#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wmissing-braces"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif

#define MAX_SEARCH_RESULTS 4U

// Nvidia GPUs do not play well with blocking queue requests
// and boost host cpu to 100%. These two are used to 
// - keep an EMA (Exponential Moving Average) of the search kernel run time
// - keep a ratio to sleep over kernel time.
constexpr double KERNEL_EMA_ALPHA = 0.25;
constexpr double SLEEP_RATIO = 0.95; // The lower the value the higher CPU usage on Nvidia

namespace dev
{
namespace eth
{
// NOTE: The following struct must match the one defined in
// ethash.cl
typedef struct
{
    unsigned int count;
    struct
    {
        // One word for gid and 8 for mix hash
        unsigned int gid;
        unsigned int mix[8];
        unsigned int pad[7];  // pad to size power of 2 (keep this so the struct is same for ethash
    } result[MAX_SEARCH_RESULTS];
} search_results;


class CLMiner : public Miner
{
public:
    CLMiner(unsigned _index, CLSettings _settings, DeviceDescriptor& _device);
    ~CLMiner() override;

    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:

    bool initDevice() override;

    bool initEpoch_internal() override;

private:

    void ethash_search(WorkPackage & _w) override;
    void progpow_search(WorkPackage & _w) override;
    void compileProgPoWKernel(int _block, int _dagelms) override;

    void workLoop() override;

    cl::Kernel m_ethash_search_kernel;
    cl::Kernel m_ethash_dag_kernel;
    cl::Kernel m_progpow_search_kernel;

    cl::Device m_device;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Event m_readEvent;

    long m_ethash_search_kernel_time = 0L;
    long m_progpow_search_kernel_time = 0L;

    cl::Buffer m_dag;
    cl::Buffer m_light;
    cl::Buffer m_header;
    cl::Buffer m_target;
    cl::Buffer m_searchBuffer;

    CLSettings m_settings;

    uint32_t m_zero = 0;
    uint64_t m_current_target = 0ULL;
};

}  // namespace eth
}  // namespace dev
