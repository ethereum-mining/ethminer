/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#pragma once

#include <fstream>

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

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

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA 1
#define OPENCL_PLATFORM_AMD 2
#define OPENCL_PLATFORM_CLOVER 3


namespace dev
{
namespace eth
{
class CLMiner : public Miner
{
public:
    /* -- default values -- */
    /// Default value of the local work size. Also known as workgroup size.
    static const unsigned c_defaultLocalWorkSize = 192;
    /// Default value of the global work size as a multiplier of the local work size
    static const unsigned c_defaultGlobalWorkSizeMultiplier = 65536;

    CLMiner(unsigned _index);
    ~CLMiner() override;

    static void enumDevices(std::map<string, DeviceDescriptorType>& _DevicesCollection);

    static bool configureGPU(unsigned _localWorkSize, unsigned _globalWorkSizeMultiplier,
        unsigned _dagLoadMode, bool _nobinary);

protected:

    bool initDevice() override;

    bool initEpoch_internal() override;

    void kick_miner() override;

private:

    boost::asio::io_service::strand m_io_strand;

    void workLoop() override;

    vector<cl::Context> m_context;
    vector<cl::CommandQueue> m_queue;
    vector<cl::CommandQueue> m_abortqueue;
    cl::Kernel m_searchKernel;
    cl::Kernel m_dagKernel;
    cl::Device m_device;

    vector<cl::Buffer> m_dag;
    vector<cl::Buffer> m_light;
    vector<cl::Buffer> m_header;
    vector<cl::Buffer> m_searchBuffer;
    unsigned m_globalWorkSize = 0;
    unsigned m_workgroupSize = 0;
    unsigned m_dagItems = 0;
    uint64_t m_lastNonce = 0;

    static bool s_noBinary;
    bool m_noBinary;
    static vector<cl::Device> s_devices;

    /// The local work size for the search
    static unsigned s_workgroupSize;
    /// The initial global work size for the searches
    static unsigned s_initialGlobalWorkSize;
};

}  // namespace eth
}  // namespace dev
