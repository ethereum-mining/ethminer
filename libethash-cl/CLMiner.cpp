/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#include <boost/dll.hpp>

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "CLMiner.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{
struct CLChannel : public LogChannel
{
    static const char* name() { return EthOrange "cl"; }
    static const int verbosity = 2;
    static const bool debug = false;
};
#define cllog clog(CLChannel)
#define ETHCL_LOG(_contents) cllog << _contents

/**
 * Returns the name of a numerical cl_int error
 * Takes constants from CL/cl.h and returns them in a readable format
 */
static const char* strClError(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";

#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
        return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
        return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
        return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif  // CL_VERSION_1_2

    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY";

#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
        return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
        return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif  // CL_VERSION_1_2

#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
        return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
        return "CL_INVALID_DEVICE_QUEUE";
#endif  // CL_VERSION_2_0

#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID:
        return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
        return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif  // CL_VERSION_2_2
    }

    return "Unknown CL error encountered";
}

/**
 * Prints cl::Errors in a uniform way
 * @param msg text prepending the error message
 * @param clerr cl:Error object
 *
 * Prints errors in the format:
 *      msg: what(), string err() (numeric err())
 */
static std::string ethCLErrorHelper(const char* msg, cl::Error const& clerr)
{
    std::ostringstream osstream;
    osstream << msg << ": " << clerr.what() << ": " << strClError(clerr.err()) << " ("
             << clerr.err() << ")";
    return osstream.str();
}

namespace
{
void addDefinition(string& _source, char const* _id, unsigned _value)
{
    char buf[256];
    sprintf(buf, "#define %s %uu\n", _id, _value);
    _source.insert(_source.begin(), buf, buf + strlen(buf));
}

std::vector<cl::Platform> getPlatforms()
{
    vector<cl::Platform> platforms;
    try
    {
        cl::Platform::get(&platforms);
    }
    catch (cl::Error const& err)
    {
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
        if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
            std::cerr << "No OpenCL platforms found" << std::endl;
        else
#endif
            std::cerr << "OpenCL error : " << err.what();
    }
    return platforms;
}

std::vector<cl::Device> getDevices(
    std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
    vector<cl::Device> devices;
    size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
    try
    {
        _platforms[platform_num].getDevices(
            CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, &devices);
    }
    catch (cl::Error const& err)
    {
        // if simply no devices found return empty vector
        if (err.err() != CL_DEVICE_NOT_FOUND)
            throw err;
    }
    return devices;
}

}  // namespace

}  // namespace eth
}  // namespace dev

CLMiner::CLMiner(unsigned _index, CLSettings _settings, DeviceDescriptor& _device)
  : Miner("cl-", _index), m_settings(_settings)
{
    m_deviceDescriptor = _device;
}

void CLMiner::workLoop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cllog, "cl-" << m_index << " CLMiner::workLoop() begin");

    if (!initDevice())
        return;

    try
    {
        minerLoop();
        m_queue.finish();
    }
    catch (cl::Error const& _e)
    {
        string _what = ethCLErrorHelper("OpenCL Error", _e);
        throw std::runtime_error(_what);
    }

    DEV_BUILD_LOG_PROGRAMFLOW(cllog, "cl-" << m_index << " CLMiner::workLoop() end");
}

void CLMiner::compileProgPoWKernel(int _block, int _dagelms)
{
    auto startCompile = std::chrono::steady_clock::now();
    cllog << "Compiling ProgPoW kernel for period : " << (_block / PROGPOW_PERIOD);

    std::string progpow_code = ProgPow::getKern(_block, ProgPow::KERNEL_CL);
    progpow_code += std::string(cl_progpow_miner_kernel, sizeof(cl_progpow_miner_kernel));

    char options[256];
    int computeCapability = 0;
    int maxRegs = 0;

    addDefinition(progpow_code, "GROUP_SIZE", m_settings.localWorkSize);
    addDefinition(progpow_code, "PROGPOW_DAG_BYTES", (unsigned int)m_epochContext.dagSize);
    addDefinition(progpow_code, "PROGPOW_DAG_ELEMENTS", _dagelms);
    addDefinition(progpow_code, "LIGHT_WORDS", m_epochContext.lightNumItems);
    addDefinition(progpow_code, "MAX_SEARCH_RESULTS", MAX_SEARCH_RESULTS);
    switch (m_deviceDescriptor.clPlatformType)
    {
    case ClPlatformTypeEnum::Nvidia:
        addDefinition(progpow_code, "PLATFORM", 1);
        computeCapability =
            (m_deviceDescriptor.clNvComputeMajor * 10) + m_deviceDescriptor.clNvComputeMinor;
        maxRegs = computeCapability > 35 ? 72 : 63;
        sprintf(options, "-cl-nv-maxrregcount=%d", maxRegs);
        break;
    case ClPlatformTypeEnum::Amd:
        addDefinition(progpow_code, "PLATFORM", 2);
        sprintf(options, "%s", "");
        break;
    case ClPlatformTypeEnum::Clover:
        addDefinition(progpow_code, "PLATFORM", 3);
        sprintf(options, "%s", "");
        break;
    default:
        addDefinition(progpow_code, "PLATFORM", 0);
        sprintf(options, "%s", "");
        break;
    }

    cl::Program::Sources progpow_sources{{progpow_code.data(), progpow_code.size()}};
    cl::Program progpow_program(m_context, progpow_sources);
    try
    {
        progpow_program.build({m_device}, options);
        m_progpow_search_kernel = cl::Kernel(progpow_program, "progpow_search");
        m_progpow_search_kernel.setArg(0, m_searchBuffer);
        m_progpow_search_kernel.setArg(2, m_dag);
        m_progpow_search_kernel.setArg(5, 0);

        // Lower current target so the arg is properly passed to the
        // new kernel
        m_current_target = 0;

    }
    catch (cl::BuildError const& buildErr)
    {
#if _DEVELOPER
        if (g_logOptions & LOG_COMPILE)
        {
            std::string buildlog = progpow_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            cllog << "OpenCL ProgPoW kernel build error : " << buildErr.err() << " "
                  << buildErr.what() << "\r\n"
                  << buildlog;
        }
#else
        (void)buildErr;
#endif
        throw std::runtime_error("Unable to build ProgPoW Kernel");
    }

    cllog << "Done compiling in "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - startCompile)
                 .count()
          << " ms. ";
}

void CLMiner::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
    // Load available platforms
    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return;

    unsigned int dIdx = 0;
    for (unsigned int pIdx = 0; pIdx < platforms.size(); pIdx++)
    {
        std::string platformName = platforms.at(pIdx).getInfo<CL_PLATFORM_NAME>();
        ClPlatformTypeEnum platformType = ClPlatformTypeEnum::Unknown;
        if (platformName == "AMD Accelerated Parallel Processing")
            platformType = ClPlatformTypeEnum::Amd;
        else if (platformName == "Clover")
            platformType = ClPlatformTypeEnum::Clover;
        else if (platformName == "NVIDIA CUDA")
            platformType = ClPlatformTypeEnum::Nvidia;
        else
        {
            std::cerr << "Unrecognized platform " << platformName << std::endl;
            continue;
        }


        std::string platformVersion = platforms.at(pIdx).getInfo<CL_PLATFORM_VERSION>();
        unsigned int platformVersionMajor = std::stoi(platformVersion.substr(7, 1));
        unsigned int platformVersionMinor = std::stoi(platformVersion.substr(9, 1));

        dIdx = 0;
        vector<cl::Device> devices = getDevices(platforms, pIdx);
        for (auto const& device : devices)
        {
            DeviceTypeEnum clDeviceType = DeviceTypeEnum::Unknown;
            cl_device_type detectedType = device.getInfo<CL_DEVICE_TYPE>();
            if (detectedType == CL_DEVICE_TYPE_GPU)
                clDeviceType = DeviceTypeEnum::Gpu;
            else if (detectedType == CL_DEVICE_TYPE_CPU)
                clDeviceType = DeviceTypeEnum::Cpu;
            else if (detectedType == CL_DEVICE_TYPE_ACCELERATOR)
                clDeviceType = DeviceTypeEnum::Accelerator;

            string uniqueId;
            DeviceDescriptor deviceDescriptor;

            if (clDeviceType == DeviceTypeEnum::Gpu && platformType == ClPlatformTypeEnum::Nvidia)
            {
                cl_int bus_id, slot_id;
                if (clGetDeviceInfo(device.get(), 0x4008, sizeof(bus_id), &bus_id, NULL) ==
                        CL_SUCCESS &&
                    clGetDeviceInfo(device.get(), 0x4009, sizeof(slot_id), &slot_id, NULL) ==
                        CL_SUCCESS)
                {
                    std::ostringstream s;
                    s << setfill('0') << setw(2) << hex << bus_id << ":" << setw(2)
                      << (unsigned int)(slot_id >> 3) << "." << (unsigned int)(slot_id & 0x7);
                    uniqueId = s.str();
                }
            }
            else if (clDeviceType == DeviceTypeEnum::Gpu &&
                     (platformType == ClPlatformTypeEnum::Amd ||
                         platformType == ClPlatformTypeEnum::Clover))
            {
                cl_char t[24];
                if (clGetDeviceInfo(device.get(), 0x4037, sizeof(t), &t, NULL) == CL_SUCCESS)
                {
                    std::ostringstream s;
                    s << setfill('0') << setw(2) << hex << (unsigned int)(t[21]) << ":" << setw(2)
                      << (unsigned int)(t[22]) << "." << (unsigned int)(t[23]);
                    uniqueId = s.str();
                }
            }
            else if (clDeviceType == DeviceTypeEnum::Cpu)
            {
                std::ostringstream s;
                s << "CPU:" << setfill('0') << setw(2) << hex << (pIdx + dIdx);
                uniqueId = s.str();
            }
            else
            {
                // We're not prepared (yet) to handle other platforms or types
                ++dIdx;
                continue;
            }

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            // Fill the blanks by OpenCL means
            deviceDescriptor.name = device.getInfo<CL_DEVICE_NAME>();
            deviceDescriptor.type = clDeviceType;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.clDetected = true;
            deviceDescriptor.clPlatformId = pIdx;
            deviceDescriptor.clPlatformName = platformName;
            deviceDescriptor.clPlatformType = platformType;
            deviceDescriptor.clPlatformVersion = platformVersion;
            deviceDescriptor.clPlatformVersionMajor = platformVersionMajor;
            deviceDescriptor.clPlatformVersionMinor = platformVersionMinor;
            deviceDescriptor.clDeviceOrdinal = dIdx;

            deviceDescriptor.clName = deviceDescriptor.name;
            deviceDescriptor.clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
            deviceDescriptor.clDeviceVersionMajor =
                std::stoi(deviceDescriptor.clDeviceVersion.substr(7, 1));
            deviceDescriptor.clDeviceVersionMinor =
                std::stoi(deviceDescriptor.clDeviceVersion.substr(9, 1));
            deviceDescriptor.totalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            deviceDescriptor.clMaxMemAlloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            deviceDescriptor.clMaxWorkGroup = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            deviceDescriptor.clMaxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

            // Apparently some 36 CU devices return a bogus 14!!!
            deviceDescriptor.clMaxComputeUnits =
                deviceDescriptor.clMaxComputeUnits == 14 ? 36 : deviceDescriptor.clMaxComputeUnits;

            // Is it an NVIDIA card ?
            if (platformType == ClPlatformTypeEnum::Nvidia)
            {
                size_t siz;
                clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
                    sizeof(deviceDescriptor.clNvComputeMajor), &deviceDescriptor.clNvComputeMajor,
                    &siz);
                clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,
                    sizeof(deviceDescriptor.clNvComputeMinor), &deviceDescriptor.clNvComputeMinor,
                    &siz);
                deviceDescriptor.clNvCompute = to_string(deviceDescriptor.clNvComputeMajor) + "." +
                                               to_string(deviceDescriptor.clNvComputeMinor);
            }

            // Upsert Devices Collection
            _DevicesCollection[uniqueId] = deviceDescriptor;
            ++dIdx;
        }
    }
}

void CLMiner::kick_miner() 
{
    Miner::kick_miner();
    if (m_initialized)
        m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, offsetof(search_results, abort), sizeof(m_one), &m_one);

}

bool CLMiner::initDevice()
{
    // LookUp device
    // Load available platforms
    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return false;

    vector<cl::Device> devices = getDevices(platforms, m_deviceDescriptor.clPlatformId);
    if (devices.empty())
        return false;

    // Set global work size
    m_settings.globalWorkSize = m_settings.localWorkSize * m_settings.globalWorkSizeMultiplier;

    // Set device and context
    m_device = devices.at(m_deviceDescriptor.clDeviceOrdinal);
    m_context = cl::Context(vector<cl::Device>(&m_device, &m_device + 1));
    m_queue = cl::CommandQueue(m_context, m_device);


    // Set Hardware Monitor Info
    if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Nvidia)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
        m_settings.noBinary = true;
    }
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Amd)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::AMD;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
    }
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::UNKNOWN;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
        m_settings.noBinary = true;
    }
    else
    {
        // Don't know what to do with this
        cllog << "Unrecognized Platform";
        return false;
    }

    if (m_deviceDescriptor.clPlatformVersionMajor == 1 &&
        (m_deviceDescriptor.clPlatformVersionMinor == 0 ||
            m_deviceDescriptor.clPlatformVersionMinor == 1))
    {
        if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
        {
            cllog
                << "OpenCL " << m_deviceDescriptor.clPlatformVersion
                << " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!";
        }
        else
        {
            cllog << "OpenCL " << m_deviceDescriptor.clPlatformVersion
                  << " not supported. Minimum required version is 1.2";
            throw new std::runtime_error("OpenCL 1.2 required");
        }
    }

    ostringstream s;
    s << "Using PciId : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.clName;
    if (!m_deviceDescriptor.clNvCompute.empty())
        s << " (Compute " + m_deviceDescriptor.clNvCompute + ")";
    else
        s << " " << m_deviceDescriptor.clDeviceVersion;

    s << " Memory : " << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory)
      << " Group size : " << m_settings.globalWorkSizeMultiplier
      << " Work size : " << m_settings.localWorkSize;

    cllog << s.str();

    // Get Programs
    char options[256] = {0};
    int computeCapability = 0;
#ifndef __clang__

    // Nvidia
    if (!m_deviceDescriptor.clNvCompute.empty())
    {
        computeCapability =
            m_deviceDescriptor.clNvComputeMajor * 10 + m_deviceDescriptor.clNvComputeMinor;
        int maxregs = computeCapability >= 35 ? 72 : 63;
        sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
    }

#endif

    string ethash_code =
        string(cl_ethash_miner_kernel, cl_ethash_miner_kernel + sizeof(cl_ethash_miner_kernel));
    addDefinition(ethash_code, "WORKSIZE", m_settings.localWorkSize);
    addDefinition(ethash_code, "ACCESSES", ETHASH_ACCESSES);
    addDefinition(ethash_code, "MAX_SEARCH_RESULTS", MAX_SEARCH_RESULTS);
    addDefinition(ethash_code, "PLATFORM", m_deviceDescriptor.clPlatformId);
    addDefinition(ethash_code, "COMPUTE", computeCapability);

    if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
        addDefinition(ethash_code, "LEGACY", 1);

    cl::Program::Sources ethash_sources{{ethash_code.data(), ethash_code.size()}};
    cl::Program ethash_program(m_context, ethash_sources);
    try
    {
        ethash_program.build({m_device}, options);
        m_ethash_search_kernel = cl::Kernel(ethash_program, "ethash_search");
        m_ethash_dag_kernel = cl::Kernel(ethash_program, "ethash_calculate_dag_item");

        // create mem buffers
        m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, 32);
        m_target = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(m_current_target));
        m_searchBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, sizeof(search_results));
        m_ethash_search_kernel.setArg(0, m_searchBuffer);
    }
    catch (cl::BuildError const& buildErr)
    {
#if _DEVELOPER
        if (g_logOptions & LOG_COMPILE)
        {
            std::string buildlog = ethash_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            cllog << "OpenCL Ethash kernel build error : " << buildErr.err() << " "
                  << buildErr.what() << "\r\n"
                  << buildlog;
        }
#else
        (void)buildErr;
#endif
        throw std::runtime_error("Unable to build Ethash Kernel");
    }

    m_initialized = true;
    return m_initialized;
}

bool CLMiner::initEpoch_internal()
{
    auto startInit = std::chrono::steady_clock::now();
    size_t RequiredMemory = (m_epochContext.dagSize + m_epochContext.lightSize);

    // Release the pause flag if any
    resume(MinerPauseEnum::PauseDueToInsufficientMemory);
    resume(MinerPauseEnum::PauseDueToInitEpochError);

    // Check whether the current device has sufficient memory every time we recreate the dag
    if (m_deviceDescriptor.totalMemory < RequiredMemory)
    {
        cllog << "Epoch " << m_epochContext.epochNumber << " requires "
              << dev::getFormattedMemory((double)RequiredMemory) << " memory. Only "
              << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory)
              << " available on device.";
        pause(MinerPauseEnum::PauseDueToInsufficientMemory);
        return true;  // This will prevent to exit the thread and
                      // Eventually resume mining when changing coin or epoch (NiceHash)
    }

    cllog << "Generating DAG + Light : " << dev::getFormattedMemory((double)RequiredMemory);

    try
    {
        // create buffer for dag
        try
        {
            m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_epochContext.lightSize);
            m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_epochContext.dagSize);
            m_queue.enqueueWriteBuffer(
                m_light, CL_TRUE, 0, m_epochContext.lightSize, m_epochContext.lightCache);
        }
        catch (cl::Error const& err)
        {
            cwarn << ethCLErrorHelper("Creating DAG buffer failed", err);
            pause(MinerPauseEnum::PauseDueToInitEpochError);
            return true;
        }


        m_ethash_dag_kernel.setArg(1, m_light);
        m_ethash_dag_kernel.setArg(2, m_dag);
        m_ethash_dag_kernel.setArg(3, (uint32_t)(m_epochContext.lightSize / 64));
        m_ethash_dag_kernel.setArg(4, ~0u);

        const uint32_t workItems =
            m_epochContext.dagNumItems * 2;  // GPU computes partial 512-bit DAG items.
        uint32_t start;
        const uint32_t chunk = 10240 * m_settings.localWorkSize;
        for (start = 0; start <= workItems - chunk; start += chunk)
        {
            m_ethash_dag_kernel.setArg(0, start);
            m_queue.enqueueNDRangeKernel(
                m_ethash_dag_kernel, cl::NullRange, chunk, m_settings.localWorkSize);
            m_queue.finish();
        }
        if (start < workItems)
        {
            uint32_t groupsLeft = workItems - start;
            groupsLeft = (groupsLeft + m_settings.localWorkSize - 1) / m_settings.localWorkSize;
            m_ethash_dag_kernel.setArg(0, start);
            m_queue.enqueueNDRangeKernel(m_ethash_dag_kernel, cl::NullRange,
                groupsLeft * m_settings.localWorkSize, m_settings.localWorkSize);
            m_queue.finish();
        }

        // Set args for future runs of the search kernel
        m_ethash_search_kernel.setArg(2, m_dag);
        m_ethash_search_kernel.setArg(3, m_epochContext.dagNumItems);
        m_ethash_search_kernel.setArg(6, 0xffffffff);

        cllog << "Generated DAG + Light in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - startInit)
                     .count()
              << " ms. "
              << dev::getFormattedMemory((double)(m_deviceDescriptor.totalMemory - RequiredMemory))
              << " left.";
    }
    catch (cl::Error const& err)
    {
        cllog << ethCLErrorHelper("OpenCL init epoch failed", err);
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        return false;
    }
    return true;
}

void CLMiner::ethash_search()
{
    using clock = std::chrono::steady_clock;
    clock::time_point start;

    m_queue.enqueueWriteBuffer(
        m_header, CL_FALSE, 0, m_work_active.header.size, m_work_active.header.data());
    m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(m_zerox3), &m_zerox3);

    uint64_t startNonce, baseNonce, target;
    startNonce = baseNonce = m_work_active.startNonce;
    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);

    // Target may be passed as pinned memory pointer
    // instead of parameter on each kernel launch
    if (target != m_current_target)
    {
        m_current_target = target;
        m_queue.enqueueWriteBuffer(
            m_target, CL_FALSE, 0, sizeof(m_current_target), &m_current_target);
        m_ethash_search_kernel.setArg(5, m_target);
    }

    volatile search_results results;

    m_ethash_search_kernel.setArg(1, m_header);
    m_ethash_search_kernel.setArg(4, startNonce);
    // m_ethash_search_kernel.setArg(5, target);

    // process batches until we get new work.
    bool done = m_new_work.load(memory_order_relaxed);

    if (!done)
    {
        // run the kernel
        start = clock::now();
        m_queue.enqueueNDRangeKernel(m_ethash_search_kernel, cl::NullRange,
            m_settings.globalWorkSize, m_settings.localWorkSize);
    }

    while (!done)
    {
        // Exit next time around if there's new work awaiting
        if (!done)
            done = m_new_work.load(memory_order_relaxed);

        if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Nvidia)
        {
            // Most OpenCL implementations of clEnqueueReadBuffer in blocking mode are
            // good, except Nvidia implementing it as a wasteful busywait, so let's
            // work around it by trying to sleep just a bit less than the expected
            // amount of time.
            // https://github.com/mbevand/silentarmy/commit/a6c3517fc189a934edfa89549664f95b51b965d8

            m_queue.enqueueReadBuffer(
                m_searchBuffer, CL_FALSE, 0, sizeof(search_results), (void*)&results);

            // We actually measure the amount of time needed for the previous kernel to
            // complete up to moment it gave results back. On first loop after initialization
            // of the miner we take a sample and then we continue to keep an EMA of the time

            if (m_ethash_search_kernel_time)
            {
                long duration = long(m_ethash_search_kernel_time * SLEEP_RATIO);
                this_thread::sleep_for(std::chrono::microseconds(duration));
            }
            m_queue.finish();
            long elapsed =
                (long)std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start)
                    .count();
#if _DEVELOPER
            if (g_logOptions & LOG_KERNEL_TIMES)
            {
                cllog << "Kernel search time : " << elapsed << " us.";
            }
#endif
            m_ethash_search_kernel_time = long(m_ethash_search_kernel_time * KERNEL_EMA_ALPHA +
                                               (1.0 - KERNEL_EMA_ALPHA) * elapsed);
        }
        else
        {
            m_queue.enqueueReadBuffer(
                m_searchBuffer, CL_TRUE, 0, sizeof(search_results), (void*)&results);
#if _DEVELOPER
            if (g_logOptions & LOG_KERNEL_TIMES)
            {
                long elapsed = (long)std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now() - start)
                                   .count();
                cllog << "Kernel search time : " << elapsed << " us.";
            }
#endif
        }

        if (!done)
        {
            // Zero the results count (in device)
            m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, offsetof(search_results, count),
                sizeof(m_zerox3), (void*)&m_zerox3);

            startNonce += (m_settings.globalWorkSize);
            m_ethash_search_kernel.setArg(4, startNonce);

            // run the kernel
            start = clock::now();
            m_queue.enqueueNDRangeKernel(m_ethash_search_kernel, cl::NullRange,
                m_settings.globalWorkSize, m_settings.localWorkSize);
        }

        // Submit solutions while kernel is running (if any)
        if (results.count)
        {
            uint32_t found_count = std::min((unsigned)results.count, MAX_SEARCH_RESULTS);
            for (uint32_t i = 0; i < found_count; i++)
            {
                h256 mix;
                uint64_t nonce = baseNonce + results.result[i].gid;
                memcpy(mix.data(), (void*)results.result[i].mix, sizeof(results.result[i].mix));
                auto sol =
                    Solution{nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};
                cllog << EthWhite << "Job: " << m_work_active.header.abridged()
                      << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;
                Farm::f().submitProof(sol);
            }
        }
        baseNonce = startNonce;

        // Update the hash rate
        updateHashRate(m_settings.localWorkSize, results.rounds);
    }

    m_queue.finish();
}

void CLMiner::progpow_search()
{
    using clock = std::chrono::steady_clock;
    clock::time_point start;

    m_queue.enqueueWriteBuffer(
        m_header, CL_FALSE, 0, m_work_active.header.size, m_work_active.header.data());
    m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(m_zerox3), &m_zerox3);

    uint64_t startNonce, baseNonce, target;
    startNonce = baseNonce = m_work_active.startNonce;
    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);

    // Target may be passed as pinned memory pointer
    // instead of parameter on each kernel launch
    if (target != m_current_target)
    {
        m_current_target = target;
        m_queue.enqueueWriteBuffer(
            m_target, CL_FALSE, 0, sizeof(m_current_target), &m_current_target);
        m_progpow_search_kernel.setArg(4, m_target);
    }

    volatile search_results results;

    m_progpow_search_kernel.setArg(1, m_header);
    m_progpow_search_kernel.setArg(3, startNonce);
    // m_progpow_search_kernel.setArg(4, target);

    // process batches until we get new work.
    bool done = m_new_work.load(memory_order_relaxed);

    if (!done)
    {
        // run the kernel
        start = clock::now();
        m_queue.enqueueNDRangeKernel(m_progpow_search_kernel, cl::NullRange,
            m_settings.globalWorkSize, m_settings.localWorkSize);
    }

    while (!done)
    {
        // Exit next time around if there's new work awaiting
        if (!done)
            done = m_new_work.load(memory_order_relaxed);

        if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Nvidia)
        {
            // Most OpenCL implementations of clEnqueueReadBuffer in blocking mode are
            // good, except Nvidia implementing it as a wasteful busywait, so let's
            // work around it by trying to sleep just a bit less than the expected
            // amount of time.
            // https://github.com/mbevand/silentarmy/commit/a6c3517fc189a934edfa89549664f95b51b965d8

            m_queue.enqueueReadBuffer(
                m_searchBuffer, CL_FALSE, 0, sizeof(search_results), (void*)&results);

            // We actually measure the amount of time needed for the previous kernel to
            // complete up to moment it gave results back. On first loop after initialization
            // of the miner we take a sample and then we continue to keep an EMA of the time

            if (m_progpow_search_kernel_time)
            {
                long duration = long(m_progpow_search_kernel_time * SLEEP_RATIO);
                this_thread::sleep_for(std::chrono::microseconds(duration));
            }
            m_queue.finish();
            long elapsed =
                (long)std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start)
                    .count();
            m_progpow_search_kernel_time = long(m_progpow_search_kernel_time * KERNEL_EMA_ALPHA +
                                                (1.0 - KERNEL_EMA_ALPHA) * elapsed);
#if _DEVELOPER
            if (g_logOptions & LOG_KERNEL_TIMES)
            {
                cllog << "Kernel search time : " << elapsed << " us.";
            }
#endif
        }
        else
        {
            m_queue.enqueueReadBuffer(
                m_searchBuffer, CL_TRUE, 0, sizeof(search_results), (void*)&results);
#if _DEVELOPER
            if (g_logOptions & LOG_KERNEL_TIMES)
            {
                long elapsed = (long)std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now() - start)
                                   .count();
                cllog << "Kernel search time : " << elapsed << " us.";
            }
#endif
        }

        if (!done)
        {
            // Zero the results count (in device)
            m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, offsetof(search_results, count),
                sizeof(m_zerox3), (void*)&m_zerox3);

            startNonce += (m_settings.globalWorkSize);
            m_progpow_search_kernel.setArg(3, startNonce);

            // run the kernel
            start = clock::now();
            m_queue.enqueueNDRangeKernel(m_progpow_search_kernel, cl::NullRange,
                m_settings.globalWorkSize, m_settings.localWorkSize);
        }

        // Submit solutions while kernel is running (if any)
        if (results.count)
        {
            uint32_t found_count = std::min((unsigned)results.count, MAX_SEARCH_RESULTS);
            for (uint32_t i = 0; i < found_count; i++)
            {
                h256 mix;
                uint64_t nonce = baseNonce + results.result[i].gid;
                memcpy(mix.data(), (void*)results.result[i].mix, sizeof(results.result[i].mix));
                auto sol =
                    Solution{nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};
                cllog << EthWhite << "Job: " << m_work_active.header.abridged()
                      << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;
                Farm::f().submitProof(sol);
            }
        }
        baseNonce = startNonce;

        // Update the hash rate
        updateHashRate(m_settings.globalWorkSize, 1);
    }

    m_queue.finish();
}
