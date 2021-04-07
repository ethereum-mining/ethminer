/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#include <boost/dll.hpp>

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "CLMiner.h"
#include "ethash.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

// WARNING: Do not change the value of the following constant
// unless you are prepared to make the neccessary adjustments
// to the assembly code for the binary kernels.
const size_t c_maxSearchResults = 4;

struct CLChannel : public LogChannel
{
    static const char* name() { return EthOrange "cl"; }
    static const int verbosity = 2;
    static const bool debug = false;
};
#define cllog clog(CLChannel)

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
    m_settings.localWorkSize = ((m_settings.localWorkSize + 7) / 8) * 8;
    m_settings.globalWorkSize = m_settings.localWorkSize * m_settings.globalWorkSizeMultiplier;
}

CLMiner::~CLMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cllog, "cl-" << m_index << " CLMiner::~CLMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(cllog, "cl-" << m_index << " CLMiner::~CLMiner() end");
}

// NOTE: The following struct must match the one defined in
// ethash.cl
struct SearchResults
{
    struct
    {
        uint32_t gid;
        // Can't use h256 data type here since h256 contains
        // more than raw data. Kernel returns raw mix hash.
        uint32_t mix[8];
        uint32_t pad[7];  // pad to 16 words for easy indexing
    } rslt[c_maxSearchResults];
    uint32_t count;
    uint32_t hashCount;
    uint32_t abort;
};

void CLMiner::workLoop()
{
    // Memory for zero-ing buffers. Cannot be static or const because crashes on macOS.
    uint32_t zerox3[3] = {0, 0, 0};

    uint64_t startNonce = 0;

    // The work package currently processed by GPU.
    WorkPackage current;
    current.header = h256();

    if (!initDevice())
    return;

    try
    {
        while (!shouldStop())
        {

            // Read results.
            volatile SearchResults results;

            if (m_queue.size())
            {
                // no need to read the abort flag.
                m_queue[0].enqueueReadBuffer(m_searchBuffer[0], CL_TRUE,
                    offsetof(SearchResults, count),
                    (m_settings.noExit ? 1 : 2) * sizeof(results.count), (void*)&results.count);
                if (results.count)
                {
                    if (results.count > c_maxSearchResults) {
                        results.count = c_maxSearchResults;
                    }

                    m_queue[0].enqueueReadBuffer(m_searchBuffer[0], CL_TRUE, 0,
                        results.count * sizeof(results.rslt[0]), (void*)&results);
                    // Reset search buffer if any solution found.
                    if (m_settings.noExit)
                        m_queue[0].enqueueWriteBuffer(m_searchBuffer[0], CL_FALSE,
                            offsetof(SearchResults, count), sizeof(results.count), zerox3);
                }
                // clean the solution count, hash count, and abort flag
                if (!m_settings.noExit)
                    m_queue[0].enqueueWriteBuffer(m_searchBuffer[0], CL_FALSE,
                        offsetof(SearchResults, count), sizeof(zerox3), zerox3);
            }
            else
                results.count = 0;

            // Wait for work or 3 seconds (whichever the first)
            const WorkPackage w = work();
            if (!w)
            {
                boost::system_time const timeout =
                    boost::get_system_time() + boost::posix_time::seconds(3);
                boost::mutex::scoped_lock l(x_work);
                m_new_work_signal.timed_wait(l, timeout);
                continue;
            }

            if (current.header != w.header)
            {

                if (current.epoch != w.epoch)
                {
                    m_abortqueue.clear();

                    if (!initEpoch())
                        break;  // This will simply exit the thread

                    m_abortqueue.push_back(cl::CommandQueue(m_context[0], m_device));
                }

                // Upper 64 bits of the boundary.
                const uint64_t target = (uint64_t)(u64)((u256)w.boundary >> 192);
                assert(target > 0);

                startNonce = w.startNonce;

                // Update header constant buffer.
                m_queue[0].enqueueWriteBuffer(
                    m_header[0], CL_FALSE, 0, w.header.size, w.header.data());
                // zero the result count
                m_queue[0].enqueueWriteBuffer(m_searchBuffer[0], CL_FALSE,
                    offsetof(SearchResults, count),
                    m_settings.noExit ? sizeof(zerox3[0]) : sizeof(zerox3), zerox3);

                m_searchKernel.setArg(0, m_searchBuffer[0]);  // Supply output buffer to kernel.
                m_searchKernel.setArg(1, m_header[0]);        // Supply header buffer to kernel.
                m_searchKernel.setArg(2, m_dag[0]);           // Supply DAG buffer to kernel.
                m_searchKernel.setArg(3, m_dag[1]);           // Supply DAG buffer to kernel.
                m_searchKernel.setArg(4, m_dagItems);
                m_searchKernel.setArg(6, target);

#ifdef DEV_BUILD
                if (g_logOptions & LOG_SWITCH)
                    cllog << "Switch time: "
                          << std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - m_workSwitchStart)
                                 .count()
                          << " us.";
#endif
            }

            // Run the kernel.
            m_searchKernel.setArg(5, startNonce);
            m_queue[0].enqueueNDRangeKernel(
                m_searchKernel, cl::NullRange, m_settings.globalWorkSize, m_settings.localWorkSize);

            if (results.count)
            {
                // Report results while the kernel is running.
                for (uint32_t i = 0; i < results.count; i++)
                {
                    uint64_t nonce = current.startNonce + results.rslt[i].gid;
                    if (nonce != m_lastNonce)
                    {
                        m_lastNonce = nonce;
                        h256 mix;
                        memcpy(mix.data(), (char*)results.rslt[i].mix, sizeof(results.rslt[i].mix));

                        Farm::f().submitProof(Solution{
                            nonce, mix, current, std::chrono::steady_clock::now(), m_index});
                        cllog << EthWhite << "Job: " << current.header.abridged() << " Sol: 0x"
                              << toHex(nonce) << EthReset;
                    }
                }
            }

            current = w;  // kernel now processing newest work
            current.startNonce = startNonce;
            // Increase start nonce for following kernel execution.
            startNonce += m_settings.globalWorkSize;
            // Report hash count
            if (m_settings.noExit)
                updateHashRate(m_settings.globalWorkSize, 1);
            else
                updateHashRate(m_settings.localWorkSize, results.hashCount);
        }

        if (m_queue.size())
            m_queue[0].finish();

        clear_buffer();
    }
    catch (cl::Error const& _e)
    {
        string _what = ethCLErrorHelper("OpenCL Error", _e);
        clear_buffer();
        throw std::runtime_error(_what);
    }
}

void CLMiner::kick_miner()
{
    // Memory for abort Cannot be static because crashes on macOS.
    const uint32_t one = 1;
    if (!m_settings.noExit && !m_abortqueue.empty())
        m_abortqueue[0].enqueueWriteBuffer(
            m_searchBuffer[0], CL_TRUE, offsetof(SearchResults, abort), sizeof(one), &one);

    m_new_work_signal.notify_one();
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
        else if (platformName == "Clover" || platformName == "Intel Gen OCL Driver")
            platformType = ClPlatformTypeEnum::Clover;
        else if (platformName == "NVIDIA CUDA")
            platformType = ClPlatformTypeEnum::Nvidia;
        else if (platformName.find("Intel") != string::npos)
            platformType = ClPlatformTypeEnum::Intel;
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
            else if (clDeviceType == DeviceTypeEnum::Gpu && platformType == ClPlatformTypeEnum::Intel)
            {
                std::ostringstream s;
                s << "Intel GPU " << pIdx << "." << dIdx;
                uniqueId = s.str();

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

    m_device = devices.at(m_deviceDescriptor.clDeviceOrdinal);

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
    else if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Intel)
    {
        m_hwmoninfo.deviceType = HwMonitorInfoType::UNKNOWN;
        m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
        m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)
        m_settings.noBinary = true;
        m_settings.noExit = true;
    }
    else
    {
        // Don't know what to do with this
        cllog << "Unrecognized Platform";
        return false;
    }
    if (!m_settings.noExit && (m_hwmoninfo.deviceType != HwMonitorInfoType::AMD))
    {
        m_settings.noExit = true;
        cllog << "no exit option enabled for non AMD opencl device";
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
    s << "Using Device : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.clName;

    if (!m_deviceDescriptor.clNvCompute.empty())
        s << " (Compute " + m_deviceDescriptor.clNvCompute + ")";
    else
        s << " " << m_deviceDescriptor.clDeviceVersion;

    s << " Memory : " << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);
    s << " (" << m_deviceDescriptor.totalMemory << " B)";
    cllog << s.str();

    if ((m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Amd) &&
        (m_deviceDescriptor.clMaxComputeUnits != 36))
    {
        m_settings.globalWorkSize =
            (m_settings.globalWorkSize * m_deviceDescriptor.clMaxComputeUnits) / 36;
        // make sure that global work size is evenly divisible by the local workgroup size
        if (m_settings.globalWorkSize % m_settings.localWorkSize != 0)
            m_settings.globalWorkSize =
                ((m_settings.globalWorkSize / m_settings.localWorkSize) + 1) *
                m_settings.localWorkSize;
        cnote << "Adjusting CL work multiplier for " << m_deviceDescriptor.clMaxComputeUnits
              << " CUs. Adjusted work multiplier: "
              << m_settings.globalWorkSize / m_settings.localWorkSize;
    }


    return true;

}

bool CLMiner::initEpoch_internal()
{
    auto startInit = std::chrono::steady_clock::now();
    size_t RequiredMemory = (m_epochContext.dagSize);

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

    cllog << "Generating split DAG + Light (total): "
          << dev::getFormattedMemory((double)RequiredMemory);

    try
    {

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
        // create context
        m_context.clear();
        m_context.push_back(cl::Context(vector<cl::Device>(&m_device, &m_device + 1)));
        m_queue.clear();
        m_queue.push_back(cl::CommandQueue(m_context[0], m_device));


        m_dagItems = m_epochContext.dagNumItems;

        // patch source code
        // note: The kernels here are simply compiled version of the respective .cl kernels
        // into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
        // See libethash-cl/CMakeLists.txt: add_custom_command()
        // TODO: Just use C++ raw string literal.
        string code;

        cllog << "OpenCL kernel";
        code = string(ethash_cl, ethash_cl + sizeof(ethash_cl));

        addDefinition(code, "WORKSIZE", m_settings.localWorkSize);
        addDefinition(code, "ACCESSES", 64);
        addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
        addDefinition(code, "PLATFORM", static_cast<unsigned>(m_deviceDescriptor.clPlatformType));
        addDefinition(code, "COMPUTE", computeCapability);

        if (m_deviceDescriptor.clPlatformType == ClPlatformTypeEnum::Clover)
            addDefinition(code, "LEGACY", 1);

        if (!m_settings.noExit)
            addDefinition(code, "FAST_EXIT", 1);


        // create miner OpenCL program
        cl::Program::Sources sources{{code.data(), code.size()}};
        cl::Program program(m_context[0], sources), binaryProgram;
        try
        {
            program.build({m_device}, options);
        }
        catch (cl::BuildError const& buildErr)
        {
            cwarn << "OpenCL kernel build log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
            cwarn << "OpenCL kernel build error (" << buildErr.err() << "):\n" << buildErr.what();
            pause(MinerPauseEnum::PauseDueToInitEpochError);
            return true;
        }

        /* If we have a binary kernel, we load it in tandem with the opencl,
           that way, we can use the dag generate opencl code and fall back on
           the default kernel if loading fails for whatever reason */
        bool loadedBinary = false;
        std::string device_name = m_deviceDescriptor.clName;

        if (!m_settings.noBinary)
        {
            std::ifstream kernel_file;
            vector<unsigned char> bin_data;
            std::stringstream fname_strm;

            /* Open kernels/ethash_{devicename}_lws{local_work_size}.bin */
            std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
            fname_strm << boost::dll::program_location().parent_path().string()
                       << "/kernels/ethash_" << device_name << "_lws" << m_settings.localWorkSize
                       << (m_settings.noExit ? "" : "_exit") << ".bin";
            cllog << "Loading binary kernel " << fname_strm.str();
            try
            {
                kernel_file.open(fname_strm.str(), ios::in | ios::binary);

                if (kernel_file.good())
                {
                    /* Load the data vector with file data */
                    kernel_file.unsetf(std::ios::skipws);
                    bin_data.insert(bin_data.begin(),
                        std::istream_iterator<unsigned char>(kernel_file),
                        std::istream_iterator<unsigned char>());

                    /* Setup the program */
                    cl::Program::Binaries blobs({bin_data});
                    cl::Program program(m_context[0], {m_device}, blobs);
                    try
                    {
                        program.build({m_device}, options);
                        cllog << "Build info success:"
                              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
                        binaryProgram = program;
                        loadedBinary = true;
                    }
                    catch (cl::Error const&)
                    {
                        cwarn << "Build failed! Info:"
                              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device);
                        cwarn << fname_strm.str();
                        cwarn << "Falling back to OpenCL kernel...";
                    }
                }
                else
                {
                    cwarn << "Failed to load binary kernel: " << fname_strm.str();
                    cwarn << "Falling back to OpenCL kernel...";
                }
            }
            catch (...)
            {
                cwarn << "Failed to load binary kernel: " << fname_strm.str();
                cwarn << "Falling back to OpenCL kernel...";
            }
        }

        // create buffer for dag
        try
        {
            cllog << "Creating DAG buffer, size: "
                  << dev::getFormattedMemory((double)m_epochContext.dagSize)
                  << ", free: "
                  << dev::getFormattedMemory(
                         (double)(m_deviceDescriptor.totalMemory - RequiredMemory));
            m_dag.clear();
            if (m_epochContext.dagNumItems & 1)
            {
                m_dag.push_back(
                    cl::Buffer(m_context[0], CL_MEM_READ_ONLY, m_epochContext.dagSize / 2 + 64));
                m_dag.push_back(
                    cl::Buffer(m_context[0], CL_MEM_READ_ONLY, m_epochContext.dagSize / 2 - 64));
            }
            else
            {
                m_dag.push_back(
                    cl::Buffer(m_context[0], CL_MEM_READ_ONLY, (m_epochContext.dagSize) / 2));
                m_dag.push_back(
                    cl::Buffer(m_context[0], CL_MEM_READ_ONLY, (m_epochContext.dagSize) / 2));
            }
            cllog << "Creating light cache buffer, size: "
                  << dev::getFormattedMemory((double)m_epochContext.lightSize);
            m_light.clear();
            bool light_on_host = false;
            try
            {
                m_light.emplace_back(m_context[0], CL_MEM_READ_ONLY, m_epochContext.lightSize);
            }
            catch (cl::Error const& err)
            {
                if ((err.err() == CL_OUT_OF_RESOURCES) || (err.err() == CL_OUT_OF_HOST_MEMORY))
                {
                    // Ok, no room for light cache on GPU. Try allocating on host
                    clog(WarnChannel) << "No room on GPU, allocating light cache on host";
                    clog(WarnChannel) << "Generating DAG will take minutes instead of seconds";
                    light_on_host = true;
                }
                else
                    throw;
            }
            if (light_on_host)
            {
                m_light.emplace_back(m_context[0], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    m_epochContext.lightSize);
                cllog << "WARNING: Generating DAG will take minutes, not seconds";
            }
            cllog << "Loading kernels";

            // If we have a binary kernel to use, let's try it
            // otherwise just do a normal opencl load
            if (loadedBinary)
                m_searchKernel = cl::Kernel(binaryProgram, "search");
            else
                m_searchKernel = cl::Kernel(program, "search");

            m_dagKernel = cl::Kernel(program, "GenerateDAG");

            m_queue[0].enqueueWriteBuffer(
                m_light[0], CL_TRUE, 0, m_epochContext.lightSize, m_epochContext.lightCache);
        }
        catch (cl::Error const& err)
        {
            cwarn << ethCLErrorHelper("Creating DAG buffer failed", err);
            pause(MinerPauseEnum::PauseDueToInitEpochError);
            return true;
        }
        // create buffer for header
        cllog << "Creating buffer for header.";
        m_header.clear();
        m_header.push_back(cl::Buffer(m_context[0], CL_MEM_READ_ONLY, 32));

        m_searchKernel.setArg(1, m_header[0]);
        m_searchKernel.setArg(2, m_dag[0]);
        m_searchKernel.setArg(3, m_dag[1]);
        m_searchKernel.setArg(4, m_dagItems);

        // create mining buffers
        cllog << "Creating mining buffer";
        m_searchBuffer.clear();
        m_searchBuffer.emplace_back(m_context[0], CL_MEM_WRITE_ONLY, sizeof(SearchResults));

        m_dagKernel.setArg(1, m_light[0]);
        m_dagKernel.setArg(2, m_dag[0]);
        m_dagKernel.setArg(3, m_dag[1]);
        m_dagKernel.setArg(4, (uint32_t)(m_epochContext.lightSize / 64));

        const uint32_t workItems = m_dagItems * 2;  // GPU computes partial 512-bit DAG items.

        uint32_t start;
        const uint32_t chunk = 10000 * m_settings.localWorkSize;
        for (start = 0; start <= workItems - chunk; start += chunk)
        {
            m_dagKernel.setArg(0, start);
            m_queue[0].enqueueNDRangeKernel(
                m_dagKernel, cl::NullRange, chunk, m_settings.localWorkSize);
            m_queue[0].finish();
        }
        if (start < workItems)
        {
            uint32_t groupsLeft = workItems - start;
            groupsLeft = (groupsLeft + m_settings.localWorkSize - 1) / m_settings.localWorkSize;
            m_dagKernel.setArg(0, start);
            m_queue[0].enqueueNDRangeKernel(m_dagKernel, cl::NullRange,
                groupsLeft * m_settings.localWorkSize, m_settings.localWorkSize);
            m_queue[0].finish();
        }

        auto dagTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startInit);
        cllog << dev::getFormattedMemory((double)m_epochContext.dagSize)
              << " of DAG data generated in "
              << dagTime.count() << " ms.";
    }
    catch (cl::Error const& err)
    {
        cllog << ethCLErrorHelper("OpenCL init failed", err);
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        return false;
    }
    return true;
}
