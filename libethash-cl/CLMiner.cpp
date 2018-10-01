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
unsigned CLMiner::s_workgroupSize = CLMiner::c_defaultLocalWorkSize;
unsigned CLMiner::s_initialGlobalWorkSize =
    CLMiner::c_defaultGlobalWorkSizeMultiplier * CLMiner::c_defaultLocalWorkSize;

// WARNING: Do not change the value of the following constant
// unless you are prepared to make the neccessary adjustments
// to the assembly code for the binary kernels.
const size_t c_maxSearchResults = 15;

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
            cwarn << "No OpenCL platforms found";
        else
#endif
            throw err;
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

unsigned CLMiner::s_platformId = 0;
unsigned CLMiner::s_numInstances = 0;
vector<int> CLMiner::s_devices(MAX_MINERS, -1);
bool CLMiner::s_noBinary = false;

CLMiner::CLMiner(unsigned _index) : Miner("cl-", _index) {}

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
    current.header = h256{1u};

    try
    {
        while (!shouldStop())
        {
            if (is_mining_paused())
            {
                // cnote << "Mining is paused: Waiting for 3s.";
                std::this_thread::sleep_for(std::chrono::seconds(3));
                continue;
            }

            // Read results.
            volatile SearchResults results;
#ifdef DEV_BUILD
            std::chrono::steady_clock::time_point submitStart;
#endif

            if (m_queue.size())
            {
                // no need to read the abort flag.
                m_queue[0].enqueueReadBuffer(m_searchBuffer[0], CL_TRUE,
                    c_maxSearchResults * sizeof(results.rslt[0]), 2 * sizeof(results.count),
                    (void*)&results.count);
                if (results.count)
                {
#ifdef DEV_BUILD
                    submitStart = std::chrono::steady_clock::now();
#endif
                    m_queue[0].enqueueReadBuffer(m_searchBuffer[0], CL_TRUE, 0,
                        results.count * sizeof(results.rslt[0]), (void*)&results);
                    // Reset search buffer if any solution found.
                }
                // clean the solution count, hash count, and abort flag
                m_queue[0].enqueueWriteBuffer(m_searchBuffer[0], CL_FALSE,
                    offsetof(SearchResults, count), sizeof(zerox3), zerox3);
            }
            else
                results.count = 0;

            const WorkPackage w = work();

            if (current.header != w.header)
            {
                // New work received. Update GPU data.
                if (!w)
                {
                    cllog << "No work. Pause for 3 s.";
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    continue;
                }

                if (current.epoch != w.epoch)
                {
                    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
                    {
                        while (s_dagLoadIndex < Index())
                            this_thread::sleep_for(chrono::seconds(1));
                        ++s_dagLoadIndex;
                    }

                    m_abortqueue.clear();
                    init(w.epoch);
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
                    offsetof(SearchResults, count), sizeof(zerox3), zerox3);

                m_searchKernel.setArg(0, m_searchBuffer[0]);  // Supply output buffer to kernel.
                m_searchKernel.setArg(1, m_header[0]);        // Supply header buffer to kernel.
                m_searchKernel.setArg(2, m_dag[0]);           // Supply DAG buffer to kernel.
                m_searchKernel.setArg(3, m_dagItems);
                m_searchKernel.setArg(5, target);
                m_searchKernel.setArg(6, 0xffffffff);

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
            m_searchKernel.setArg(4, startNonce);
            m_queue[0].enqueueNDRangeKernel(
                m_searchKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);

            if (results.count)
            {
                // Report results while the kernel is running.
                for (uint32_t i = 0; i < results.count; i++)
                {
                    uint64_t nonce = current.startNonce + results.rslt[i].gid;
                    if (nonce != m_lastNonce)
                    {
                        m_lastNonce = nonce;
                        if (s_noeval)
                        {
                            h256 mix;
                            memcpy(mix.data(), (char*)results.rslt[i].mix,
                                sizeof(results.rslt[i].mix));
                            Farm::f().submitProof(Solution{nonce, mix, current, false, Index()});
                        }
                        else
                        {
                            Result r = EthashAux::eval(current.epoch, current.header, nonce);
                            if (r.value <= current.boundary)
                                Farm::f().submitProof(
                                    Solution{nonce, r.mixHash, current, false, Index()});
                            else
                            {
                                Farm::f().failedSolution(Index());
                                cwarn << "GPU gave incorrect result!";
                            }
                        }
                    }
                }
#ifdef DEV_BUILD
                if (g_logOptions & LOG_SUBMIT)
                    cllog << "Submit time: "
                          << std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - submitStart)
                                 .count()
                          << " us.";
#endif
            }

            current = w;  // kernel now processing newest work
            current.startNonce = startNonce;
            // Increase start nonce for following kernel execution.
            startNonce += results.hashCount * m_workgroupSize;

            // Report hash count
            updateHashRate(m_workgroupSize, results.hashCount);
        }
        m_queue[0].finish();
    }
    catch (cl::Error const& _e)
    {
        cwarn << ethCLErrorHelper("OpenCL Error", _e);
        if (s_exit)
            exit(1);
    }
}

void CLMiner::kick_miner()
{
    // Memory for abort Cannot be static because crashes on macOS.
    const uint32_t one = 1;
    if (m_abortqueue.size())
        m_abortqueue[0].enqueueWriteBuffer(
            m_searchBuffer[0], CL_TRUE, offsetof(SearchResults, abort), sizeof(one), &one);
}

unsigned CLMiner::getNumDevices()
{
    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return 0;

    vector<cl::Device> devices = getDevices(platforms, s_platformId);
    if (devices.empty())
    {
        cwarn << "No OpenCL devices found.";
        return 0;
    }
    return devices.size();
}

void CLMiner::listDevices()
{
    string outString = "\nListing OpenCL devices.\nFORMAT: [platformID] [deviceID] deviceName\n";
    unsigned int i = 0;

    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return;
    for (unsigned j = 0; j < platforms.size(); ++j)
    {
        i = 0;
        vector<cl::Device> devices = getDevices(platforms, j);
        for (auto const& device : devices)
        {
            outString += "[" + to_string(j) + "] [" + to_string(i) + "] " +
                         device.getInfo<CL_DEVICE_NAME>() + "\n";
            outString += "\tCL_DEVICE_TYPE: ";
            switch (device.getInfo<CL_DEVICE_TYPE>())
            {
            case CL_DEVICE_TYPE_CPU:
                outString += "CPU\n";
                break;
            case CL_DEVICE_TYPE_GPU:
                outString += "GPU\n";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                outString += "ACCELERATOR\n";
                break;
            default:
                outString += "DEFAULT\n";
                break;
            }
            outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " +
                         to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
            outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " +
                         to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
            outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " +
                         to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
            ++i;
        }
    }
    std::cout << outString;
}

bool CLMiner::configureGPU(unsigned _localWorkSize, unsigned _globalWorkSizeMultiplier,
    unsigned _platformId, int epoch, unsigned _dagLoadMode, unsigned _dagCreateDevice, bool _noeval,
    bool _exit, bool _nobinary)
{
    s_noeval = _noeval;
    s_dagLoadMode = _dagLoadMode;
    s_dagCreateDevice = _dagCreateDevice;
    s_exit = _exit;
    s_noBinary = _nobinary;

    s_platformId = _platformId;

    _localWorkSize = ((_localWorkSize + 7) / 8) * 8;
    s_workgroupSize = _localWorkSize;
    s_initialGlobalWorkSize = _globalWorkSizeMultiplier * _localWorkSize;

    auto dagSize = ethash::get_full_dataset_size(ethash::calculate_full_dataset_num_items(epoch));

    vector<cl::Platform> platforms = getPlatforms();
    if (platforms.empty())
        return false;
    if (_platformId >= platforms.size())
        return false;

    vector<cl::Device> devices = getDevices(platforms, _platformId);
    bool foundSuitableDevice = false;
    for (auto const& device : devices)
    {
        cl_ulong result = 0;
        device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
        if (result >= dagSize)
        {
            cnote << "Found suitable OpenCL device [" << device.getInfo<CL_DEVICE_NAME>()
                  << "] with " << FormattedMemSize(result) << " of GPU memory";
            foundSuitableDevice = true;
        }
        else
        {
            cnote << "OpenCL device " << device.getInfo<CL_DEVICE_NAME>()
                  << " has insufficient GPU memory." << FormattedMemSize(result)
                  << " GB of memory found < " << FormattedMemSize(dagSize) << " of memory required";
        }
    }
    if (foundSuitableDevice)
    {
        return true;
    }
    cout << "No GPU device with sufficient memory was found" << endl;
    return false;
}

bool CLMiner::init(int epoch)
{
    // get all platforms
    try
    {
        vector<cl::Platform> platforms = getPlatforms();
        if (platforms.empty())
            return false;

        // use selected platform
        unsigned platformIdx = min<unsigned>(s_platformId, platforms.size() - 1);

        string platformName = platforms[platformIdx].getInfo<CL_PLATFORM_NAME>();
        ETHCL_LOG("Platform: " << platformName);

        int platformId = OPENCL_PLATFORM_UNKNOWN;
        if (platformName == "NVIDIA CUDA")
        {
            platformId = OPENCL_PLATFORM_NVIDIA;
            m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
            m_hwmoninfo.indexSource = HwMonitorIndexSource::OPENCL;
        }
        else if (platformName == "AMD Accelerated Parallel Processing")
        {
            platformId = OPENCL_PLATFORM_AMD;
            m_hwmoninfo.deviceType = HwMonitorInfoType::AMD;
            m_hwmoninfo.indexSource = HwMonitorIndexSource::OPENCL;
        }
        else if (platformName == "Clover")
        {
            platformId = OPENCL_PLATFORM_CLOVER;
        }

        // get GPU device of the default platform
        vector<cl::Device> devices = getDevices(platforms, platformIdx);
        if (devices.empty())
        {
            ETHCL_LOG("No OpenCL devices found.");
            return false;
        }

        // use selected device
        int idx = Index() % devices.size();
        unsigned deviceId = s_devices[idx] > -1 ? s_devices[idx] : Index();
        m_hwmoninfo.deviceIndex = deviceId % devices.size();
        m_device = devices[deviceId % devices.size()];
        string device_version = m_device.getInfo<CL_DEVICE_VERSION>();
        string device_name = m_device.getInfo<CL_DEVICE_NAME>();
        ETHCL_LOG("Device:   " << device_name << " / " << device_version);

        string clVer = device_version.substr(7, 3);
        if (clVer == "1.0" || clVer == "1.1")
        {
            if (platformId == OPENCL_PLATFORM_CLOVER)
            {
                ETHCL_LOG("OpenCL " << clVer
                                    << " not supported, but platform Clover might work "
                                       "nevertheless. USE AT OWN RISK!");
            }
            else
            {
                ETHCL_LOG("OpenCL " << clVer << " not supported - minimum required version is 1.2");
                return false;
            }
        }

        char options[256] = {0};
        int computeCapability = 0;
#ifndef __clang__
        if (platformId == OPENCL_PLATFORM_NVIDIA)
        {
            cl_uint computeCapabilityMajor =
                m_device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>();
            cl_uint computeCapabilityMinor =
                m_device.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>();

            computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
            int maxregs = computeCapability >= 35 ? 72 : 63;
            sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
        }
#endif
        // create context
        m_context.clear();
        m_context.push_back(cl::Context(vector<cl::Device>(&m_device, &m_device + 1)));
        m_queue.clear();
        m_queue.push_back(cl::CommandQueue(m_context[0], m_device));

        m_workgroupSize = s_workgroupSize;
        m_globalWorkSize = s_initialGlobalWorkSize;

        unsigned int computeUnits = m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        // Apparently some 36 CU devices return a bogus 14!!!
        computeUnits = computeUnits == 14 ? 36 : computeUnits;
        if ((platformId == OPENCL_PLATFORM_AMD) && (computeUnits != 36))
        {
            m_globalWorkSize = (m_globalWorkSize * computeUnits) / 36;
            // make sure that global work size is evenly divisible by the local workgroup size
            if (m_globalWorkSize % m_workgroupSize != 0)
                m_globalWorkSize = ((m_globalWorkSize / m_workgroupSize) + 1) * m_workgroupSize;
            cnote << "Adjusting CL work multiplier for " << computeUnits << " CUs."
                  << "Adjusted work multiplier: " << m_globalWorkSize / m_workgroupSize;
        }

        const auto& context = ethash::get_global_epoch_context(epoch);
        const auto lightNumItems = context.light_cache_num_items;
        const auto lightSize = ethash::get_light_cache_size(lightNumItems);
        m_dagItems = context.full_dataset_num_items;
        const auto dagSize = ethash::get_full_dataset_size(m_dagItems);

        // patch source code
        // note: The kernels here are simply compiled version of the respective .cl kernels
        // into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
        // See libethash-cl/CMakeLists.txt: add_custom_command()
        // TODO: Just use C++ raw string literal.
        string code;

        cllog << "OpenCL kernel";
        code = string(ethash_cl, ethash_cl + sizeof(ethash_cl));

        addDefinition(code, "WORKSIZE", m_workgroupSize);
        addDefinition(code, "ACCESSES", 64);
        addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
        addDefinition(code, "PLATFORM", platformId);
        addDefinition(code, "COMPUTE", computeCapability);
        if (platformId == OPENCL_PLATFORM_CLOVER)
        {
            addDefinition(code, "LEGACY", 1);
            s_noBinary = true;
        }

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
            return false;
        }

        /* If we have a binary kernel, we load it in tandem with the opencl,
           that way, we can use the dag generate opencl code and fall back on
           the default kernel if loading fails for whatever reason */
        bool loadedBinary = false;

        if (!s_noBinary)
        {
            std::ifstream kernel_file;
            vector<unsigned char> bin_data;
            std::stringstream fname_strm;

            /* Open kernels/ethash_{devicename}_lws{local_work_size}.bin */
            std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
            fname_strm << boost::dll::program_location().parent_path().string()
#if defined(_WIN32)
                       << "\\kernels\\ethash_"
#else
                       << "/kernels/ethash_"
#endif
                       << device_name << "_lws" << m_workgroupSize << ".bin";
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

        // check whether the current dag fits in memory everytime we recreate the DAG
        cl_ulong result = 0;
        m_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
        if (result < dagSize)
        {
            cnote << "OpenCL device " << device_name << " has insufficient GPU memory."
                  << FormattedMemSize(result) << " of memory found, " << FormattedMemSize(dagSize)
                  << " of memory required";
            return false;
        }

        // create buffer for dag
        try
        {
            cllog << "Creating light cache buffer, size: " << FormattedMemSize(lightSize);
            m_light.clear();
            m_light.push_back(cl::Buffer(m_context[0], CL_MEM_READ_ONLY, lightSize));
            cllog << "Creating DAG buffer, size: " << FormattedMemSize(dagSize)
                  << ", free: " << FormattedMemSize(result - lightSize - dagSize);
            m_dag.clear();
            m_dag.push_back(cl::Buffer(m_context[0], CL_MEM_READ_ONLY, dagSize));
            cllog << "Loading kernels";

            // If we have a binary kernel to use, let's try it
            // otherwise just do a normal opencl load
            if (loadedBinary)
                m_searchKernel = cl::Kernel(binaryProgram, "search");
            else
                m_searchKernel = cl::Kernel(program, "search");

            m_dagKernel = cl::Kernel(program, "GenerateDAG");

            cllog << "Writing light cache buffer";
            m_queue[0].enqueueWriteBuffer(m_light[0], CL_TRUE, 0, lightSize, context.light_cache);
        }
        catch (cl::Error const& err)
        {
            cwarn << ethCLErrorHelper("Creating DAG buffer failed", err);
            return false;
        }
        // create buffer for header
        ETHCL_LOG("Creating buffer for header.");
        m_header.clear();
        m_header.push_back(cl::Buffer(m_context[0], CL_MEM_READ_ONLY, 32));

        m_searchKernel.setArg(1, m_header[0]);
        m_searchKernel.setArg(2, m_dag[0]);
        m_searchKernel.setArg(3, m_dagItems);
        m_searchKernel.setArg(6, ~0u);

        // create mining buffers
        ETHCL_LOG("Creating mining buffer");
        m_searchBuffer.clear();
        m_searchBuffer.push_back(
            cl::Buffer(m_context[0], CL_MEM_WRITE_ONLY, sizeof(SearchResults)));

        m_dagKernel.setArg(1, m_light[0]);
        m_dagKernel.setArg(2, m_dag[0]);
        m_dagKernel.setArg(3, (uint32_t)(lightSize / 64));
        m_dagKernel.setArg(4, ~0u);

        const uint32_t workItems = m_dagItems * 2;  // GPU computes partial 512-bit DAG items.

        auto startDAG = std::chrono::steady_clock::now();
        uint32_t start;
        const uint32_t chunk = 10000 * m_workgroupSize;
        for (start = 0; start <= workItems - chunk; start += chunk)
        {
            m_dagKernel.setArg(0, start);
            m_queue[0].enqueueNDRangeKernel(m_dagKernel, cl::NullRange, chunk, m_workgroupSize);
            m_queue[0].finish();
        }
        if (start < workItems)
        {
            uint32_t groupsLeft = workItems - start;
            groupsLeft = (groupsLeft + m_workgroupSize - 1) / m_workgroupSize;
            m_dagKernel.setArg(0, start);
            m_queue[0].enqueueNDRangeKernel(
                m_dagKernel, cl::NullRange, groupsLeft * m_workgroupSize, m_workgroupSize);
            m_queue[0].finish();
        }
        auto endDAG = std::chrono::steady_clock::now();

        auto dagTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDAG - startDAG);
        cnote << FormattedMemSize(dagSize) << " of DAG data generated in " << dagTime.count()
              << " ms.";
    }
    catch (cl::Error const& err)
    {
        cwarn << ethCLErrorHelper("OpenCL init failed", err);
        if (s_exit)
            exit(1);
        return false;
    }
    return true;
}

