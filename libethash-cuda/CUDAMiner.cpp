/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "CUDAMiner.h"
#include "nvrtc.h"


using namespace std;
using namespace dev;
using namespace eth;

struct CUDAChannel : public LogChannel
{
    static const char* name() { return EthOrange "cu"; }
    static const int verbosity = 2;
};
#define cudalog clog(CUDAChannel)

CUDAMiner::CUDAMiner(unsigned _index, CUSettings _settings, DeviceDescriptor& _device)
  : Miner("cuda-", _index),
    m_settings(_settings),
    m_batch_size(_settings.gridSize * _settings.blockSize),
    m_streams_batch_size(_settings.gridSize * _settings.blockSize * _settings.streams)
{
    m_deviceDescriptor = _device;
}

bool CUDAMiner::initDevice()
{
    cudalog << "Using Pci Id : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.cuName
            << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
            << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory)
            << " Grid size : " << m_settings.gridSize << " Block size : " << m_settings.blockSize;

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)

    try
    {
        CUDA_SAFE_CALL(cudaSetDevice(m_deviceDescriptor.cuDeviceIndex));
        CUDA_SAFE_CALL(cudaDeviceReset());
    }
    catch (const cuda_runtime_error& ec)
    {
        cudalog << "Could not set CUDA device on Pci Id " << m_deviceDescriptor.uniqueId
                << " Error : " << ec.what();
        cudalog << "Mining aborted on this device.";
        return false;
    }
    return true;
}

bool CUDAMiner::initEpoch_internal()
{
    // If we get here it means epoch has changed so it's not necessary
    // to check again dag sizes. They're changed for sure
    bool retVar = false;
    m_current_target = 0;
    auto startInit = std::chrono::steady_clock::now();
    size_t RequiredMemory = (m_epochContext.dagSize + m_epochContext.lightSize);

    // Release the pause flag if any
    resume(MinerPauseEnum::PauseDueToInsufficientMemory);
    resume(MinerPauseEnum::PauseDueToInitEpochError);

    try
    {
        // If we have already enough memory allocated, we just have to
        // copy light_cache and regenerate the DAG
        if (m_allocated_memory_dag < m_epochContext.dagSize ||
            m_allocated_memory_light_cache < m_epochContext.lightSize)
        {
            // We need to reset the device and (re)create the dag
            // cudaDeviceReset() frees all previous allocated memory
            CUDA_SAFE_CALL(cudaDeviceReset());
            CUDA_SAFE_CALL(cudaSetDeviceFlags(m_settings.schedule));
            // CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

            // Check whether the current device has sufficient memory every time we recreate the dag
            if (m_deviceDescriptor.totalMemory < RequiredMemory)
            {
                cudalog << "Epoch " << m_epochContext.epochNumber << " requires "
                        << dev::getFormattedMemory((double)RequiredMemory) << " memory.";
                cudalog << "This device hasn't available. Mining suspended ...";
                pause(MinerPauseEnum::PauseDueToInsufficientMemory);
                return true;  // This will prevent to exit the thread and
                              // Eventually resume mining when changing coin or epoch (NiceHash)
            }

            cudalog << "Generating DAG + Light : "
                    << dev::getFormattedMemory((double)RequiredMemory);

            // create buffer for cache
            CUDA_SAFE_CALL(
                cudaMalloc(reinterpret_cast<void**>(&m_light), m_epochContext.lightSize));
            m_allocated_memory_light_cache = m_epochContext.lightSize;
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_dag), m_epochContext.dagSize));
            m_allocated_memory_dag = m_epochContext.dagSize;

            // create mining buffers
            for (unsigned i = 0; i != m_settings.streams; ++i)
            {
                CUDA_SAFE_CALL(cudaMallocHost(&m_search_results[i], sizeof(search_results)));
                CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
            }
        }
        else
        {
            cudalog << "Generating DAG + Light (reusing buffers): "
                    << dev::getFormattedMemory((double)RequiredMemory);
            get_constants(&m_dag, NULL, &m_light, NULL);
        }

        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(m_light), m_epochContext.lightCache,
            m_epochContext.lightSize, cudaMemcpyHostToDevice));

        set_constants(m_dag, m_epochContext.dagNumItems, m_light,
            m_epochContext.lightNumItems);  // in ethash_cuda_miner_kernel.cu

        ethash_generate_dag(
            m_epochContext.dagSize, m_settings.gridSize, m_settings.blockSize, m_streams[0]);

        cudalog << "Generated DAG + Light in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startInit)
                       .count()
                << " ms. "
                << dev::getFormattedMemory(
                       (double)(m_deviceDescriptor.totalMemory - RequiredMemory))
                << " left.";


        m_dag_progpow = reinterpret_cast<hash64_t*>(m_dag);
        retVar = true;
    }
    catch (const cuda_runtime_error& ec)
    {
        cudalog << "Unexpected error " << ec.what() << " on CUDA device "
                << m_deviceDescriptor.uniqueId;
        cudalog << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        retVar = true;
    }

    return retVar;
}

void CUDAMiner::workLoop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cu-" << m_index << " CUDAMiner::workLoop() begin");

    m_search_results.resize(m_settings.streams);
    m_streams.resize(m_settings.streams);

    if (!initDevice())
        return;

    try
    {
        minerLoop();  // In base class Miner

        // Reset miner and stop working
        CUDA_SAFE_CALL(cudaDeviceReset());
    }
    catch (cuda_runtime_error const& _e)
    {
        string _what = "GPU error: ";
        _what.append(_e.what());
        throw std::runtime_error(_what);
    }

    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cu-" << m_index << " CUDAMiner::workLoop() end");
}

void CUDAMiner::compileProgPoWKernel(int _block, int _dagelms)
{
    auto startCompile = std::chrono::steady_clock::now();
    cudalog << "Compiling ProgPoW kernel for period : " << (_block / PROGPOW_PERIOD);

    const char* name = "progpow_search";
    std::string text = ProgPow::getKern(_block, ProgPow::KERNEL_CUDA);
    text += std::string(cu_progpow_miner_kernel(), sizeof_cu_progpow_miner_kernel());

    ofstream write;
    write.open("kernel.cu");
    write << text;
    write.close();

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,  // prog
        text.c_str(),                          // buffer
        "kernel.cu",                           // name
        0,                                     // numHeaders
        NULL,                                  // headers
        NULL));                                // includeNames

    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, name));
    std::string op_arch = "--gpu-architecture=compute_" +
                          to_string(m_deviceDescriptor.cuComputeMajor) +
                          to_string(m_deviceDescriptor.cuComputeMinor);
    std::string op_dag = "-DPROGPOW_DAG_ELEMENTS=" + to_string(_dagelms);

    const char* opts[] = {op_arch.c_str(), op_dag.c_str(), "-lineinfo"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
        3,                                                 // numOptions
        opts);                                             // options

#ifdef _DEVELOPER

    if (g_logOptions & LOG_COMPILE)
    {
        // Obtain compilation log from the program.
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char* log = new char[logSize];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
        cudalog << "Compile log: " << log;
        delete[] log;
    }

#endif

    NVRTC_SAFE_CALL(compileResult);

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    write.open("kernel.ptx");
    write << ptx;
    write.close();

    // Load the generated PTX and get a handle to the kernel.
    char* jitInfo = new char[32 * 1024];
    char* jitErr = new char[32 * 1024];
    CUjit_option jitOpt[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE,
        CU_JIT_GENERATE_LINE_INFO};

    void* jitOptVal[] = {
        jitInfo, jitErr, (void*)(32 * 1024), (void*)(32 * 1024), (void*)(1), (void*)(1)};

    CU_SAFE_CALL(cuModuleLoadDataEx(&m_module, ptx, 6, jitOpt, jitOptVal));
    m_progpow_kernel_loaded = true;

#ifdef _DEVELOPER
    if (g_logOptions & LOG_COMPILE)
    {
        cudalog << "JIT info: \n" << jitInfo;
        cudalog << "JIT err: \n" << jitErr;
    }
#endif

    delete[] ptx;
    delete[] jitInfo;
    delete[] jitErr;

    // Find the mangled name
    const char* mangledName;
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, name, &mangledName));
#ifdef _DEVELOPER
    if (g_logOptions & LOG_COMPILE)
        cudalog << "Mangled name: " << mangledName;
#endif
    CU_SAFE_CALL(cuModuleGetFunction(&m_kernel, m_module, mangledName));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    cudalog << "Done compiling in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - startCompile)
                   .count()
            << " ms. ";
}

void CUDAMiner::unloadProgPoWKernel()
{
    if (!m_progpow_kernel_loaded)
        return;
    cudalog << "UNLOADING";
    CU_SAFE_CALL(cuModuleUnload(m_module));
    m_progpow_kernel_loaded = false;
}

int CUDAMiner::getNumDevices()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver)
    {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            std::cerr << "CUDA Error : No CUDA driver found" << std::endl;
        else
            std::cerr << "CUDA Error : Insufficient CUDA driver " << std::to_string(driverVersion)
                      << std::endl;
    }
    else
    {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}

void CUDAMiner::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
    int numDevices = getNumDevices();

    for (int i = 0; i < numDevices; i++)
    {
        string uniqueId;
        ostringstream s;
        DeviceDescriptor deviceDescriptor;
        cudaDeviceProp props;

        try
        {
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));
            s << setw(2) << setfill('0') << hex << props.pciBusID << ":" << setw(2)
              << props.pciDeviceID << ".0";
            uniqueId = s.str();

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            deviceDescriptor.name = string(props.name);
            deviceDescriptor.cuDetected = true;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.type = DeviceTypeEnum::Gpu;
            deviceDescriptor.cuDeviceIndex = i;
            deviceDescriptor.cuDeviceOrdinal = i;
            deviceDescriptor.cuName = string(props.name);
            deviceDescriptor.totalMemory = props.totalGlobalMem;
            deviceDescriptor.cuCompute = (to_string(props.major) + "." + to_string(props.minor));
            deviceDescriptor.cuComputeMajor = props.major;
            deviceDescriptor.cuComputeMinor = props.minor;

            _DevicesCollection[uniqueId] = deviceDescriptor;
        }
        catch (const cuda_runtime_error& _e)
        {
            std::cerr << _e.what() << std::endl;
        }
    }
}

void CUDAMiner::ethash_search()
{
    uint64_t startNonce, target;
    startNonce = m_work_active.startNonce;
    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);

    set_header(*reinterpret_cast<hash32_t const*>(m_work_active.header.data()));
    if (m_current_target != target)
    {
        set_target(target);
        m_current_target = target;
    }

    // process batches until we get new work.
    bool done = m_new_work.load(memory_order_relaxed);

    // prime each stream, clear search result buffers and start the search
    uint32_t current_index;
    if (!done)
    {
        for (current_index = 0; current_index < m_settings.streams;
             current_index++, startNonce += m_batch_size)
        {
            cudaStream_t stream = m_streams[current_index];
            volatile search_results& buffer(*m_search_results[current_index]);
            buffer.count = 0;

            // Run the batch for this stream
            run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer,
                startNonce, m_settings.parallelHash);
        }
    }

    while (!done)
    {
        // This inner loop will process each cuda stream individually
        for (current_index = 0; current_index < m_settings.streams;
             current_index++, startNonce += m_batch_size)
        {
            // Each pass of this loop will wait for a stream to exit,
            // save any found solutions, then restart the stream
            // on the next group of nonces.
            cudaStream_t stream = m_streams[current_index];

            // Wait for the stream complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            // Check on every stream if we need to stop
            if (!done)
                done = m_new_work.load(memory_order_relaxed);

            // Detect solutions in current stream's solution buffer
            volatile search_results& buffer(*m_search_results[current_index]);
            uint32_t found_count = std::min((unsigned)buffer.count, MAX_SEARCH_RESULTS);

            if (found_count)
            {
                buffer.count = 0;
                uint64_t nonce_base = startNonce - m_streams_batch_size;

                // Extract solution and pass to higer level
                // using io_service as dispatcher

                for (uint32_t i = 0; i < found_count; i++)
                {
                    h256 mix;
                    uint64_t nonce = nonce_base + buffer.result[i].gid;
                    memcpy(mix.data(), (void*)&buffer.result[i].mix, sizeof(buffer.result[i].mix));
                    auto sol = Solution{
                        nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};

                    cudalog << EthWhite << "Job: " << m_work_active.header.abridged()
                            << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;

                    Farm::f().submitProof(sol);
                }
            }

            // restart the stream on the next batch of nonces
            // unless we are done for this round.
            if (!done)
                run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer,
                    startNonce, m_settings.parallelHash);
        }

        // Update the hash rate
        updateHashRate(m_batch_size, m_settings.streams);
    }

#ifdef _DEVELOPER
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH))
        cudalog << "Switch time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - m_workSwitchStart)
                       .count()
                << " ms.";
#endif
}

void CUDAMiner::progpow_search()
{
    uint64_t startNonce, target;
    startNonce = m_work_active.startNonce;
    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);

    hash32_t header = *reinterpret_cast<hash32_t const*>(m_work_active.header.data());

    // process batches until we get new work.
    bool done = m_new_work.load(memory_order_relaxed);

    // prime each stream, clear search result buffers and start the search
    uint32_t current_index;
    if (!done)
    {
        for (current_index = 0; current_index < m_settings.streams;
             current_index++, startNonce += m_batch_size)
        {
            cudaStream_t stream = m_streams[current_index];
            volatile search_results* buffer = m_search_results[current_index];
            buffer->count = 0;

            // Run the batch for this stream
            bool hack_false = false;
            uint64_t batchNonce = startNonce;
            void* args[] = {&batchNonce, &header, &target, &m_dag_progpow, &buffer, &hack_false};
            CU_SAFE_CALL(cuLaunchKernel(m_kernel, m_settings.gridSize, 1, 1,  // grid dim
                m_settings.blockSize, 1, 1,                                   // block dim
                0,                                                            // shared mem
                stream,                                                       // stream
                args, 0));                                                    // arguments
        }
    }

    while (!done)
    {

        // This inner loop will process each cuda stream individually
        for (current_index = 0; current_index < m_settings.streams;
             current_index++, startNonce += m_batch_size)
        {
            // Each pass of this loop will wait for a stream to exit,
            // save any found solutions, then restart the stream
            // on the next group of nonces.
            cudaStream_t stream = m_streams[current_index];

            // Wait for the stream complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            // Check on every stream if we need to stop
            if (!done)
                done = m_new_work.load(memory_order_relaxed);

            // Detect solutions in current stream's solution buffer
            volatile search_results* buffer = m_search_results[current_index];
            uint32_t found_count = std::min((unsigned)buffer->count, MAX_SEARCH_RESULTS);

            if (found_count)
            {
                buffer->count = 0;
                uint64_t nonce_base = startNonce - m_streams_batch_size;

                // Extract solution and pass to higer level
                // using io_service as dispatcher

                for (uint32_t i = 0; i < found_count; i++)
                {
                    h256 mix;
                    uint64_t nonce = nonce_base + buffer->result[i].gid;
                    memcpy(
                        mix.data(), (void*)&buffer->result[i].mix, sizeof(buffer->result[i].mix));
                    auto sol = Solution{
                        nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};

                    cudalog << EthWhite << "Job: " << m_work_active.header.abridged()
                            << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;

                    Farm::f().submitProof(sol);
                }
            }

            // restart the stream on the next batch of nonces
            // unless we are done for this round.
            if (!done)
            {
                // Run the batch for this stream
                bool hack_false = false;
                uint64_t batchNonce = startNonce;
                void* args[] = {
                    &batchNonce, &header, &target, &m_dag_progpow, &buffer, &hack_false};
                CU_SAFE_CALL(cuLaunchKernel(m_kernel, m_settings.gridSize, 1, 1,  // grid dim
                    m_settings.blockSize, 1, 1,                                   // block dim
                    0,                                                            // shared mem
                    stream,                                                       // stream
                    args, 0));                                                    // arguments
            }
        }

        // Update the hash rate
        updateHashRate(m_batch_size, m_settings.streams);

    }

#ifdef _DEVELOPER
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH))
        cudalog << "Switch time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - m_workSwitchStart)
                       .count()
                << " ms.";
#endif
}
