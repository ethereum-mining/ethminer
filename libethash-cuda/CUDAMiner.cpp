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
#include "CUDAMiner.h"
#include "CUDAMiner_kernel.h"
#include <nvrtc.h>

#include <ethash/ethash.hpp>

#include "CUDAMiner.h"

using namespace std;
using namespace dev;
using namespace eth;

unsigned CUDAMiner::s_numInstances = 0;

vector<int> CUDAMiner::s_devices(MAX_MINERS, -1);

struct CUDAChannel : public LogChannel
{
    static const char* name() { return EthOrange "cu"; }
    static const int verbosity = 2;
};
#define cudalog clog(CUDAChannel)

CUDAMiner::CUDAMiner(unsigned _index) : Miner("cuda-", _index), m_light(getNumDevices()) {}

CUDAMiner::~CUDAMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cuda-" << m_index << " CUDAMiner::~CUDAMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(cudalog, "cuda-" << m_index << " CUDAMiner::~CUDAMiner() end");
}

unsigned CUDAMiner::m_parallelHash = 4;
unsigned CUDAMiner::s_blockSize = CUDAMiner::c_defaultBlockSize;
unsigned CUDAMiner::s_gridSize = CUDAMiner::c_defaultGridSize;
unsigned CUDAMiner::s_numStreams = CUDAMiner::c_defaultNumStreams;
unsigned CUDAMiner::s_scheduleFlag = 0;

bool CUDAMiner::init(int epoch)
{
    try
    {
        if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
            while (s_dagLoadIndex < Index())
                this_thread::sleep_for(chrono::milliseconds(100));
        unsigned device = s_devices[Index()] > -1 ? s_devices[Index()] : Index();

        cnote << "Initialising miner " << Index();

        auto numDevices = getNumDevices();
        if (numDevices == 0)
            return false;

        // use selected device
        m_device_num = device < numDevices - 1 ? device : numDevices - 1;
        m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
        m_hwmoninfo.indexSource = HwMonitorIndexSource::CUDA;
        m_hwmoninfo.deviceIndex = m_device_num;

        cudaDeviceProp device_props;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));

        cudalog << "Using device: " << device_props.name
                << " (Compute " + to_string(device_props.major) + "." +
                       to_string(device_props.minor) + ")";

        const auto& context = ethash::get_global_epoch_context(epoch);
        const auto lightWords = context.light_cache_num_items;
        const auto lightSize = ethash::get_light_cache_size(lightWords);
        const auto dagElms = context.full_dataset_num_items;
        const auto dagBytes = ethash::get_full_dataset_size(dagElms);

        CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
        cudalog << "Set Device to current";
        if (dagElms != m_dag_elms || !m_dag)
        {
            // Check whether the current device has sufficient memory every time we recreate the dag
            if (device_props.totalGlobalMem < dagBytes)
            {
                cudalog << "CUDA device " << string(device_props.name)
                        << " has insufficient GPU memory. "
                        << FormattedMemSize(device_props.totalGlobalMem) << " of memory found, "
                        << FormattedMemSize(dagBytes) << " of memory required";
                return false;
            }
            // We need to reset the device and recreate the dag
            cudalog << "Resetting device";
            CUDA_SAFE_CALL(cudaDeviceReset());
            CUdevice device;
            CUcontext context;
            cuDeviceGet(&device, m_device_num);
            cuCtxCreate(&context, s_scheduleFlag, device);
            // We need to reset the light and the Dag for the following code to reallocate
            // since cudaDeviceReset() frees all previous allocated memory
            m_light[m_device_num] = nullptr;
            m_dag = nullptr;
        }
        // create buffer for cache
        hash64_t * dag = m_dag;
        hash64_t* light = m_light[m_device_num];

        compileKernel(_light->block_number, dagElms);

        if (!light)
        {
            cudalog << "Allocating light with size: " << FormattedMemSize(lightSize);
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), lightSize));
        }
        // copy lightData to device
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), context.light_cache, lightSize,
            cudaMemcpyHostToDevice));
        m_light[m_device_num] = light;

        if (dagElms != m_dag_elms || !dag)  // create buffer for dag
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagBytes));

        if (dagElms != m_dag_elms || !dag)
        {
            // create mining buffers
            cudalog << "Generating mining buffers";
            for (unsigned i = 0; i != s_numStreams; ++i)
            {
                CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(Search_results)));
                CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
            }

            m_current_target = 0;

            if (!s_dagInHostMemory)
            {
                if ((m_device_num == s_dagCreateDevice) || (s_dagLoadMode != DAG_LOAD_MODE_SINGLE))
                {
                    cudalog << "Generating DAG for GPU #" << m_device_num
                            << " with dagBytes: " << FormattedMemSize(dagBytes) << " ("
                            << FormattedMemSize(device_props.totalGlobalMem - dagBytes - lightSize)
                            << " left)";
                    auto startDAG = std::chrono::steady_clock::now();

                    ethash_generate_dag(dag, dagBytes, light, lightWords, s_gridSize, s_blockSize, m_streams[0], m_device_num);

                    cudalog << "Generated DAG for GPU" << m_device_num << " in: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startDAG)
                                   .count()
                            << " ms.";

                    if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
                    {
                        uint8_t* memoryDAG = new uint8_t[dagBytes];
                        cudalog << "Copying DAG from GPU #" << m_device_num << " to host";
                        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagBytes,
                            cudaMemcpyDeviceToHost));

                        s_dagInHostMemory = memoryDAG;
                    }
                }
                else
                {
                    while (!s_dagInHostMemory)
                        this_thread::sleep_for(chrono::milliseconds(100));
                    goto cpyDag;
                }
            }
            else
            {
            cpyDag:
                cudalog << "Copying DAG from host to GPU #" << m_device_num;
                const void* hdag = (const void*)s_dagInHostMemory;
                CUDA_SAFE_CALL(cudaMemcpy(
                    reinterpret_cast<void*>(dag), hdag, dagBytes, cudaMemcpyHostToDevice));
            }
        }

        m_dag = dag;
        m_dag_elms = dagElms;

        s_dagLoadIndex++;

        if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
        {
            if (s_dagLoadIndex >= s_numInstances && s_dagInHostMemory)
            {
                // all devices have loaded DAG, we can free now
                delete[] s_dagInHostMemory;
                s_dagInHostMemory = nullptr;
                cnote << "Freeing DAG from host";
            }
        }
    }
    catch (cuda_runtime_error const& _e)
    {
        cwarn << "Fatal GPU error: " << _e.what();
        cwarn << "Terminating.";
        exit(-1);
    }
    catch (std::runtime_error const& _e)
    {
        cwarn << "Error CUDA mining: " << _e.what();
        if (s_exit)
            exit(1);
        return false;
    }
    return true;
}

void CUDAMiner::workLoop()
{
    WorkPackage current;
    current.header = h256{1u};

    m_search_buf.resize(s_numStreams);
    m_streams.resize(s_numStreams);


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

            // take local copy of work since it may end up being overwritten.
            const WorkPackage w = work();

            // Take actions in proper order

            // No work ?
            if (!w || w.header == h256())
            {
                cnote << "No work. Pause for 3 s.";
                std::this_thread::sleep_for(std::chrono::seconds(3));
                continue;
            }
            // Epoch change ?
            else if (current.epoch != w.epoch)
            {
                if (!init(w.epoch))
                    break;
            }

            // Persist most recent job anyway. No need to do another
            // conditional check if they're different
            current = w;

            uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);

            // Eventually start searching
            search(current.header.data(), upper64OfBoundary, current.startNonce, w);
        }

        // Reset miner and stop working
        CUDA_SAFE_CALL(cudaDeviceReset());
    }
    catch (cuda_runtime_error const& _e)
    {
        cwarn << "GPU error: " << _e.what();
        if (s_exit)
        {
            cwarn << "Terminating.";
            exit(1);
        }
    }
}

void CUDAMiner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
}

void CUDAMiner::setNumInstances(unsigned _instances)
{
    s_numInstances = std::min<unsigned>(_instances, getNumDevices());
}

void CUDAMiner::setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount)
{
    for (unsigned i = 0; i < _selectedDeviceCount; i++)
        s_devices[i] = _devices[i];
}

unsigned CUDAMiner::getNumDevices()
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
            throw std::runtime_error{"No CUDA driver found"};
        throw std::runtime_error{"Insufficient CUDA driver: " + std::to_string(driverVersion)};
    }

    throw std::runtime_error{cudaGetErrorString(err)};
}

void CUDAMiner::listDevices()
{
    try
    {
        cout << "\nListing CUDA devices.\nFORMAT: [deviceID] deviceName\n";
        int numDevices = getNumDevices();
        for (int i = 0; i < numDevices; ++i)
        {
            cudaDeviceProp props;
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

            cout << "[" + to_string(i) + "] " + string(props.name) + "\n";
            cout << "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) +
                        "\n";
            cout << "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
            cout << "\tPci: " << setw(4) << setfill('0') << hex << props.pciDomainID << ':'
                 << setw(2) << props.pciBusID << ':' << setw(2) << props.pciDeviceID << '\n';
        }
    }
    catch (std::runtime_error const& err)
    {
        cwarn << "CUDA error: " << err.what();
        if (s_exit)
            exit(1);
    }
}

unsigned const CUDAMiner::c_defaultBlockSize = 512;
unsigned const CUDAMiner::c_defaultGridSize = 1024; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const CUDAMiner::c_defaultNumStreams = 2;

bool CUDAMiner::configureGPU(unsigned _blockSize, unsigned _gridSize, unsigned _numStreams,
    unsigned _scheduleFlag, unsigned _dagLoadMode, unsigned _dagCreateDevice, bool _noeval,
    bool _exit)
{
    s_dagLoadMode = _dagLoadMode;
    s_dagCreateDevice = _dagCreateDevice;
    s_exit = _exit;
    s_blockSize = _blockSize;
    s_gridSize = _gridSize;
    s_numStreams = _numStreams;
    s_scheduleFlag = _scheduleFlag;
    s_noeval = _noeval;

    try
    {

        cudalog << "Using grid size: " << s_gridSize << ", block size: " << s_blockSize;

        // by default let's only consider the DAG of the first epoch
        const auto dagSize =
            ethash::get_full_dataset_size(ethash::calculate_full_dataset_num_items(0));
        int devicesCount = static_cast<int>(getNumDevices());
        for (int i = 0; i < devicesCount; i++)
        {
            if (s_devices[i] != -1)
            {
                int deviceId = min(devicesCount - 1, s_devices[i]);
                cudaDeviceProp props;
                CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
                if (props.totalGlobalMem >= dagSize)
                {
                    cudalog << "Found suitable CUDA device [" << string(props.name) << "] with "
                            << props.totalGlobalMem << " bytes of GPU memory";
                }
                else
                {
                    cudalog << "CUDA device " << string(props.name)
                            << " has insufficient GPU memory. " << props.totalGlobalMem
                            << " bytes of memory found < " << dagSize
                            << " bytes of memory required";
                    return false;
                }
            }
        }
        return true;
    }
    catch (cuda_runtime_error const& _e)
    {
        cwarn << "Fatal GPU error: " << _e.what();
        cwarn << "Terminating.";
        exit(-1);
    }
    catch (std::runtime_error const& _e)
    {
        cwarn << "Error CUDA mining: " << _e.what();
        if (s_exit)
            exit(1);
        return false;
    }

}

void CUDAMiner::setParallelHash(unsigned _parallelHash)
{
    m_parallelHash = _parallelHash;
}

#include <iostream>
#include <fstream>

void CUDAMiner::compileKernel(
    uint64_t block_number,
    uint64_t dag_elms)
{
    const char* name = "progpow_search";

    std::string text = ProgPow::getKern(block_number, ProgPow::KERNEL_CUDA);
    text += std::string(CUDAMiner_kernel, sizeof(CUDAMiner_kernel));

    ofstream write;
    write.open("kernel.cu");
    write << text;
    write.close();

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(
            &prog,         // prog
            text.c_str(),  // buffer
            "kernel.cu",    // name
            0,             // numHeaders
            NULL,          // headers
            NULL));        // includeNames

    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, name));
    cudaDeviceProp device_props;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));
    std::string op_arch = "--gpu-architecture=compute_" + to_string(device_props.major) + to_string(device_props.minor);
    std::string op_dag = "-DPROGPOW_DAG_ELEMENTS=" + to_string(dag_elms);

    const char *opts[] = {
        op_arch.c_str(),
        op_dag.c_str(),
        "-lineinfo"
    };
    nvrtcResult compileResult = nvrtcCompileProgram(
        prog,  // prog
        3,     // numOptions
        opts); // options
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    cudalog << "Compile log: " << log;
    delete[] log;
    NVRTC_SAFE_CALL(compileResult);
    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    write.open("kernel.ptx");
    write << ptx;
    write.close();
    // Load the generated PTX and get a handle to the kernel.
    char *jitInfo = new char[32 * 1024];
    char *jitErr = new char[32 * 1024];
    CUjit_option jitOpt[] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE,
        CU_JIT_GENERATE_LINE_INFO
    };
    void *jitOptVal[] = {
        jitInfo,
        jitErr,
        (void*)(32 * 1024),
        (void*)(32 * 1024),
        (void*)(1),
        (void*)(1)
    };
    CU_SAFE_CALL(cuModuleLoadDataEx(&m_module, ptx, 6, jitOpt, jitOptVal));
    cudalog << "JIT info: \n" << jitInfo;
    cudalog << "JIT err: \n" << jitErr;
    delete[] ptx;
    delete[] jitInfo;
    delete[] jitErr;
    // Find the mangled name
    const char* mangledName;
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, name, &mangledName));
    cudalog << "Mangled name: " << mangledName;
    CU_SAFE_CALL(cuModuleGetFunction(&m_kernel, m_module, mangledName));
    cudalog << "done compiling";
    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
}

void CUDAMiner::search(
    uint8_t const* header, uint64_t target, uint64_t current_nonce, const dev::eth::WorkPackage& w)
{
    uint64_t stream_nonce;
    if (m_current_target != target)
    {
        m_current_target = target;
    }

    hash32_t current_header = *reinterpret_cast<hash32_t const *>(header);

    // Nonces processed in one pass by a single stream
    const uint32_t batch_size = s_gridSize * s_blockSize;
    // Nonces processed in one pass by all streams
    const uint32_t streams_batch_size = batch_size * s_numStreams;

    // prime each stream, clear search result buffers and start the search
    uint32_t current_index;
    for (current_index = 0, stream_nonce = current_nonce;
         current_index < s_numStreams;
         current_index++, stream_nonce += batch_size)
    {
        cudaStream_t stream = m_streams[current_index];
        volatile Search_results& buffer(*m_search_buf[current_index]);
        buffer.count = 0;

        // Run the batch for this stream
        bool hack_false = false;
        void *args[] = {&current_nonce, &current_header, &m_current_target, &m_dag, &buffer, &hack_false};
        CU_SAFE_CALL(cuLaunchKernel(m_kernel,
            s_gridSize, 1, 1,   // grid dim
            s_blockSize, 1, 1,  // block dim
            0,                  // shared mem
            stream,             // stream
            args, 0));          // arguments
    }

    // process stream batches until we get new work.
    bool done = false;
    while (!done)
    {
        // Each pass of this outer loop will handle all cuda streams once

        // Exit next time around if there's new work awaiting
        bool t = true;
        if (m_new_work.compare_exchange_strong(t, false))
            done = true;

        Search_results save_buf;
#ifdef DEV_BUILD
        std::chrono::steady_clock::time_point submitStart;
#endif
        // This inner loop will process each cuda stream individually
        for (current_index = 0, stream_nonce = current_nonce;
             current_index < s_numStreams;
             current_index++, stream_nonce += batch_size)
        {
            // Each pass of this loop will wait for a stream to exit,
            // save any found solutions, then restart the stream
            // on the next group of nonces.
            cudaStream_t stream = m_streams[current_index];

            // Wait for the stream complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            // refer to the current stream's solution buffer
            volatile Search_results& buffer(*m_search_buf[current_index]);
            // See if the stream found any solutions
            uint32_t found_count = buffer.count;
            if (found_count)
            {
#ifdef DEV_BUILD
                submitStart = std::chrono::steady_clock::now();
#endif
                // Found one or more solutions, save them for later.
                // We are in a hurry to restart the stream, so defer
                // non-critical activities, such as hash rate calculations
                // and solution submission.

                if (found_count > MAX_SEARCH_RESULTS)
                    found_count = MAX_SEARCH_RESULTS;

                // refer to the current stream's save buffer

                // copy the solution to the save buffer for this stream
                save_buf.count = found_count;
                for (uint32_t i = 0; i < found_count; i++)
                {
                    save_buf.result[i].gid = buffer.result[i].gid;
                    if (s_noeval)
                        for (uint32_t j = 0; j < 8; j++)
                            save_buf.result[i].mix[j] = buffer.result[i].mix[j];
                }

                // Ok, solutions are saved. We can now reset the stream's
                // solution buffer, releasing it for reuse.
                buffer.count = 0;
            }

            // restart the stream on the next batch of nonces
            // unless we are done for this round.
            if (!done)
            {
                bool hack_false = false;
                void *args[] = {&current_nonce, &current_header, &m_current_target, &m_dag, &buffer, &hack_false};
                CU_SAFE_CALL(cuLaunchKernel(m_kernel,
                    s_gridSize, 1, 1,   // grid dim
                    s_blockSize, 1, 1,  // block dim
                    0,                  // shared mem
                    stream,             // stream
                    args, 0));          // arguments
            }

            found_count = save_buf.count;
            if (found_count)
            {
                // Submit this stream's solutions
                for (uint32_t i = 0; i < found_count; i++)
                {
                    uint64_t nonce = stream_nonce + save_buf.result[i].gid;
                    if (s_noeval)
                    {
                        // noeval... use the GPU calculated mix hash.
                        h256 mix;
                        memcpy(mix.data(), (void*)&save_buf.result[i].mix,
                            sizeof(save_buf.result[0].mix));
                        Farm::f().submitProof(Solution{nonce, mix, w, done, Index()});
                    }
                    else
                    {
                        // eval... recalculate the mix hash in software and verify
                        // the GPU's result
                        Result r = EthashAux::eval(w.epoch, w.header, nonce);
                        if (r.value <= w.boundary)
                        {
                            Farm::f().submitProof(Solution{nonce, r.mixHash, w, done, Index()});
                        }
                        else
                        {
                            Farm::f().failedSolution(Index());
                            cwarn
                                << "GPU gave incorrect result! Lower OC if this happens frequently";
                        }
                    }
                }
                // Reset this stream's buffer for the next pass
                save_buf.count = 0;
#ifdef DEV_BUILD
                if (g_logOptions & LOG_SUBMIT)
                    cudalog << "Submit time: "
                            << std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - submitStart)
                                   .count()
                            << " us.";
#endif
            }
        }

        // All streams have been restarted and should be busy for a while
        // searching. Now is a good time to burn CPU cycles on bookeeping.

        // Update the hash rate
        updateHashRate(batch_size, s_numStreams);

        // Bail out if it's shutdown time
        if (shouldStop())
        {
            m_new_work.store(false, std::memory_order_relaxed);
            break;
        }

        current_nonce += streams_batch_size;
    }

#ifdef DEV_BUILD
    // Optionally log job switch time
    if (!shouldStop() && (g_logOptions & LOG_SWITCH))
        cudalog << "Switch time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - m_workSwitchStart)
                       .count()
                << " ms.";
#endif
}
