/*
This file is part of cpp-ethereum.

cpp-ethereum is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cpp-ethereum is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/

#undef min
#undef max

#include "ProgPowMiner.h"
#include "CUDAMiner_kernel.h"
#include "../dag_generation_kernel.h"
#include "../dag_globals.h"
#include <libethcore/Farm.h>
#include <nvrtc.h>


using namespace std;
using namespace dev;
using namespace eth;

unsigned ProgPowMiner::s_numInstances = 0;

vector<int> ProgPowMiner::s_devices(MAX_MINERS, -1);

struct CUDAChannel: public LogChannel
{
	static const char* name() { return EthOrange " cu"; }
	static const int verbosity = 2;
	static const bool debug = false;
};

struct CUDASwitchChannel: public LogChannel
{
	static const char* name() { return EthOrange " cu"; }
	static const int verbosity = 6;
	static const bool debug = false;
};

#define cudalog clog(CUDAChannel)
#define cudaswitchlog clog(CUDASwitchChannel)

ProgPowMiner::ProgPowMiner(unsigned _index) :
	Miner("cuda-", _index),
	m_light(getNumDevices()) {}

ProgPowMiner::~ProgPowMiner()
{
	stopWorking();
	kick_miner();
}

bool ProgPowMiner::init(int epoch)
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
        const auto lightNumItems = context.light_cache_num_items;
        const auto lightSize = ethash::get_light_cache_size(lightNumItems);
        const auto dagNumItems = context.full_dataset_num_items;
        const auto dagSize = ethash::get_full_dataset_size(dagNumItems);

        CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
        cudalog << "Set Device to current";
        if (dagNumItems != m_dag_elms || !m_dag)
        {
            // Check whether the current device has sufficient memory every time we recreate the dag
            if (device_props.totalGlobalMem < dagSize)
            {
                cudalog << "CUDA device " << string(device_props.name)
                        << " has insufficient GPU memory. "
                        << FormattedMemSize(device_props.totalGlobalMem) << " of memory found, "
                        << FormattedMemSize(dagSize) << " of memory required";
                return false;
            }
            // We need to reset the device and recreate the dag
            cudalog << "Resetting device";
            CUDA_SAFE_CALL(cudaDeviceReset());
            CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
            CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
            // We need to reset the light and the Dag for the following code to reallocate
            // since cudaDeviceReset() frees all previous allocated memory
            m_light[m_device_num] = nullptr;
            m_dag = nullptr;
        }
        // create buffer for cache
        hash64_t* dag = m_dag;
        hash64_t* light = m_light[m_device_num];

        if (!light)
        {
            cudalog << "Allocating light with size: " << FormattedMemSize(lightSize);
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), lightSize));
        }
        // copy lightData to device
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), context.light_cache, lightSize,
            cudaMemcpyHostToDevice));
        m_light[m_device_num] = light;

        if (dagNumItems != m_dag_elms || !dag)  // create buffer for dag
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));

        set_constants(dag, dagNumItems, light, lightNumItems);

        if (dagNumItems != m_dag_elms || !dag)
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
                            << " with dagSize: " << FormattedMemSize(dagSize) << " ("
                            << FormattedMemSize(device_props.totalGlobalMem - dagSize - lightSize)
                            << " left)";
                    auto startDAG = std::chrono::steady_clock::now();

                    ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0]);

                    cudalog << "Generated DAG for GPU" << m_device_num << " in: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startDAG)
                                   .count()
                            << " ms.";

                    if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
                    {
                        uint8_t* memoryDAG = new uint8_t[dagSize];
                        cudalog << "Copying DAG from GPU #" << m_device_num << " to host";
                        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagSize,
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
                    reinterpret_cast<void*>(dag), hdag, dagSize, cudaMemcpyHostToDevice));
            }
        }

        m_dag = dag;
        m_dag_elms = dagNumItems;

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
    catch (std::runtime_error const& _e)
    {
        cwarn << "Error CUDA mining: " << _e.what();
        if (s_exit)
            exit(1);
        return false;
    }
    return true;
}

void ProgPowMiner::workLoop()
{
	WorkPackage current;
	current.header = h256{1u};
	uint64_t old_period_seed = -1;

	try
	{
		while(!shouldStop())
		{
	                // take local copy of work since it may end up being overwritten.
			const WorkPackage w = work();
			uint64_t period_seed = w.height / PROGPOW_PERIOD;

			if (current.header != w.header || current.epoch != w.epoch || old_period_seed != period_seed)
			{
				if(!w || w.header == h256())
				{
					cnote << "No work.";
					//std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}
				if (current.epoch != w.epoch)
					if(!init(w.epoch))
						break;
				if (old_period_seed != period_seed)
				{
					//uint64_t dagBytes = ethash_get_datasize(w.height);
					//uint32_t dagElms   = (unsigned)(dagBytes / ETHASH_MIX_BYTES);
					uint64_t dagBytes = ethash::get_full_dataset_size(w.height);
					uint32_t dagElms   = (unsigned)(dagBytes / 256);
					compileKernel(w.height, dagElms);
				}
				old_period_seed = period_seed;
				current = w;
			}
			uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);
			uint64_t startN = current.startNonce;
			if (current.exSizeBits >= 0)
			{
				// this can support up to 2^c_log2Max_miners devices
				startN = current.startNonce | ((uint64_t)m_index << (64 - LOG2_MAX_MINERS - current.exSizeBits));
			}
			search(current.header.data(), upper64OfBoundary, startN, w);
		}

		// Reset miner and stop working
		CUDA_SAFE_CALL(cudaDeviceReset());
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
		if(s_exit)
			exit(1);
	}
}

void ProgPowMiner::kick_miner()
{
	m_new_work.store(true, std::memory_order_relaxed);
}

void ProgPowMiner::setNumInstances(unsigned _instances)
{
        s_numInstances = std::min<unsigned>(_instances, getNumDevices());
}

unsigned ProgPowMiner::instances()
{
	return s_numInstances > 0 ? s_numInstances : 1;
}

void ProgPowMiner::setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount)
{
        for (unsigned i = 0; i < _selectedDeviceCount; i++)
                s_devices[i] = _devices[i];
}

unsigned ProgPowMiner::getNumDevices()
{
	int deviceCount = -1;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err == cudaSuccess)
		return deviceCount;

	if (err == cudaErrorInsufficientDriver)
	{
		int driverVersion = -1;
		cudaDriverGetVersion(&driverVersion);
		if (driverVersion == 0)
			throw std::runtime_error{"No CUDA driver found"};
		throw std::runtime_error{"Insufficient CUDA driver: " + std::to_string(driverVersion)};
	}

	throw std::runtime_error{cudaGetErrorString(err)};
}

void ProgPowMiner::listDevices()
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
			cout << "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) + "\n";
			cout << "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
			cout << "\tPci: " << setw(4) << setfill('0') << hex << props.pciDomainID << ':' << setw(2)
				<< props.pciBusID << ':' << setw(2) << props.pciDeviceID << '\n';
		}
	}
	catch(std::runtime_error const& err)
	{
		cwarn << "CUDA error: " << err.what();
		if(s_exit)
			exit(1);
	}
}

bool ProgPowMiner::configureGPU(
	unsigned _blockSize,
	unsigned _gridSize,
	unsigned _numStreams,
	unsigned _scheduleFlag,
	uint64_t _currentBlock,
	unsigned _dagLoadMode,
	unsigned _dagCreateDevice,
	bool _noeval,
	bool _exit
	)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;
	s_exit  = _exit;

	if (!cuda_configureGPU(
		getNumDevices(),
		s_devices,
		((_blockSize + 7) / 8) * 8,
		_gridSize,
		_numStreams,
		_scheduleFlag,
		_currentBlock,
		_noeval)
		)
	{
		cout << "No CUDA device with sufficient memory was found. Can't CUDA mine. Remove the -U argument" << endl;
		return false;
	}
	return true;
}

void ProgPowMiner::setParallelHash(unsigned _parallelHash)
{
  	m_parallelHash = _parallelHash;
}

unsigned const ProgPowMiner::c_defaultBlockSize = 512;
unsigned const ProgPowMiner::c_defaultGridSize = 1024; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const ProgPowMiner::c_defaultNumStreams = 2;

bool ProgPowMiner::cuda_configureGPU(
	size_t numDevices,
	const vector<int>& _devices,
	unsigned _blockSize,
	unsigned _gridSize,
	unsigned _numStreams,
	unsigned _scheduleFlag,
	uint64_t _currentBlock,
	bool _noeval
	)
{
	try
	{
		s_blockSize = _blockSize;
		s_gridSize = _gridSize;
		s_numStreams = _numStreams;
		s_scheduleFlag = _scheduleFlag;
		s_noeval = _noeval;

		cudalog << "Using grid size " << s_gridSize << ", block size " << s_blockSize;

		uint64_t dagSize = ethash::get_full_dataset_size(_currentBlock);
		int devicesCount = static_cast<int>(numDevices);
		for (int i = 0; i < devicesCount; i++)
		{
			if (_devices[i] != -1)
			{
				int deviceId = min(devicesCount - 1, _devices[i]);
				cudaDeviceProp props;
				CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, deviceId));
				if (props.totalGlobalMem >= dagSize)
				{
					cudalog <<  "Found suitable CUDA device [" << string(props.name) << "] with " << props.totalGlobalMem << " bytes of GPU memory";
				}
				else
				{
					cudalog <<  "CUDA device " << string(props.name) << " has insufficient GPU memory." << props.totalGlobalMem << " bytes of memory found < " << dagSize << " bytes of memory required";
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
		if(s_exit)
			exit(1);
		return false;
	}
}

unsigned ProgPowMiner::m_parallelHash = 4;
unsigned ProgPowMiner::s_blockSize = ProgPowMiner::c_defaultBlockSize;
unsigned ProgPowMiner::s_gridSize = ProgPowMiner::c_defaultGridSize;
unsigned ProgPowMiner::s_numStreams = ProgPowMiner::c_defaultNumStreams;
unsigned ProgPowMiner::s_scheduleFlag = 0;
bool ProgPowMiner::s_noeval = false;

void ProgPowMiner::search(
    uint8_t const* header, uint64_t target, uint64_t start_nonce, const dev::eth::WorkPackage& w)
{
     set_header(*reinterpret_cast<hash32_t const*>(header));
     if (m_current_target != target)
     {
         set_target(target);
        m_current_target = target;
    }
     hash32_t current_header = *reinterpret_cast<hash32_t const *>(header);
    hash64_t* dag;
    get_constants(&dag, NULL, NULL, NULL);
     // prime each stream, clear search result buffers and start the search
	 // Nonces processed in one pass by a single stream
    const uint32_t batch_size = s_gridSize * s_blockSize;
    // Nonces processed in one pass by all streams
    const uint32_t streams_batch_size = batch_size * s_numStreams;
	uint64_t stream_nonce;
    // prime each stream, clear search result buffers and start the search
    uint32_t current_index;
    for (current_index = 0, stream_nonce = m_current_nonce;
         current_index < s_numStreams;
         current_index++, stream_nonce += batch_size)
    {
         cudaStream_t stream = m_streams[current_index];
         volatile Search_results& buffer(*m_search_buf[current_index]);
        buffer.count = 0;
         // Run the batch for this stream
        // run_ethash_search(s_gridSize, s_blockSize, stream, &buffer, start_nonce, m_parallelHash);
        volatile Search_results *Buffer = &buffer;
        bool hack_false = false;
        void *args[] = {&start_nonce, &current_header, &m_current_target, &dag, &Buffer, &hack_false};
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
         // Exit next time around if there's new work awaiting
         bool t = true;
         done = m_new_work.compare_exchange_strong(t, false);
 
         // This inner loop will process each cuda stream individually
         for (current_index = 0; current_index < s_numStreams;
              current_index++, start_nonce += batch_size)
         {
             // Each pass of this loop will wait for a stream to exit,
             // save any found solutions, then restart the stream
             // on the next group of nonces.
             cudaStream_t stream = m_streams[current_index];
 
             // Wait for the stream complete
             CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
 
             if (shouldStop())
             {
                 m_new_work.store(false, std::memory_order_relaxed);
                 done = true;
             }
 
             // Detect solutions in current stream's solution buffer
             volatile Search_results& buffer(*m_search_buf[current_index]);
             uint32_t found_count = std::min((unsigned)buffer.count, MAX_SEARCH_RESULTS);
 
             if (found_count)
             {
                 buffer.count = 0;
                 uint64_t nonce_base = start_nonce - streams_batch_size;
 
                 // Extract solution and pass to higer level
                 // using io_service as dispatcher
 
                 for (uint32_t i = 0; i < found_count; i++)
                 {
                     h256 mix;
                     uint64_t nonce = nonce_base + buffer.result[i].gid;
                     memcpy(mix.data(), (void*)&buffer.result[i].mix, sizeof(buffer.result[i].mix));
                     Farm::f().submitProof(Solution{nonce, mix, w, done, Index()});
                 }
             }
 
            // restart the stream on the next batch of nonces
            // unless we are done for this round.
            if (!done)
                //run_ethash_search(
                    //m_settings.gridSize, m_settings.blockSize, stream, &buffer, start_nonce, m_settings.parallelHash);
            {
                volatile Search_results *Buffer = &buffer;
                bool hack_false = false;
                void *args[] = {&start_nonce, &current_header, &m_current_target, &dag, &Buffer, &hack_false};
                CU_SAFE_CALL(cuLaunchKernel(m_kernel,
                    s_gridSize, 1, 1,   // grid dim
                    s_blockSize, 1, 1,  // block dim
                    0,                  // shared mem
                    stream,             // stream
                    args, 0));          // arguments
            }
        }
         // Update the hash rate
         updateHashRate(batch_size, s_numStreams);
 
         // Bail out if it's shutdown time
         if (shouldStop())
         {
             m_new_work.store(false, std::memory_order_relaxed);
             break;
         }
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