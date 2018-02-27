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

#include "CUDAMiner.h"

using namespace std;
using namespace dev;
using namespace eth;

unsigned CUDAMiner::s_numInstances = 0;

vector<int> CUDAMiner::s_devices(MAX_MINERS, -1);

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

CUDAMiner::CUDAMiner(FarmFace& _farm, unsigned _index) :
	Miner("cuda-", _farm, _index),
	m_light(getNumDevices()) {}

CUDAMiner::~CUDAMiner()
{
	stopWorking();
	kick_miner();
}

bool CUDAMiner::init(const h256& seed)
{
	try {
		if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
			while (s_dagLoadIndex < index)
				this_thread::sleep_for(chrono::milliseconds(100));
		unsigned device = s_devices[index] > -1 ? s_devices[index] : index;

		cnote << "Initialising miner " << index;

		EthashAux::LightType light;
		light = EthashAux::light(seed);
		bytesConstRef lightData = light->data();

		cuda_init(getNumDevices(), light->light, lightData.data(), lightData.size(), 
			device, (s_dagLoadMode == DAG_LOAD_MODE_SINGLE), s_dagInHostMemory, s_dagCreateDevice);
		s_dagLoadIndex++;
    
		if (s_dagLoadMode == DAG_LOAD_MODE_SINGLE)
		{
			if (s_dagLoadIndex >= s_numInstances && s_dagInHostMemory)
			{
				// all devices have loaded DAG, we can free now
				delete[] s_dagInHostMemory;
				s_dagInHostMemory = NULL;
				cnote << "Freeing DAG from host";
			}
		}
		return true;
	}
	catch (std::runtime_error const& _e)
	{
		cwarn << "Error CUDA mining: " << _e.what();
		if(s_exit)
			exit(1);
		return false;
	}
}

void CUDAMiner::workLoop()
{
	WorkPackage current;
	current.header = h256{1u};
	current.seed = h256{1u};

	try
	{
		while(true)
		{
	                // take local copy of work since it may end up being overwritten.
			const WorkPackage w = work();
			
			if (current.header != w.header || current.seed != w.seed)
			{
				if(!w || w.header == h256())
				{
					cnote << "No work. Pause for 3 s.";
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}
				if (current.seed != w.seed)
					if(!init(w.seed))
						break;
				current = w;
			}
			uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);
			uint64_t startN = current.startNonce;
			if (current.exSizeBits >= 0)
			{
				// this can support up to 2^c_log2Max_miners devices
				startN = current.startNonce | ((uint64_t)index << (64 - LOG2_MAX_MINERS - current.exSizeBits));
			}
			search(current.header.data(), upper64OfBoundary, (current.exSizeBits >= 0), startN, w);

			// Check if we should stop.
			if (shouldStop())
				break;
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
		if(s_exit)
			exit(1);
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

bool CUDAMiner::configureGPU(
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

void CUDAMiner::setParallelHash(unsigned _parallelHash)
{
  	m_parallelHash = _parallelHash;
}

unsigned const CUDAMiner::c_defaultBlockSize = 128;
unsigned const CUDAMiner::c_defaultGridSize = 8192; // * CL_DEFAULT_LOCAL_WORK_SIZE
unsigned const CUDAMiner::c_defaultNumStreams = 2;

bool CUDAMiner::cuda_configureGPU(
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

		// by default let's only consider the DAG of the first epoch
		uint64_t dagSize = ethash_get_datasize(_currentBlock);
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
	catch (runtime_error)
	{
		if(s_exit)
			exit(1);
		return false;
	}
}

unsigned CUDAMiner::m_parallelHash = 4;
unsigned CUDAMiner::s_blockSize = CUDAMiner::c_defaultBlockSize;
unsigned CUDAMiner::s_gridSize = CUDAMiner::c_defaultGridSize;
unsigned CUDAMiner::s_numStreams = CUDAMiner::c_defaultNumStreams;
unsigned CUDAMiner::s_scheduleFlag = 0;
bool CUDAMiner::s_noeval = false;

bool CUDAMiner::cuda_init(
	size_t numDevices,
	ethash_light_t _light,
	uint8_t const* _lightData,
	uint64_t _lightSize,
	unsigned _deviceId,
	bool _cpyToHost,
	uint8_t* &hostDAG,
	unsigned dagCreateDevice)
{
	try
	{
		if (numDevices == 0)
			return false;

		// use selected device
		m_device_num = _deviceId < numDevices -1 ? _deviceId : numDevices - 1;
		m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
		m_hwmoninfo.deviceIndex = m_device_num;

		cudaDeviceProp device_props;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_props, m_device_num));

		cudalog << "Using device: " << device_props.name << " (Compute " + to_string(device_props.major) + "." + to_string(device_props.minor) + ")";

		m_search_buf = new volatile search_results *[s_numStreams];
		m_streams = new cudaStream_t[s_numStreams];

		uint64_t dagSize = ethash_get_datasize(_light->block_number);
		uint32_t dagSize128   = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(_lightSize / sizeof(node));

		
		
		CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
		cudalog << "Set Device to current";
		if(dagSize128 != m_dag_size || !m_dag)
		{
			//Check whether the current device has sufficient memory everytime we recreate the dag
			if (device_props.totalGlobalMem < dagSize)
			{
				cudalog <<  "CUDA device " << string(device_props.name) << " has insufficient GPU memory." << device_props.totalGlobalMem << " bytes of memory found < " << dagSize << " bytes of memory required";
				return false;
			}
			//We need to reset the device and recreate the dag  
			cudalog << "Resetting device";
			CUDA_SAFE_CALL(cudaDeviceReset());
			CUDA_SAFE_CALL(cudaSetDeviceFlags(s_scheduleFlag));
			CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
			//We need to reset the light and the Dag for the following code to reallocate
			//since cudaDeviceReset() free's all previous allocated memory
			m_light[m_device_num] = nullptr;
			m_dag = nullptr; 
		}
		// create buffer for cache
		hash128_t * dag = m_dag;
		hash64_t * light = m_light[m_device_num];

		if(!light){ 
			cudalog << "Allocating light with size: " << _lightSize;
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), _lightSize));
		}
		// copy lightData to device
		CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), _lightData, _lightSize, cudaMemcpyHostToDevice));
		m_light[m_device_num] = light;
		
		if(dagSize128 != m_dag_size || !dag) // create buffer for dag
			CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagSize));
			
		set_constants(dag, dagSize128, light, lightSize64); //in ethash_cuda_miner_kernel.cu
		
		if(dagSize128 != m_dag_size || !dag)
		{
			// create mining buffers
			cudalog << "Generating mining buffers"; //TODO whats up with this?
			for (unsigned i = 0; i != s_numStreams; ++i)
			{
				CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(search_results)));
				CUDA_SAFE_CALL(cudaStreamCreate(&m_streams[i]));
			}
			
			memset(&m_current_header, 0, sizeof(hash32_t));
			m_current_target = 0;
			m_current_nonce = 0;
			m_current_index = 0;

			if (!hostDAG)
			{
				if((m_device_num == dagCreateDevice) || !_cpyToHost){ //if !cpyToHost -> All devices shall generate their DAG
					cudalog << "Generating DAG for GPU #" << m_device_num << " with dagSize: " 
							<< dagSize <<" gridSize: " << s_gridSize;
					ethash_generate_dag(dagSize, s_gridSize, s_blockSize, m_streams[0], m_device_num);

					if (_cpyToHost)
					{
						uint8_t* memoryDAG = new uint8_t[dagSize];
						cudalog << "Copying DAG from GPU #" << m_device_num << " to host";
						CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(memoryDAG), dag, dagSize, cudaMemcpyDeviceToHost));

						hostDAG = memoryDAG;
					}
				}else{
					while(!hostDAG)
						this_thread::sleep_for(chrono::milliseconds(100)); 
					goto cpyDag;
				}
			}
			else
			{
cpyDag:
				cudalog << "Copying DAG from host to GPU #" << m_device_num;
				const void* hdag = (const void*)hostDAG;
				CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(dag), hdag, dagSize, cudaMemcpyHostToDevice));
			}
		}
    
		m_dag = dag;
		m_dag_size = dagSize128;
		return true;
	}
	catch (runtime_error const&)
	{
		if(s_exit)
			exit(1);
		return false;
	}
}

void CUDAMiner::search(
	uint8_t const* header,
	uint64_t target,
	bool _ethStratum,
	uint64_t _startN,
	const dev::eth::WorkPackage& w)
{
	bool initialize = false;
	if (memcmp(&m_current_header, header, sizeof(hash32_t)))
	{
		m_current_header = *reinterpret_cast<hash32_t const *>(header);
		set_header(m_current_header);
		initialize = true;
	}
	if (m_current_target != target)
	{
		m_current_target = target;
		set_target(m_current_target);
		initialize = true;
	}
	if (_ethStratum)
	{
		if (initialize)
		{
			m_starting_nonce = 0;
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i]->count = 0;
		}
		if (m_starting_nonce != _startN)
		{
			// reset nonce counter
			m_starting_nonce = _startN;
			m_current_nonce = m_starting_nonce;
		}
	}
	else
	{
		if (initialize)
		{
			m_current_nonce = get_start_nonce();
			m_current_index = 0;
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			for (unsigned int i = 0; i < s_numStreams; i++)
				m_search_buf[i]->count = 0;
		}
	}
	const uint32_t batch_size = s_gridSize * s_blockSize;
	while (true)
	{
		m_current_index++;
		m_current_nonce += batch_size;
		auto stream_index = m_current_index % s_numStreams;
		cudaStream_t stream = m_streams[stream_index];
		volatile search_results* buffer = m_search_buf[stream_index];
		uint32_t found_count = 0;
		uint64_t nonces[SEARCH_RESULTS];
		uint32_t mixes[SEARCH_RESULTS][8];
		uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;
		if (m_current_index >= s_numStreams)
		{
			CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
			found_count = buffer->count;
			if (found_count) {
				buffer->count = 0;
				if (found_count > SEARCH_RESULTS)
					found_count = SEARCH_RESULTS;
				for (unsigned int j = 0; j < found_count; j++) {
					nonces[j] = nonce_base + buffer->result[j].gid;
					if (s_noeval) {
						mixes[j][0] = buffer->result[j].mix[0];
						mixes[j][1] = buffer->result[j].mix[1];
						mixes[j][2] = buffer->result[j].mix[2];
						mixes[j][3] = buffer->result[j].mix[3];
						mixes[j][4] = buffer->result[j].mix[4];
						mixes[j][5] = buffer->result[j].mix[5];
						mixes[j][6] = buffer->result[j].mix[6];
						mixes[j][7] = buffer->result[j].mix[7];
					}
				}
			}
		}
		run_ethash_search(s_gridSize, s_blockSize, stream, buffer, m_current_nonce, m_parallelHash);
		if (m_current_index >= s_numStreams)
		{
			if (found_count)
				for (uint32_t i = 0; i < found_count; i++)
					if (s_noeval)
						farm.submitProof(Solution{nonces[i], *((const h256 *)mixes[i]), w, m_new_work});
					else
					{
						Result r = EthashAux::eval(w.seed, w.header, nonces[i]);
						if (r.value < w.boundary)
							farm.submitProof(Solution{nonces[i], r.mixHash, w, m_new_work});
						else
						{
							farm.failedSolution();
							cwarn << "GPU gave incorrect result!";
						}
					}

			addHashCount(batch_size);
			bool t = true;
			if (m_new_work.compare_exchange_strong(t, false)) {
				cudaswitchlog << "Switch time "
					<< std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - workSwitchStart).count()
					<< "ms.";
				break;
			}
			if (shouldStop())
			{
				m_new_work.store(false, std::memory_order_relaxed);
				break;
			}
		}
	}
}

