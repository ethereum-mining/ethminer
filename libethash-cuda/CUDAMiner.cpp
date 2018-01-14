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

#include "CUDAMiner.h"

using namespace std;
using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{
	class EthashCUDAHook : public ethash_cuda_miner::search_hook
	{
	public:
		EthashCUDAHook(CUDAMiner& _owner): m_owner(_owner) {}

		EthashCUDAHook(EthashCUDAHook const&) = delete;

		void abort()
		{
			{
				UniqueGuard l(x_all);
				if (m_aborted)
					return;

				m_abort = true;
			}
                }

                void wait()
                {
			// m_abort is true so now searched()/found() will return true to abort the search.
			// we hang around on this thread waiting for them to point out that they have aborted since
			// otherwise we may end up deleting this object prior to searched()/found() being called.
			m_aborted.wait(true);
		}

		void reset()
		{
			UniqueGuard l(x_all);
			m_aborted = m_abort = false;
		}

		bool isStale()
		{
			return m_abort;
		}


	protected:
		void found(uint64_t const* _nonces, uint32_t count) override
		{
			for (uint32_t i = 0; i < count; i++)
				m_owner.report(_nonces[i]);
		}

		void searched(uint32_t _count) override
		{
			m_owner.addHashCount(_count);
		}

		bool shouldStop() override
		{
			if (m_abort || m_owner.shouldStop())
				return (m_aborted = true);
			return false;
		}

	private:
		Mutex x_all;
		bool m_abort = false;
		Notified<bool> m_aborted = { true };
		CUDAMiner& m_owner;
	};
}
}
unsigned CUDAMiner::s_numInstances = 0;
int CUDAMiner::s_devices[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

CUDAMiner::CUDAMiner(FarmFace& _farm, unsigned _index) :
	Miner("CUDA", _farm, _index),
	m_hook(new EthashCUDAHook(*this)),
	m_miner(getNumDevices())
{}

CUDAMiner::~CUDAMiner()
{
	stopWorking();
	pause();
	delete m_hook;
}

void CUDAMiner::report(uint64_t _nonce)
{
	// FIXME: This code is exactly the same as in EthashGPUMiner.
	WorkPackage w = work();  // Copy work package to avoid repeated mutex lock.
	Result r = EthashAux::eval(w.seed, w.header, _nonce);
	if (r.value < w.boundary)
		farm.submitProof(Solution{_nonce, r.mixHash, w.header, w.seed, w.boundary, w.job, m_hook->isStale()});
	else
	{
		farm.failedSolution();
		cwarn << "FAILURE: GPU gave incorrect result!";

	}
}

void CUDAMiner::kickOff()
{
	m_hook->reset();
}

bool CUDAMiner::init(const h256& seed)
{
	// take local copy of work since it may end up being overwritten by kickOff/pause.
	try {
		unsigned device = s_devices[index] > -1 ? s_devices[index] : index;

		cnote << "Initialising miner...";

		EthashAux::LightType light;
		light = EthashAux::light(seed);
		bytesConstRef lightData = light->data();

		m_miner.init(getNumDevices(), light->light, lightData.data(), lightData.size(), 
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
			const WorkPackage w = getWork();
			
			if (current.header != w.header || current.seed != w.seed)
			{
				if(!w || w.header == h256())
				{
					cnote << "No work. Pause for 3 s.";
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}
				
				//cnote << "set work; seed: " << "#" + w.seed.hex().substr(0, 8) + ", target: " << "#" + w.boundary.hex().substr(0, 12);
				if (current.seed != w.seed)
				{
					if(!init(w.seed))
						break;
				}
				current = w;
			}
			uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)current.boundary >> 192);
			uint64_t startN = current.startNonce;
			if (current.exSizeBits >= 0) 
				startN = current.startNonce | ((uint64_t)index << (64 - 4 - current.exSizeBits)); // this can support up to 16 devices
			m_miner.search(current.header.data(), upper64OfBoundary, *m_hook, (current.exSizeBits >= 0), startN);

			// Check if we should stop.
			if (shouldStop())
			{
				break;
			}
		}
	}
	catch (std::runtime_error const& _e)
	{
		cwarn << "Error CUDA mining: " << _e.what();
	}
}

void CUDAMiner::pause()
{
	m_hook->abort();
}

void CUDAMiner::waitPaused()
{
	m_hook->wait();
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
		string outString = "\nListing CUDA devices.\nFORMAT: [deviceID] deviceName\n";
		int numDevices = getNumDevices();
		for (int i = 0; i < numDevices; ++i)
		{
			cudaDeviceProp props;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

			outString += "[" + to_string(i) + "] " + string(props.name) + "\n";
			outString += "\tCompute version: " + to_string(props.major) + "." + to_string(props.minor) + "\n";
			outString += "\tcudaDeviceProp::totalGlobalMem: " + to_string(props.totalGlobalMem) + "\n";
		}
		std::cout << outString;
	}
	catch(std::runtime_error const& err)
	{
		cwarn << "CUDA error: " << err.what();
	}
}

HwMonitor CUDAMiner::hwmon()
{
	return m_miner.hwmon();
}

bool CUDAMiner::configureGPU(
	unsigned _blockSize,
	unsigned _gridSize,
	unsigned _numStreams,
	unsigned _scheduleFlag,
	uint64_t _currentBlock,
	unsigned _dagLoadMode,
	unsigned _dagCreateDevice
	)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;
	_blockSize = ((_blockSize + 7) / 8) * 8;

	if (!ethash_cuda_miner::configureGPU(
		getNumDevices(),
		s_devices,
		_blockSize,
		_gridSize,
		_numStreams,
		_scheduleFlag,
		_currentBlock)
		)
	{
		cout << "No CUDA device with sufficient memory was found. Can't CUDA mine. Remove the -U argument" << endl;
		return false;
	}
	return true;
}

void CUDAMiner::setParallelHash(unsigned _parallelHash)
{
	ethash_cuda_miner::setParallelHash(_parallelHash);
}
