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
/** @file EthashGPUMiner.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Determines the PoW algorithm.
 */

#if ETH_ETHASHCL

#include "EthashGPUMiner.h"
#include <thread>
#include <chrono>
#include <libethash-cl/ethash_cl_miner.h>

using namespace std;
using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

class EthashCLHook: public ethash_cl_miner::search_hook
{
public:
	EthashCLHook(EthashGPUMiner* _owner): m_owner(_owner) {}
	EthashCLHook(EthashCLHook const&) = delete;

	void abort()
	{
		{
			UniqueGuard l(x_all);
			if (m_aborted)
				return;
//		cdebug << "Attempting to abort";

			m_abort = true;
		}
		// m_abort is true so now searched()/found() will return true to abort the search.
		// we hang around on this thread waiting for them to point out that they have aborted since
		// otherwise we may end up deleting this object prior to searched()/found() being called.
		m_aborted.wait(true);
//		for (unsigned timeout = 0; timeout < 100 && !m_aborted; ++timeout)
//			std::this_thread::sleep_for(chrono::milliseconds(30));
//		if (!m_aborted)
//			cwarn << "Couldn't abort. Abandoning OpenCL process.";
	}

	void reset()
	{
		UniqueGuard l(x_all);
		m_aborted = m_abort = false;
	}

protected:
	virtual bool found(uint64_t const* _nonces, uint32_t _count) override
	{
//		dev::operator <<(std::cerr << "Found nonces: ", vector<uint64_t>(_nonces, _nonces + _count)) << std::endl;
		for (uint32_t i = 0; i < _count; ++i)
			if (m_owner->report(_nonces[i]))
				return (m_aborted = true);
		return m_owner->shouldStop();
	}

	virtual bool searched(uint64_t _startNonce, uint32_t _count) override
	{
		UniqueGuard l(x_all);
//		std::cerr << "Searched " << _count << " from " << _startNonce << std::endl;
		m_owner->accumulateHashes(_count);
		m_last = _startNonce + _count;
		if (m_abort || m_owner->shouldStop())
			return (m_aborted = true);
		return false;
	}

private:
	Mutex x_all;
	uint64_t m_last;
	bool m_abort = false;
	Notified<bool> m_aborted = {true};
	EthashGPUMiner* m_owner = nullptr;
};

}
}

unsigned EthashGPUMiner::s_platformId = 0;
unsigned EthashGPUMiner::s_deviceId = 0;
unsigned EthashGPUMiner::s_numInstances = 0;
int EthashGPUMiner::s_devices[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

EthashGPUMiner::EthashGPUMiner(ConstructionInfo const& _ci):
	Miner(_ci),
	Worker("openclminer" + toString(index())),
	m_hook(new EthashCLHook(this))
{
}

EthashGPUMiner::~EthashGPUMiner()
{
	pause();
	delete m_miner;
	delete m_hook;
}

bool EthashGPUMiner::report(uint64_t _nonce)
{
	Nonce n = (Nonce)(u64)_nonce;
	Result r = EthashAux::eval(work().seedHash, work().headerHash, n);
	if (r.value < work().boundary)
		return submitProof(Solution{n, r.mixHash, work().headerHash, work().seedHash, work().boundary});
	return false;
}

void EthashGPUMiner::kickOff()
{
	m_hook->reset();
	startWorking();
}

void EthashGPUMiner::workLoop()
{
	// take local copy of work since it may end up being overwritten by kickOff/pause.
	try {
		WorkPackage w = work();
		cnote << "set work; seed: " << "#" + w.seedHash.hex().substr(0, 8) + ", target: " << "#" + w.boundary.hex().substr(0, 12);
		if (!m_miner || m_minerSeed != w.seedHash)
		{
			if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
			{
				while (s_dagLoadIndex < index()) {
					this_thread::sleep_for(chrono::seconds(1));
				}
			}

			cnote << "Initialising miner...";
			m_minerSeed = w.seedHash;

			delete m_miner;
			m_miner = new ethash_cl_miner;

			unsigned device = s_devices[index()] > -1 ? s_devices[index()] : index();

			EthashAux::LightType light;
			light = EthashAux::light(w.seedHash);
			bytesConstRef lightData = light->data();

			m_miner->init(light->light, lightData.data(), lightData.size(), s_platformId,  device);
			s_dagLoadIndex++;
		}

		uint64_t upper64OfBoundary = (uint64_t)(u64)((u256)w.boundary >> 192);
		uint64_t startN = 0;
		if (w.exSizeBits >= 0)
			startN = w.startNonce | ((uint64_t)index() << (64 - 4 - w.exSizeBits)); // this can support up to 16 devices
		m_miner->search(w.headerHash.data(), upper64OfBoundary, *m_hook, (w.exSizeBits >= 0), startN);
	}
	catch (cl::Error const& _e)
	{
		delete m_miner;
		m_miner = nullptr;
		cwarn << "Error GPU mining: " << _e.what() << "(" << _e.err() << ")";
	}
}

void EthashGPUMiner::pause()
{
	m_hook->abort();
	stopWorking();
}

std::string EthashGPUMiner::platformInfo()
{
	return ethash_cl_miner::platform_info(s_platformId, s_deviceId);
}

unsigned EthashGPUMiner::getNumDevices()
{
	return ethash_cl_miner::getNumDevices(s_platformId);
}

void EthashGPUMiner::listDevices()
{
	return ethash_cl_miner::listDevices();
}

bool EthashGPUMiner::configureGPU(
	unsigned _localWorkSize,
	unsigned _globalWorkSizeMultiplier,
	unsigned _platformId,
	unsigned _deviceId,
	unsigned _extraGPUMemory,
	uint64_t _currentBlock,
	unsigned _dagLoadMode,
	unsigned _dagCreateDevice
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;

	s_platformId = _platformId;
	s_deviceId = _deviceId;

	_localWorkSize = ((_localWorkSize + 7) / 8) * 8;

	if (!ethash_cl_miner::configureGPU(
			_platformId,
			_localWorkSize,
			_globalWorkSizeMultiplier * _localWorkSize,
			_extraGPUMemory,
			_currentBlock
			)
	)
	{
		cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl;
		return false;
	}
	return true;
}

#endif
