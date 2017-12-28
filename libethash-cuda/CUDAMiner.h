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
/** @file CUDAMiner.h
* @author Gav Wood <i@gavwood.com>
* @date 2014
*
* Determines the PoW algorithm.
*/
#pragma once

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include "ethash_cuda_miner.h"

namespace dev
{
namespace eth
{
class EthashCUDAHook;

	class CUDAMiner: public Miner
	{
		friend class dev::eth::EthashCUDAHook;

	public:
		CUDAMiner(FarmFace& _farm, unsigned _index);
		~CUDAMiner();

		static unsigned instances() 
		{ 
			return s_numInstances > 0 ? s_numInstances : 1; 
		}
		static std::string platformInfo();
		static unsigned getNumDevices();
		static void listDevices();
		static void setParallelHash(unsigned _parallelHash);
		static bool configureGPU(
			unsigned _blockSize,
			unsigned _gridSize,
			unsigned _numStreams,
			unsigned _scheduleFlag,
			uint64_t _currentBlock,
			unsigned _dagLoadMode,
			unsigned _dagCreateDevice
			);
		static void setNumInstances(unsigned _instances) 
		{ 
			s_numInstances = std::min<unsigned>(_instances, getNumDevices());
		}
		static void setDevices(unsigned * _devices, unsigned _selectedDeviceCount) 
		{
			for (unsigned i = 0; i < _selectedDeviceCount; i++) 
			{
				s_devices[i] = _devices[i];
			}
		}
		HwMonitor hwmon() override;
	protected:
		void kickOff() override;
		void pause() override;
	private:
		void workLoop() override;
		void report(uint64_t _nonce);
		void initDevice(WorkPackage w);

		bool init(const h256& seed);

		EthashCUDAHook* m_hook = nullptr;
		ethash_cuda_miner* m_miner = nullptr;

		h256 m_minerSeed;		///< Last seed in m_miner
		static unsigned s_platformId;
		static unsigned s_deviceId;
		static unsigned s_numInstances;
		static int s_devices[16];

	};
}
}
