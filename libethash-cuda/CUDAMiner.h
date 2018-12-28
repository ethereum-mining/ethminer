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

#pragma once

#include "ethash_miner_kernel.h"
#include "progpow_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include <libprogpow/ProgPow.h>

#include <cuda.h>
#include <functional>

namespace dev
{
namespace eth
{
class CUDAMiner : public Miner
{
public:
    CUDAMiner(unsigned _index, CUSettings _settings, DeviceDescriptor& _device);
    ~CUDAMiner() override = default;

    static int getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);


protected:

    bool initDevice() override;

    bool initEpoch_internal() override;

private:
    
    void workLoop() override;

    void ethash_search() override;
    void progpow_search() override;
    void compileProgPoWKernel(int _block, int _dagelms) override;

    CUdevice m_device;
    CUmodule m_module;
    CUfunction m_kernel;

    std::vector<volatile search_results *> m_search_results;

    hash128_t* m_dag;
    hash64_t* m_dag_progpow;
    hash64_t* m_light;

    std::vector<cudaStream_t> m_streams;
    uint64_t m_current_target = 0;

    CUSettings m_settings;

    const uint32_t m_batch_size;
    const uint32_t m_streams_batch_size;

    uint64_t m_allocated_memory_dag = 0; // dag_size is a uint64_t in EpochContext struct
    size_t m_allocated_memory_light_cache = 0;
};


}  // namespace eth
}  // namespace dev
