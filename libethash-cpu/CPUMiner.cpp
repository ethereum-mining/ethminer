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

/*
 CPUMiner simulates mining devices but does NOT real mine!
 USE FOR DEVELOPMENT ONLY !
*/

#if defined(__linux__)
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* we need sched_setaffinity() */
#endif
#include <error.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include <boost/version.hpp>

#if 0
#include <boost/fiber/numa/pin_thread.hpp>
#include <boost/fiber/numa/topology.hpp>
#endif

#include "CPUMiner.h"


/* Sanity check for defined OS */
#if defined(__APPLE__) || defined(__MACOSX)
/* MACOSX */
#elif defined(__linux__)
/* linux */
#elif defined(_WINDOWS) || defined(_WIN32)
/* windows */
#else
#error "Invalid OS configuration"
#endif


using namespace std;
using namespace dev;
using namespace eth;


/* ################## OS-specific functions ################## */

/*
 * returns physically available memory (no swap)
 */
static size_t getTotalPhysAvailableMemory()
{
#if defined(__APPLE__) || defined(__MACOSX)
#error "TODO: Function CPUMiner getTotalPhysAvailableMemory() on MAXOSX not implemented"
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    if (pages == -1L)
    {
        cwarn << "Error in func " << __FUNCTION__ << " at sysconf(_SC_AVPHYS_PAGES) \""
              << strerror(errno) << "\"\n";
        return 0;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1L)
    {
        cwarn << "Error in func " << __FUNCTION__ << " at sysconf(_SC_PAGESIZE) \""
              << strerror(errno) << "\"\n";
        return 0;
    }

    return (size_t)pages * (size_t)page_size;
#else
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo) == 0)
    {
        // Handle Errorcode (GetLastError) ??
        return 0;
    }
    return memInfo.ullAvailPhys;
#endif
}

/*
 * return numbers of available CPUs
 */
unsigned CPUMiner::getNumDevices()
{
#if 0
    static unsigned cpus = 0;

    if (cpus == 0)
    {
        std::vector< boost::fibers::numa::node > topo = boost::fibers::numa::topology();
        for (auto n : topo) {
            cpus += n.logical_cpus.size();
        }
    }
    return cpus;
#elif defined(__APPLE__) || defined(__MACOSX)
#error "TODO: Function CPUMiner::getNumDevices() on MAXOSX not implemented"
#elif defined(__linux__)
    long cpus_available;
    cpus_available = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus_available == -1L)
    {
        cwarn << "Error in func " << __FUNCTION__ << " at sysconf(_SC_NPROCESSORS_ONLN) \""
              << strerror(errno) << "\"\n";
        return 0;
    }
    return cpus_available;
#else
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#endif
}


/* ######################## CPU Miner ######################## */

struct CPUChannel : public LogChannel
{
    static const char* name() { return EthOrange "cp"; }
    static const int verbosity = 2;
};
#define cpulog clog(CPUChannel)


CPUMiner::CPUMiner(unsigned _index, CPSettings _settings, DeviceDescriptor& _device)
  : Miner("cpu-", _index), m_settings(_settings)
{
    m_deviceDescriptor = _device;
}


CPUMiner::~CPUMiner()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::~CPUMiner() begin");
    stopWorking();
    kick_miner();
    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::~CPUMiner() end");
}


/*
 * Bind the current thread to a spcific CPU
 */
bool CPUMiner::initDevice()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::initDevice begin");

    cpulog << "Using CPU: " << m_deviceDescriptor.cpCpuNumber << " " << m_deviceDescriptor.cuName
           << " Memory : " << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory);

#if defined(__APPLE__) || defined(__MACOSX)
#error "TODO: Function CPUMiner::initDevice() on MAXOSX not implemented"
#elif defined(__linux__)
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);
    CPU_SET(m_deviceDescriptor.cpCpuNumber, &cpuset);

    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err != 0)
    {
        cwarn << "Error in func " << __FUNCTION__ << " at sched_setaffinity() \"" << strerror(errno)
              << "\"\n";
        cwarn << "cp-" << m_index << "could not bind thread to cpu" << m_deviceDescriptor.cpCpuNumber
              << "\n";
    }
#else
    DWORD_PTR dwThreadAffinityMask = 1i64 << m_deviceDescriptor.cpCpuNumber;
    DWORD_PTR previous_mask;
    previous_mask = SetThreadAffinityMask(GetCurrentThread(), dwThreadAffinityMask);
    if (previous_mask == NULL)
    {
        cwarn << "cp-" << m_index << "could not bind thread to cpu" << m_deviceDescriptor.cpCpuNumber
              << "\n";
        // Handle Errorcode (GetLastError) ??
    }
#endif
    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::initDevice end");
    return true;
}


void CPUMiner::ethash_search(WorkPackage& _w)
{
    const auto& context = ethash::get_global_epoch_context_full(_w.epoch);
    auto header = ethash::hash256_from_bytes(_w.header.data());
    auto boundary = ethash::hash256_from_bytes(_w.boundary.data());

    const auto& context = ethash::get_global_epoch_context_full(w.epoch);
    const auto header = ethash::hash256_from_bytes(w.header.data());
    const auto boundary = ethash::hash256_from_bytes(w.boundary.data());
    auto nonce = w.startNonce;

    while (true)
    {

        // Exit next time around if there's new work awaiting
        bool t = true;
        if (m_new_work.compare_exchange_strong(t, false) || paused() || shouldStop())
            break;

        auto r = ethash::search(context, header, boundary, _w.startNonce, m_settings.batchSize);
        if (r.solution_found)
        {
            h256 mix{reinterpret_cast<byte*>(r.mix_hash.bytes), h256::ConstructFromPointer};
            auto sol = Solution{r.nonce, mix, _w, std::chrono::steady_clock::now(), m_index};

            cpulog << EthWhite << "Job: " << _w.header.abridged()
                   << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;

            Farm::f().submitProof(sol);
        }

        // Update the hash rate
        updateHashRate(m_settings.batchSize, 1);
    }
}

void dev::eth::CPUMiner::progpow_search(WorkPackage& _w) 
{
    const auto& context = progpow::get_global_epoch_context_full(_w.epoch);
    auto header = progpow::hash256_from_bytes(_w.header.data());
    auto boundary = progpow::hash256_from_bytes(_w.boundary.data());

    while (true)
    {
        // Exit next time around if there's new work awaiting
        bool t = true;
        if (m_new_work.compare_exchange_strong(t, false) || paused() || shouldStop())
            break;

        auto r = progpow::search(context, _w.block, header, boundary, _w.startNonce, m_settings.batchSize);
        if (r.solution_found)
        {
            h256 mix{reinterpret_cast<byte*>(r.mix_hash.bytes), h256::ConstructFromPointer};
            auto sol = Solution{r.nonce, mix, _w, std::chrono::steady_clock::now(), m_index};

            cpulog << EthWhite << "Job: " << _w.header.abridged()
                   << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;
            Farm::f().submitProof(sol);
        }
        nonce += blocksize;

        // Update the hash rate
        updateHashRate(m_settings.batchSize, 1);
    }
}

void dev::eth::CPUMiner::compileProgPoWKernel(int _block, int _dagelms) 
{
    // CPU miner does not have any kernel to compile
    // Nevertheless the class must override base class
    (void)_block;
    (void)_dagelms;
}


/*
 * The main work loop of a Worker thread
 */
void CPUMiner::workLoop()
{
    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::workLoop() begin");

    if (!initDevice())
        return;

    minerLoop();

    DEV_BUILD_LOG_PROGRAMFLOW(cpulog, "cp-" << m_index << " CPUMiner::workLoop() end");
}


void CPUMiner::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
    unsigned numDevices = getNumDevices();

    for (unsigned i = 0; i < numDevices; i++)
    {
        string uniqueId;
        ostringstream s;
        DeviceDescriptor deviceDescriptor;

        s << "cpu-" << i;
        uniqueId = s.str();
        if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
            deviceDescriptor = _DevicesCollection[uniqueId];
        else
            deviceDescriptor = DeviceDescriptor();

        s.str("");
        s.clear();
        s << "ethash::eval()/boost " << (BOOST_VERSION / 100000) << "."
          << (BOOST_VERSION / 100 % 1000) << "." << (BOOST_VERSION % 100);
        deviceDescriptor.name = s.str();
        deviceDescriptor.uniqueId = uniqueId;
        deviceDescriptor.type = DeviceTypeEnum::Cpu;
        deviceDescriptor.totalMemory = getTotalPhysAvailableMemory();

        deviceDescriptor.cpCpuNumber = i;

        _DevicesCollection[uniqueId] = deviceDescriptor;
    }
}
