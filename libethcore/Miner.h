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

#include <bitset>
#include <list>
#include <numeric>
#include <string>

#include "EthashAux.h"
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/Worker.h>

#include <boost/format.hpp>
#include <boost/thread.hpp>

#include <libprogpow/ProgPow.h>

#define DAG_LOAD_MODE_PARALLEL 0
#define DAG_LOAD_MODE_SEQUENTIAL 1

#define ETHASH_REVISION 23
#define ETHASH_DATASET_BYTES_INIT 1073741824U  // 2**30
#define ETHASH_DATASET_BYTES_GROWTH 8388608U   // 2**23
#define ETHASH_CACHE_BYTES_INIT 1073741824U    // 2**24
#define ETHASH_CACHE_BYTES_GROWTH 131072U      // 2**17
#define ETHASH_MIX_BYTES 256
#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 256
#define ETHASH_CACHE_ROUNDS 3
#define ETHASH_ACCESSES 64

using namespace std;

namespace dev
{
namespace eth
{
enum class DeviceTypeEnum
{
    Unknown,
    Cpu,
    Gpu,
    Accelerator
};

enum class DeviceSubscriptionTypeEnum
{
    None,
    OpenCL,
    Cuda,
    Cpu

};

enum class MinerType
{
    Mixed,
    CL,
    CUDA,
    CPU
};

enum class HwMonitorInfoType
{
    UNKNOWN,
    NVIDIA,
    AMD,
    CPU
};

enum class ClPlatformTypeEnum
{
    Unknown,
    Amd,
    Clover,
    Nvidia
};

enum class SolutionAccountingEnum
{
    Accepted,
    Rejected,
    Wasted,
    Failed
};

// Holds settings for CUDA Miner
struct CUSettings
{
    vector<unsigned> devices;
    unsigned streams = 2;
    unsigned schedule = 4;
    unsigned gridSize = 8192;
    unsigned blockSize = 128;
    unsigned parallelHash = 4;
};

// Holds settings for OpenCL Miner
struct CLSettings
{
    vector<unsigned> devices;
    bool noBinary = false;
    unsigned globalWorkSize = 0;
    unsigned globalWorkSizeMultiplier = 65536;
    unsigned localWorkSize = 128;
};

// Holds settings for CPU Miner
struct CPSettings
{
    vector<unsigned> devices;
    unsigned batchSize = 30U;
};

struct SolutionAccountType
{
    unsigned accepted = 0;
    unsigned rejected = 0;
    unsigned wasted = 0;
    unsigned failed = 0;
    std::chrono::steady_clock::time_point tstamp = std::chrono::steady_clock::now();
    string str()
    {
        string _ret = "A" + to_string(accepted);
        if (wasted)
            _ret.append(":W" + to_string(wasted));
        if (rejected)
            _ret.append(":R" + to_string(rejected));
        if (failed)
            _ret.append(":F" + to_string(failed));
        return _ret;
    };
};

struct HwSensorsType
{
    int tempC = 0;
    int fanP = 0;
    double powerW = 0.0;
    string str()
    {
        string _ret = to_string(tempC) + "C " + to_string(fanP) + "%";
        if (powerW)
            _ret.append(boost::str(boost::format("%6.1f") % powerW));
        return _ret;
    };
};

struct TelemetryAccountType
{
    string prefix = "";
    float hashrate = 0.0f;
    bool paused = false;
    HwSensorsType sensors;
    SolutionAccountType solutions;
    unsigned long totalJobs;  // Total number of jobs received from WorkProvider(s)
};

struct DeviceDescriptor
{
    DeviceTypeEnum type = DeviceTypeEnum::Unknown;
    DeviceSubscriptionTypeEnum subscriptionType = DeviceSubscriptionTypeEnum::None;

    string uniqueId;     // For GPUs this is the PCI ID
    size_t totalMemory;  // Total memory available by device
    string name;         // Device Name

    bool clDetected;  // For OpenCL detected devices
    string clName;
    unsigned int clPlatformId;
    string clPlatformName;
    ClPlatformTypeEnum clPlatformType = ClPlatformTypeEnum::Unknown;
    string clPlatformVersion;
    unsigned int clPlatformVersionMajor;
    unsigned int clPlatformVersionMinor;
    unsigned int clDeviceOrdinal;
    unsigned int clDeviceIndex;
    string clDeviceVersion;
    unsigned int clDeviceVersionMajor;
    unsigned int clDeviceVersionMinor;
    string clBoardName;
    size_t clMaxMemAlloc;
    size_t clMaxWorkGroup;
    unsigned int clMaxComputeUnits;
    string clNvCompute;
    unsigned int clNvComputeMajor;
    unsigned int clNvComputeMinor;

    bool cuDetected;  // For CUDA detected devices
    string cuName;
    unsigned int cuDeviceOrdinal;
    unsigned int cuDeviceIndex;
    string cuCompute;
    unsigned int cuComputeMajor;
    unsigned int cuComputeMinor;

    int cpCpuNumber;  // For CPU
};

struct HwMonitorInfo
{
    HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
    string devicePciId;
    int deviceIndex = -1;
};

/// Pause mining
enum MinerPauseEnum
{
    PauseDueToOverHeating,
    PauseDueToAPIRequest,
    PauseDueToFarmPaused,
    PauseDueToInsufficientMemory,
    PauseDueToInitEpochError,
    Pause_MAX  // Must always be last as a placeholder of max count
};

/// Keeps track of progress for farm and miners
struct TelemetryType
{
    bool hwmon = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    TelemetryAccountType farm;
    std::vector<TelemetryAccountType> miners;
    std::string str()
    {
        std::stringstream _ret;

        /*

        Output is formatted as

        Run <h:mm> <Solutions> <Speed> [<miner> ...]
        where
        - Run h:mm    Duration of the batch
        - Solutions   Detailed solutions (A+R+F) per farm
        - Speed       Actual hashing rate

        each <miner> reports
        - speed       Actual speed at the same level of
                      magnitude for farm speed
        - sensors     Values of sensors (temp, fan, power)
        - solutions   Optional (LOG_PER_GPU) Solutions detail per GPU
        */

        /*
        Calculate duration
        */
        auto duration = std::chrono::steady_clock::now() - start;
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        int hoursSize = (hours.count() > 9 ? (hours.count() > 99 ? 3 : 2) : 1);
        duration -= hours;
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        _ret << EthGreen << setw(hoursSize) << hours.count() << ":" << setfill('0') << setw(2)
             << minutes.count() << EthReset << EthWhiteBold << " " << farm.solutions.str()
             << EthReset << " ";

        /*
        Github : @AndreaLanfranchi
        I whish I could simply make use of getFormattedHashes but in this case
        this would be misleading as total hashrate could be of a different order
        of magnitude than the hashrate expressed by single devices.
        Thus I need to set the vary same scaling index on the farm and on devices
        */
        static string suffixes[] = {"h", "Kh", "Mh", "Gh"};
        float hr = farm.hashrate;
        int magnitude = 0;
        while (hr > 1000.0f && magnitude <= 3)
        {
            hr /= 1000.0f;
            magnitude++;
        }

        _ret << EthTealBold << std::fixed << std::setprecision(2) << hr << " "
             << suffixes[magnitude] << EthReset << " { ";

        int i = -1;                 // Current miner index
        int m = miners.size() - 1;  // Max miner index
        for (TelemetryAccountType miner : miners)
        {
            i++;
            hr = miner.hashrate;
            if (hr > 0.0f)
                hr /= pow(1000.0f, magnitude);

            _ret << (miner.paused ? EthRed : "") << miner.prefix << i << " " << EthTeal
                 << std::fixed << std::setprecision(2) << hr << EthReset;

            if (hwmon)
                _ret << " " << EthTeal << miner.sensors.str() << EthReset;

            // Eventually push also solutions per single GPU
            if (g_logOptions & LOG_PER_GPU)
                _ret << " " << EthTeal << miner.solutions.str() << EthReset;

            // Separator if not the last miner index
            if (i < m)
                _ret << " | ";
        }
        _ret << " }";

        return _ret.str();
    };
};


/**
 * @brief Class for hosting one or more Miners.
 * @warning Must be implemented in a threadsafe manner since it will be called from multiple
 * miner threads.
 */
class FarmFace
{
public:
    FarmFace() { m_this = this; }
    static FarmFace& f() { return *m_this; };

    virtual ~FarmFace() = default;
    virtual unsigned get_tstart() = 0;
    virtual unsigned get_tstop() = 0;
    virtual unsigned get_ergodicity() = 0;

    /**
     * @brief Called from a Miner to note a WorkPackage has a solution.
     * @param _p The solution.
     * @return true iff the solution was good (implying that mining should be .
     */
    virtual void submitProof(Solution const& _p) = 0;
    virtual void accountSolution(unsigned _minerIdx, SolutionAccountingEnum _accounting) = 0;
    virtual uint64_t get_nonce_scrambler() = 0;
    virtual unsigned get_segment_width() = 0;

private:
    static FarmFace* m_this;
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */

class Miner : public Worker
{
public:
    Miner(std::string const& _name, unsigned _index);

    ~Miner() override = default;

    // Sets basic info for eventual serialization of DAG load
    static void setDagLoadInfo(unsigned _mode, unsigned _devicecount)
    {
        s_dagLoadMode = _mode;
        s_dagLoadIndex = 0;
        s_minersCount = _devicecount;
    };

    /**
     * @brief Gets the device descriptor assigned to this instance
     */
    DeviceDescriptor getDescriptor();

    /**
     * @brief Assigns hashing work to this instance
     */
    void setWork(WorkPackage const& _work);

    /**
     * @brief Assigns Epoch context to this instance
     */
    void setEpoch(EpochContext const& _ec) { m_epochContext = _ec; }

    unsigned Index() { return m_index; };

    HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    void setHwmonDeviceIndex(int i) { m_hwmoninfo.deviceIndex = i; }

    // Stop worker thread; causes call to stopWorking() and waits till thread has stopped.
    void stopWorking() override;

    /**
     * @brief Kick an asleep miner.
     */
    virtual void kick_miner();

    /**
     * @brief Pauses mining setting a reason flag
     */
    void pause(MinerPauseEnum what);

    /**
     * @brief Whether or not this miner is paused for any reason
     */
    bool paused();

    /**
     * @brief Checks if the given reason for pausing is currently active
     */
    bool pauseTest(MinerPauseEnum what);

    /**
     * @brief Returns the human readable reason for this miner being paused
     */
    std::string pausedString();

    /**
     * @brief Cancels a pause flag.
     * @note Miner can be paused for multiple reasons at a time.
     */
    void resume(MinerPauseEnum fromwhat);

    /**
     * @brief Retrieves currrently collected hashrate
     */
    float RetrieveHashRate() noexcept;


protected:
    /**
     * @brief Initializes miner's device.
     */
    virtual bool initDevice() = 0;

    /**
     * @brief Initializes miner to current (or changed) epoch.
     */
    bool initEpoch();

    /**
     * @brief Miner's specific initialization to current (or changed) epoch.
     */
    virtual bool initEpoch_internal();

    // This is the effective miner Loop
    void minerLoop();

    // Collects and averages (EMA) hashrate
    void updateHashRate(uint32_t _hashes, uint64_t _microseconds, float _alpha = 0.85f) noexcept;

    static unsigned s_minersCount;   // Total Number of Miners
    static unsigned s_dagLoadMode;   // Way dag should be loaded
    static unsigned s_dagLoadIndex;  // In case of serialized load of dag this is the index of miner
                                     // which should load next

    const unsigned m_index = 0;           // Ordinal index of the Instance (not the device)
    DeviceDescriptor m_deviceDescriptor;  // Info about the device

    EpochContext m_epochContext;

#ifdef _DEVELOPER
    std::chrono::steady_clock::time_point m_workSwitchStart;
#endif

    HwMonitorInfo m_hwmoninfo;
    mutable boost::mutex x_work;
    mutable boost::mutex x_pause;

    atomic<bool> m_new_work = {false};
    boost::condition_variable m_new_work_signal;
    boost::condition_variable m_dag_loaded_signal;

    WorkPackage m_work_latest, m_work_active;
    uint64_t m_current_target = 0;

    std::chrono::steady_clock::time_point m_workSearchStart;  // Actual start point of hashing time
    uint64_t m_workSearchDuration = 0;                        // Total search duration on a job
    uint32_t m_workHashes;                                    // Total hashes processed on a job

private:
    bitset<MinerPauseEnum::Pause_MAX> m_pauseFlags;


    virtual void ethash_search() = 0;
    virtual void progpow_search() = 0;
    virtual void compileProgPoWKernel(int _block, int _dagelms) = 0;
    virtual void unloadProgPoWKernel(){};

    std::atomic<float> m_hr = {0.0};
    std::atomic<bool> m_hrLive = {false};
};

}  // namespace eth
}  // namespace dev
