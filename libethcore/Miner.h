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
<<<<<<< HEAD
=======
#include "EthashAux.h"

#define MINER_WAIT_STATE_WORK    1
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

#include <boost/format.hpp>
#include <boost/thread.hpp>

<<<<<<< HEAD
#define DAG_LOAD_MODE_PARALLEL 0
#define DAG_LOAD_MODE_SEQUENTIAL 1
=======
#define DAG_LOAD_MODE_PARALLEL   0
#define DAG_LOAD_MODE_SEQUENTIAL 1
#define DAG_LOAD_MODE_SINGLE     2
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

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
<<<<<<< HEAD
    CPU
=======
    Fpga
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

enum class HwMonitorInfoType
{
    UNKNOWN,
    NVIDIA,
<<<<<<< HEAD
    AMD,
    CPU
=======
    AMD
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

enum class ClPlatformTypeEnum
{
<<<<<<< HEAD
    Unknown,
    Amd,
    Clover,
    Nvidia,
    Intel
=======
    UNKNOWN,
    OPENCL,
    CUDA,
    FPGA
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

enum class SolutionAccountingEnum
{
<<<<<<< HEAD
    Accepted,
    Rejected,
    Wasted,
    Failed
=======
    HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
    HwMonitorIndexSource indexSource = HwMonitorIndexSource::UNKNOWN;
    int deviceIndex = -1;

>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

struct MinerSettings
{
<<<<<<< HEAD
    vector<unsigned> devices;
=======
    int tempC = 0;
    int fanP = 0;
    double powerW = 0;
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

// Holds settings for CUDA Miner
struct CUSettings : public MinerSettings
{
<<<<<<< HEAD
    unsigned streams = 2;
    unsigned schedule = 4;
    unsigned gridSize = 8192;
    unsigned blockSize = 128;
};
=======
    string power = "";
    if(_hw.powerW != 0){
        ostringstream stream;
        stream << fixed << setprecision(0) << _hw.powerW << "W";
        power = stream.str();
    }
    return os << _hw.tempC << "C " << _hw.fanP << "% " << power;
}
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

// Holds settings for OpenCL Miner
struct CLSettings : public MinerSettings
{
<<<<<<< HEAD
    bool noBinary = false;
    bool noExit = false;
    unsigned globalWorkSize = 0;
    unsigned globalWorkSizeMultiplier = 65536;
    unsigned localWorkSize = 128;
=======
    uint64_t hashes = 0;        ///< Total number of hashes computed.
    uint64_t ms = 0;            ///< Total number of milliseconds of mining thus far.
    uint64_t rate() const { return ms == 0 ? 0 : hashes * 1000 / ms; }

    std::vector<uint64_t> minersHashes;
    std::vector<HwMonitor> minerMonitors;
    uint64_t minerRate(const uint64_t hashCount) const { return ms == 0 ? 0 : hashCount * 1000 / ms; }
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

// Holds settings for CPU Miner
struct CPSettings : public MinerSettings
{
<<<<<<< HEAD
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
            _ret.append(" " + boost::str(boost::format("%0.2f") % powerW) + "W");
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

    int cpCpuNumer;   // For CPU
};

struct HwMonitorInfo
{
    HwMonitorInfoType deviceType = HwMonitorInfoType::UNKNOWN;
    string devicePciId;
    int deviceIndex = -1;
=======
    float mh = _p.rate() / 1000000.0f;
    _out << "Speed "
         << EthTealBold << std::fixed << std::setw(6) << std::setprecision(2) << mh << EthReset
         << " Mh/s    ";

    for (size_t i = 0; i < _p.minersHashes.size(); ++i)
    {
        mh = _p.minerRate(_p.minersHashes[i]) / 1000000.0f;
        _out << "gpu/" << i << " " << EthTeal << std::fixed << std::setw(5) << std::setprecision(2) << mh << EthReset;
        if (_p.minerMonitors.size() == _p.minersHashes.size())
            _out << " " << EthTeal << _p.minerMonitors[i] << EthReset;
        _out << "  ";
    }

    return _out;
}

class SolutionStats {
public:
    void accepted() { accepts++;  }
    void rejected() { rejects++;  }
    void failed()   { failures++; }

    void acceptedStale() { acceptedStales++; }
    void rejectedStale() { rejectedStales++; }

    void reset() { accepts = rejects = failures = acceptedStales = rejectedStales = 0; }

    unsigned getAccepts()           { return accepts; }
    unsigned getRejects()           { return rejects; }
    unsigned getFailures()          { return failures; }
    unsigned getAcceptedStales()    { return acceptedStales; }
    unsigned getRejectedStales()    { return rejectedStales; }
private:
    unsigned accepts  = 0;
    unsigned rejects  = 0;
    unsigned failures = 0;

    unsigned acceptedStales = 0;
    unsigned rejectedStales = 0;
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

/// Pause mining
enum MinerPauseEnum
{
<<<<<<< HEAD
    PauseDueToOverHeating,
    PauseDueToAPIRequest,
    PauseDueToFarmPaused,
    PauseDueToInsufficientMemory,
    PauseDueToInitEpochError,
    Pause_MAX  // Must always be last as a placeholder of max count
};
=======
    return os << "[A" << s.getAccepts() << "+" << s.getAcceptedStales() << ":R" << s.getRejects() << "+" << s.getRejectedStales() << ":F" << s.getFailures() << "]";
}
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

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
             << suffixes[magnitude] << EthReset << " - ";

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
                _ret << ", ";
        }

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
<<<<<<< HEAD
    FarmFace() { m_this = this; }
    static FarmFace& f() { return *m_this; };

    virtual ~FarmFace() = default;
    virtual unsigned get_tstart() = 0;
    virtual unsigned get_tstop() = 0;
    virtual unsigned get_ergodicity() = 0;
=======
    virtual ~FarmFace() = default;
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

    /**
     * @brief Called from a Miner to note a WorkPackage has a solution.
     * @param _p The solution.
     * @return true iff the solution was good (implying that mining should be .
     */
    virtual void submitProof(Solution const& _p) = 0;
<<<<<<< HEAD
    virtual void accountSolution(unsigned _minerIdx, SolutionAccountingEnum _accounting) = 0;
    virtual uint64_t get_nonce_scrambler() = 0;
    virtual unsigned get_segment_width() = 0;

private:
    static FarmFace* m_this;
=======
    virtual void failedSolution() = 0;
    virtual uint64_t get_nonce_scrambler() = 0;
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

/**
 * @brief A miner - a member and adoptee of the Farm.
 * @warning Not threadsafe. It is assumed Farm will synchronise calls to/from this class.
 */

class Miner : public Worker
{
public:
    Miner(std::string const& _name, unsigned _index)
      : Worker(_name + std::to_string(_index)), m_index(_index)
    {}

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

    /**
     * @brief Kick an asleep miner.
     */
    virtual void kick_miner() = 0;

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

    void TriggerHashRateUpdate() noexcept;

<<<<<<< HEAD
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
    virtual bool initEpoch_internal() = 0;

    /**
     * @brief Returns current workpackage this miner is working on
     */
    WorkPackage work() const;

    void updateHashRate(uint32_t _groupSize, uint32_t _increment) noexcept;

    static unsigned s_minersCount;   // Total Number of Miners
    static unsigned s_dagLoadMode;   // Way dag should be loaded
    static unsigned s_dagLoadIndex;  // In case of serialized load of dag this is the index of miner
                                     // which should load next
=======
    Miner(std::string const& _name, FarmFace& _farm, size_t _index):
        Worker(_name + std::to_string(_index)),
        index(_index),
        farm(_farm)
    {}

    virtual ~Miner() = default;

    void setWork(WorkPackage const& _work)
    {
        {
            Guard l(x_work);
            m_work = _work;
            workSwitchStart = std::chrono::high_resolution_clock::now();
        }
        kick_miner();
    }

    uint64_t hashCount() const { return m_hashCount.load(std::memory_order_relaxed); }

    void resetHashCount() { m_hashCount.store(0, std::memory_order_relaxed); }

    unsigned Index() { return index; };
    HwMonitorInfo hwmonInfo() { return m_hwmoninfo; }

    uint64_t get_start_nonce()
    {
        // Each GPU is given a non-overlapping 2^40 range to search
        return farm.get_nonce_scrambler() + ((uint64_t) index << 40);
    }
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c

    const unsigned m_index = 0;           // Ordinal index of the Instance (not the device)
    DeviceDescriptor m_deviceDescriptor;  // Info about the device

<<<<<<< HEAD
    EpochContext m_epochContext;

#ifdef DEV_BUILD
    std::chrono::steady_clock::time_point m_workSwitchStart;
#endif

    HwMonitorInfo m_hwmoninfo;
    mutable boost::mutex x_work;
    mutable boost::mutex x_pause;
    boost::condition_variable m_new_work_signal;
    boost::condition_variable m_dag_loaded_signal;

private:
    bitset<MinerPauseEnum::Pause_MAX> m_pauseFlags;

    WorkPackage m_work;

    std::chrono::steady_clock::time_point m_hashTime = std::chrono::steady_clock::now();
    std::atomic<float> m_hashRate = {0.0};
    uint64_t m_groupCount = 0;
    atomic<bool> m_hashRateUpdate = {false};
=======
    /**
     * @brief No work left to be done. Pause until told to kickOff().
     */
    virtual void kick_miner() = 0;

    WorkPackage work() const { Guard l(x_work); return m_work; }

    void addHashCount(uint64_t _n) { m_hashCount.fetch_add(_n, std::memory_order_relaxed); }

    static unsigned s_dagLoadMode;
    static unsigned s_dagLoadIndex;
    static unsigned s_dagCreateDevice;
    static uint8_t* s_dagInHostMemory;
    static bool s_exit;
    static bool s_noeval;

    const size_t index = 0;
    FarmFace& farm;
    std::chrono::high_resolution_clock::time_point workSwitchStart;
    HwMonitorInfo m_hwmoninfo;
private:
    std::atomic<uint64_t> m_hashCount = {0};

    WorkPackage m_work;
    mutable Mutex x_work;
>>>>>>> d0edd204915db4bedfa757d0ca9e1e734619688c
};

}  // namespace eth
}  // namespace dev
