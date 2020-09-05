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

#include <CLI/CLI.hpp>

#include <ethminer/buildinfo.h>
#include <condition_variable>

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#include <libethcore/Farm.h>
#if ETH_ETHASHCL
#include <libethash-cl/CLMiner.h>
#endif
#if ETH_ETHASHCUDA
#include <libethash-cuda/CUDAMiner.h>
#endif
#if ETH_ETHASHCPU
#include <libethash-cpu/CPUMiner.h>
#endif
#include <libpoolprotocols/PoolManager.h>

#if API_CORE
#include <libapicore/ApiServer.h>
#include <regex>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#elif defined(_WIN32)
#include <Windows.h>
#endif

using namespace std;
using namespace dev;
using namespace dev::eth;


// Global vars
bool g_running = false;
bool g_exitOnError = false;  // Whether or not ethminer should exit on mining threads errors

condition_variable g_shouldstop;
boost::asio::io_service g_io_service;  // The IO service itself

struct MiningChannel : public LogChannel
{
    static const char* name() { return EthGreen " m"; }
    static const int verbosity = 2;
};

#define minelog clog(MiningChannel)

#if ETH_DBUS
#include <ethminer/DBusInt.h>
#endif

class MinerCLI
{
public:
    enum class OperationMode
    {
        None,
        Simulation,
        Mining
    };

    MinerCLI() : m_cliDisplayTimer(g_io_service), m_io_strand(g_io_service)
    {
        // Initialize display timer as sleeper
        m_cliDisplayTimer.expires_from_now(boost::posix_time::pos_infin);
        m_cliDisplayTimer.async_wait(m_io_strand.wrap(boost::bind(
            &MinerCLI::cliDisplayInterval_elapsed, this, boost::asio::placeholders::error)));

        // Start io_service in it's own thread
        m_io_thread = std::thread{boost::bind(&boost::asio::io_service::run, &g_io_service)};

        // Io service is now live and running
        // All components using io_service should post to reference of g_io_service
        // and should not start/stop or even join threads (which heavily time consuming)
    }

    virtual ~MinerCLI()
    {
        m_cliDisplayTimer.cancel();
        g_io_service.stop();
        m_io_thread.join();
    }

    void cliDisplayInterval_elapsed(const boost::system::error_code& ec)
    {
        if (!ec && g_running)
        {
            string logLine =
                PoolManager::p().isConnected() ? Farm::f().Telemetry().str() : "Not connected";
            minelog << logLine;

#if ETH_DBUS
            dbusint.send(Farm::f().Telemetry().str().c_str());
#endif
            // Resubmit timer
            m_cliDisplayTimer.expires_from_now(boost::posix_time::seconds(m_cliDisplayInterval));
            m_cliDisplayTimer.async_wait(m_io_strand.wrap(boost::bind(
                &MinerCLI::cliDisplayInterval_elapsed, this, boost::asio::placeholders::error)));
        }
    }

    static void signalHandler(int sig)
    {
        dev::setThreadName("main");

        switch (sig)
        {
#if defined(__linux__) || defined(__APPLE__)
#define BACKTRACE_MAX_FRAMES 100
        case SIGSEGV:
            static bool in_handler = false;
            if (!in_handler)
            {
                int j, nptrs;
                void* buffer[BACKTRACE_MAX_FRAMES];
                char** symbols;

                in_handler = true;

                dev::setThreadName("main");
                cerr << "SIGSEGV encountered ...\n";
                cerr << "stack trace:\n";

                nptrs = backtrace(buffer, BACKTRACE_MAX_FRAMES);
                cerr << "backtrace() returned " << nptrs << " addresses\n";

                symbols = backtrace_symbols(buffer, nptrs);
                if (symbols == NULL)
                {
                    perror("backtrace_symbols()");
                    exit(EXIT_FAILURE);  // Also exit 128 ??
                }
                for (j = 0; j < nptrs; j++)
                    cerr << symbols[j] << "\n";
                free(symbols);

                in_handler = false;
            }
            exit(128);
#undef BACKTRACE_MAX_FRAMES
#endif
        case (999U):
            // Compiler complains about the lack of
            // a case statement in Windows
            // this makes it happy.
            break;
        default:
            cnote << "Got interrupt ...";
            g_running = false;
            g_shouldstop.notify_all();
            break;
        }
    }

#if API_CORE

    static void ParseBind(
        const std::string& inaddr, std::string& outaddr, int& outport, bool advertise_negative_port)
    {
        std::regex pattern("([\\da-fA-F\\.\\:]*)\\:([\\d\\-]*)");
        std::smatch matches;

        if (std::regex_match(inaddr, matches, pattern))
        {
            // Validate Ip address
            boost::system::error_code ec;
            outaddr = boost::asio::ip::address::from_string(matches[1], ec).to_string();
            if (ec)
                throw std::invalid_argument("Invalid Ip Address");

            // Parse port ( Let exception throw )
            outport = std::stoi(matches[2]);
            if (advertise_negative_port)
            {
                if (outport < -65535 || outport > 65535 || outport == 0)
                    throw std::invalid_argument(
                        "Invalid port number. Allowed non zero values in range [-65535 .. 65535]");
            }
            else
            {
                if (outport < 1 || outport > 65535)
                    throw std::invalid_argument(
                        "Invalid port number. Allowed non zero values in range [1 .. 65535]");
            }
        }
        else
        {
            throw std::invalid_argument("Invalid syntax");
        }
    }
#endif
    bool validateArgs(int argc, char** argv)
    {
        std::queue<string> warnings;

        CLI::App app("Ethminer - GPU Ethash miner");

        bool bhelp = false;
        string shelpExt;

        app.set_help_flag();
        app.add_flag("-h,--help", bhelp, "Show help");

        app.add_set("-H,--help-ext", shelpExt,
            {
                "con", "test",
#if ETH_ETHASHCL
                    "cl",
#endif
#if ETH_ETHASHCUDA
                    "cu",
#endif
#if ETH_ETHASHCPU
                    "cp",
#endif
#if API_CORE
                    "api",
#endif
                    "misc", "env"
            },
            "", true);

        bool version = false;

        app.add_option("--ergodicity", m_FarmSettings.ergodicity, "", true)->check(CLI::Range(0, 2));

        app.add_flag("-V,--version", version, "Show program version");

        app.add_option("-v,--verbosity", g_logOptions, "", true)->check(CLI::Range(LOG_NEXT - 1));

        app.add_option("--farm-recheck", m_PoolSettings.getWorkPollInterval, "", true)->check(CLI::Range(1, 99999));

        app.add_option("--farm-retries", m_PoolSettings.connectionMaxRetries, "", true)->check(CLI::Range(0, 99999));

        app.add_option("--retry-delay", m_PoolSettings.delayBeforeRetry, "", true)
            ->check(CLI::Range(1, 999));
        
        app.add_option("--work-timeout", m_PoolSettings.noWorkTimeout, "", true)
            ->check(CLI::Range(180, 99999));

        app.add_option("--response-timeout", m_PoolSettings.noResponseTimeout, "", true)
            ->check(CLI::Range(2, 999));

        app.add_flag("-R,--report-hashrate,--report-hr", m_PoolSettings.reportHashrate, "");

        app.add_option("--display-interval", m_cliDisplayInterval, "", true)
            ->check(CLI::Range(1, 1800));

        app.add_option("--HWMON", m_FarmSettings.hwMon, "", true)->check(CLI::Range(0, 2));

        app.add_flag("--exit", g_exitOnError, "");

        vector<string> pools;
        app.add_option("-P,--pool", pools, "");

        app.add_option("--failover-timeout", m_PoolSettings.poolFailoverTimeout, "", true)
            ->check(CLI::Range(0, 999));

        app.add_flag("--nocolor", g_logNoColor, "");

        app.add_flag("--syslog", g_logSyslog, "");

        app.add_flag("--stdout", g_logStdout, "");

#if API_CORE

        app.add_option("--api-bind", m_api_bind, "", true)
            ->check([this](const string& bind_arg) -> string {
                try
                {
                    MinerCLI::ParseBind(bind_arg, this->m_api_address, this->m_api_port, true);
                }
                catch (const std::exception& ex)
                {
                    throw CLI::ValidationError("--api-bind", ex.what());
                }
                // not sure what to return, and the documentation doesn't say either.
                // https://github.com/CLIUtils/CLI11/issues/144
                return string("");
            });

        app.add_option("--api-port", m_api_port, "", true)->check(CLI::Range(-65535, 65535));

        app.add_option("--api-password", m_api_password, "");

#endif

#if ETH_ETHASHCL || ETH_ETHASHCUDA || ETH_ETHASH_CPU

        app.add_flag("--list-devices", m_shouldListDevices, "");

#endif

#if ETH_ETHASHCL

        app.add_option("--opencl-device,--opencl-devices,--cl-devices", m_CLSettings.devices, "");

        app.add_option("--cl-global-work", m_CLSettings.globalWorkSize, "", true);

        app.add_set("--cl-local-work", m_CLSettings.localWorkSize, {64, 128, 256}, "", true);

        app.add_flag("--cl-nobin", m_CLSettings.noBinary, "");

        app.add_flag("--cl-noexit", m_CLSettings.noExit, "");

#endif

#if ETH_ETHASHCUDA

        app.add_option("--cuda-devices,--cu-devices", m_CUSettings.devices, "");

        app.add_option("--cuda-grid-size,--cu-grid-size", m_CUSettings.gridSize, "", true)
            ->check(CLI::Range(1, 131072));

        app.add_set(
            "--cuda-block-size,--cu-block-size", m_CUSettings.blockSize, {32, 64, 128, 256}, "", true);

        string sched = "sync";
        app.add_set(
            "--cuda-schedule,--cu-schedule", sched, {"auto", "spin", "yield", "sync"}, "", true);

        app.add_option("--cuda-streams,--cu-streams", m_CUSettings.streams, "", true)
            ->check(CLI::Range(1, 99));

#endif

#if ETH_ETHASHCPU

        app.add_option("--cpu-devices,--cp-devices", m_CPSettings.devices, "");

#endif

        app.add_flag("--noeval", m_FarmSettings.noEval, "");

        app.add_option("-L,--dag-load-mode", m_FarmSettings.dagLoadMode, "", true)->check(CLI::Range(1));

        bool cl_miner = false;
        app.add_flag("-G,--opencl", cl_miner, "");

        bool cuda_miner = false;
        app.add_flag("-U,--cuda", cuda_miner, "");

        bool cpu_miner = false;
#if ETH_ETHASHCPU
        app.add_flag("--cpu", cpu_miner, "");
#endif
        auto sim_opt = app.add_option("-Z,--simulation,-M,--benchmark", m_PoolSettings.benchmarkBlock, "", true);

        app.add_option("--tstop", m_FarmSettings.tempStop, "", true)->check(CLI::Range(30, 100));
        app.add_option("--tstart", m_FarmSettings.tempStart, "", true)->check(CLI::Range(30, 100));


        // Exception handling is held at higher level
        app.parse(argc, argv);
        if (bhelp)
        {
            help();
            return false;
        }
        else if (!shelpExt.empty())
        {
            helpExt(shelpExt);
            return false;
        }
        else if (version)
        {
            return false;
        }


#ifndef DEV_BUILD

        if (g_logOptions & LOG_CONNECT)
            warnings.push("Socket connections won't be logged. Compile with -DDEVBUILD=ON");
        if (g_logOptions & LOG_SWITCH)
            warnings.push("Job switch timings won't be logged. Compile with -DDEVBUILD=ON");
        if (g_logOptions & LOG_SUBMIT)
            warnings.push(
                "Solution internal submission timings won't be logged. Compile with -DDEVBUILD=ON");
        if (g_logOptions & LOG_PROGRAMFLOW)
            warnings.push("Program flow won't be logged. Compile with -DDEVBUILD=ON");

#endif


        if (cl_miner)
            m_minerType = MinerType::CL;
        else if (cuda_miner)
            m_minerType = MinerType::CUDA;
        else if (cpu_miner)
            m_minerType = MinerType::CPU;
        else
            m_minerType = MinerType::Mixed;

        /*
            Operation mode Simulation do not require pool definitions
            Operation mode Stratum or GetWork do need at least one
        */

        if (sim_opt->count())
        {
            m_mode = OperationMode::Simulation;
            pools.clear();
            m_PoolSettings.connections.push_back(
                std::shared_ptr<URI>(new URI("simulation://localhost:0", true)));
        }
        else
        {
            m_mode = OperationMode::Mining;
        }

        if (!m_shouldListDevices && m_mode != OperationMode::Simulation)
        {
            if (!pools.size())
                throw std::invalid_argument(
                    "At least one pool definition required. See -P argument.");

            for (size_t i = 0; i < pools.size(); i++)
            {
                std::string url = pools.at(i);
                if (url == "exit")
                {
                    if (i == 0)
                        throw std::invalid_argument(
                            "'exit' failover directive can't be the first in -P arguments list.");
                    else
                        url = "stratum+tcp://-:x@exit:0";
                }

                try
                {
                    std::shared_ptr<URI> uri = std::shared_ptr<URI>(new URI(url));
                    if (uri->SecLevel() != dev::SecureLevel::NONE &&
                        uri->HostNameType() != dev::UriHostNameType::Dns && !getenv("SSL_NOVERIFY"))
                    {
                        warnings.push(
                            "You have specified host " + uri->Host() + " with encryption enabled.");
                        warnings.push("Certificate validation will likely fail");
                    }
                    m_PoolSettings.connections.push_back(uri);
                }
                catch (const std::exception& _ex)
                {
                    string what = _ex.what();
                    throw std::runtime_error("Bad URI : " + what);
                }
            }
        }


#if ETH_ETHASHCUDA
        if (sched == "auto")
            m_CUSettings.schedule = 0;
        else if (sched == "spin")
            m_CUSettings.schedule = 1;
        else if (sched == "yield")
            m_CUSettings.schedule = 2;
        else if (sched == "sync")
            m_CUSettings.schedule = 4;
#endif

        if (m_FarmSettings.tempStop)
        {
            // If temp threshold set HWMON at least to 1
            m_FarmSettings.hwMon = std::max((unsigned int)m_FarmSettings.hwMon, 1U);
            if (m_FarmSettings.tempStop <= m_FarmSettings.tempStart)
            {
                std::string what = "-tstop must be greater than -tstart";
                throw std::invalid_argument(what);
            }
        }

        // Output warnings if any
        if (warnings.size())
        {
            while (warnings.size())
            {
                cout << warnings.front() << endl;
                warnings.pop();
            }
            cout << endl;
        }
        return true;
    }

    void execute()
    {
#if ETH_ETHASHCL
        if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
            CLMiner::enumDevices(m_DevicesCollection);
#endif
#if ETH_ETHASHCUDA
        if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
            CUDAMiner::enumDevices(m_DevicesCollection);
#endif
#if ETH_ETHASHCPU
        if (m_minerType == MinerType::CPU)
            CPUMiner::enumDevices(m_DevicesCollection);
#endif

        // Can't proceed without any GPU
        if (!m_DevicesCollection.size())
            throw std::runtime_error("No usable mining devices found");

        // If requested list detected devices and exit
        if (m_shouldListDevices)
        {
            cout << setw(4) << " Id ";
            cout << setiosflags(ios::left) << setw(10) << "Pci Id    ";
            cout << setw(5) << "Type ";
            cout << setw(30) << "Name                          ";

#if ETH_ETHASHCUDA
            if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
            {
                cout << setw(5) << "CUDA ";
                cout << setw(4) << "SM  ";
            }
#endif
#if ETH_ETHASHCL
            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                cout << setw(5) << "CL   ";
#endif
            cout << resetiosflags(ios::left) << setw(13) << "Total Memory"
                 << " ";
#if ETH_ETHASHCL
            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
            {
                cout << resetiosflags(ios::left) << setw(13) << "Cl Max Alloc"
                     << " ";
                cout << resetiosflags(ios::left) << setw(13) << "Cl Max W.Grp"
                     << " ";
            }
#endif

            cout << resetiosflags(ios::left) << endl;
            cout << setw(4) << "--- ";
            cout << setiosflags(ios::left) << setw(10) << "--------- ";
            cout << setw(5) << "---- ";
            cout << setw(30) << "----------------------------- ";

#if ETH_ETHASHCUDA
            if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
            {
                cout << setw(5) << "---- ";
                cout << setw(4) << "--- ";
            }
#endif
#if ETH_ETHASHCL
            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                cout << setw(5) << "---- ";
#endif
            cout << resetiosflags(ios::left) << setw(13) << "------------"
                 << " ";
#if ETH_ETHASHCL
            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
            {
                cout << resetiosflags(ios::left) << setw(13) << "------------"
                     << " ";
                cout << resetiosflags(ios::left) << setw(13) << "------------"
                     << " ";
            }
#endif
            cout << resetiosflags(ios::left) << endl;
            std::map<string, DeviceDescriptor>::iterator it = m_DevicesCollection.begin();
            while (it != m_DevicesCollection.end())
            {
                auto i = std::distance(m_DevicesCollection.begin(), it);
                cout << setw(3) << i << " ";
                cout << setiosflags(ios::left) << setw(10) << it->first;
                cout << setw(5);
                switch (it->second.type)
                {
                case DeviceTypeEnum::Cpu:
                    cout << "Cpu";
                    break;
                case DeviceTypeEnum::Gpu:
                    cout << "Gpu";
                    break;
                case DeviceTypeEnum::Accelerator:
                    cout << "Acc";
                    break;
                default:
                    break;
                }
                cout << setw(30) << (it->second.name).substr(0, 28);
#if ETH_ETHASHCUDA
                if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
                {
                    cout << setw(5) << (it->second.cuDetected ? "Yes" : "");
                    cout << setw(4) << it->second.cuCompute;
                }
#endif
#if ETH_ETHASHCL
                if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                    cout << setw(5) << (it->second.clDetected ? "Yes" : "");
#endif
                cout << resetiosflags(ios::left) << setw(13)
                     << getFormattedMemory((double)it->second.totalMemory) << " ";
#if ETH_ETHASHCL
                if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                {
                    cout << resetiosflags(ios::left) << setw(13)
                         << getFormattedMemory((double)it->second.clMaxMemAlloc) << " ";
                    cout << resetiosflags(ios::left) << setw(13)
                         << getFormattedMemory((double)it->second.clMaxWorkGroup) << " ";
                }
#endif
                cout << resetiosflags(ios::left) << endl;
                it++;
            }

            return;
        }

        // Subscribe devices with appropriate Miner Type
        // Use CUDA first when available then, as second, OpenCL

        // Apply discrete subscriptions (if any)
#if ETH_ETHASHCUDA
        if (m_CUSettings.devices.size() &&
            (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed))
        {
            for (auto index : m_CUSettings.devices)
            {
                if (index < m_DevicesCollection.size())
                {
                    auto it = m_DevicesCollection.begin();
                    std::advance(it, index);
                    if (!it->second.cuDetected)
                        throw std::runtime_error("Can't CUDA subscribe a non-CUDA device.");
                    it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cuda;
                }
            }
        }
#endif
#if ETH_ETHASHCL
        if (m_CLSettings.devices.size() &&
            (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed))
        {
            for (auto index : m_CLSettings.devices)
            {
                if (index < m_DevicesCollection.size())
                {
                    auto it = m_DevicesCollection.begin();
                    std::advance(it, index);
                    if (!it->second.clDetected)
                        throw std::runtime_error("Can't OpenCL subscribe a non-OpenCL device.");
                    if (it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
                        throw std::runtime_error(
                            "Can't OpenCL subscribe a CUDA subscribed device.");
                    it->second.subscriptionType = DeviceSubscriptionTypeEnum::OpenCL;
                }
            }
        }
#endif
#if ETH_ETHASHCPU
        if (m_CPSettings.devices.size() && (m_minerType == MinerType::CPU))
        {
            for (auto index : m_CPSettings.devices)
            {
                if (index < m_DevicesCollection.size())
                {
                    auto it = m_DevicesCollection.begin();
                    std::advance(it, index);
                    it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cpu;
                }
            }
        }
#endif


        // Subscribe all detected devices
#if ETH_ETHASHCUDA
        if (!m_CUSettings.devices.size() &&
            (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed))
        {
            for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
            {
                if (!it->second.cuDetected ||
                    it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
                    continue;
                it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cuda;
            }
        }
#endif
#if ETH_ETHASHCL
        if (!m_CLSettings.devices.size() &&
            (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed))
        {
            for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
            {
                if (!it->second.clDetected ||
                    it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
                    continue;
                it->second.subscriptionType = DeviceSubscriptionTypeEnum::OpenCL;
            }
        }
#endif
#if ETH_ETHASHCPU
        if (!m_CPSettings.devices.size() &&
            (m_minerType == MinerType::CPU))
        {
            for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
            {
                it->second.subscriptionType = DeviceSubscriptionTypeEnum::Cpu;
            }
        }
#endif
        // Count of subscribed devices
        int subscribedDevices = 0;
        for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
        {
            if (it->second.subscriptionType != DeviceSubscriptionTypeEnum::None)
                subscribedDevices++;
        }

        // If no OpenCL and/or CUDA devices subscribed then throw error
        if (!subscribedDevices)
            throw std::runtime_error("No mining device selected. Aborting ...");

        // Enable
        g_running = true;

        // Signal traps
#if defined(__linux__) || defined(__APPLE__)
        signal(SIGSEGV, MinerCLI::signalHandler);
#endif
        signal(SIGINT, MinerCLI::signalHandler);
        signal(SIGTERM, MinerCLI::signalHandler);

        // Initialize Farm
        new Farm(m_DevicesCollection, m_FarmSettings, m_CUSettings, m_CLSettings, m_CPSettings);

        // Run Miner
        doMiner();
    }

    void help()
    {
        cout << "Ethminer - GPU ethash miner" << endl
             << "minimal usage : ethminer [DEVICES_TYPE] [OPTIONS] -P... [-P...]" << endl
             << endl
             << "Devices type options :" << endl
             << endl
             << "    By default ethminer will try to use all devices types" << endl
             << "    it can detect. Optionally you can limit this behavior" << endl
             << "    setting either of the following options" << endl
#if ETH_ETHASHCL
             << "    -G,--opencl         Mine/Benchmark using OpenCL only" << endl
#endif
#if ETH_ETHASHCUDA
             << "    -U,--cuda           Mine/Benchmark using CUDA only" << endl
#endif
#if ETH_ETHASHCPU
             << "    --cpu               Development ONLY ! (NO MINING)" << endl
#endif
             << endl
             << "Connection options :" << endl
             << endl
             << "    -P,--pool           Stratum pool or http (getWork) connection as URL" << endl
             << "                        "
                "scheme://[user[.workername][:password]@]hostname:port[/...]"
             << endl
             << "                        For an explication and some samples about" << endl
             << "                        how to fill in this value please use" << endl
             << "                        ethminer --help-ext con" << endl
             << endl

             << "Common Options :" << endl
             << endl
             << "    -h,--help           Displays this help text and exits" << endl
             << "    -H,--help-ext       TEXT {'con','test',"
#if ETH_ETHASHCL
             << "cl,"
#endif
#if ETH_ETHASHCUDA
             << "cu,"
#endif
#if ETH_ETHASHCPU
             << "cp,"
#endif
#if API_CORE
             << "api,"
#endif
             << "'misc','env'}" << endl
             << "                        Display help text about one of these contexts:" << endl
             << "                        'con'  Connections and their definitions" << endl
             << "                        'test' Benchmark/Simulation options" << endl
#if ETH_ETHASHCL
             << "                        'cl'   Extended OpenCL options" << endl
#endif
#if ETH_ETHASHCUDA
             << "                        'cu'   Extended CUDA options" << endl
#endif
#if ETH_ETHASHCPU
             << "                        'cp'   Extended CPU options" << endl
#endif
#if API_CORE
             << "                        'api'  API and Http monitoring interface" << endl
#endif
             << "                        'misc' Other miscellaneous options" << endl
             << "                        'env'  Using environment variables" << endl
             << "    -V,--version        Show program version and exits" << endl
             << endl;
    }

    void helpExt(std::string ctx)
    {
        // Help text for benchmarking options
        if (ctx == "test")
        {
            cout << "Benchmarking / Simulation options :" << endl
                 << endl
                 << "    When playing with benchmark or simulation no connection specification "
                    "is"
                 << endl
                 << "    needed ie. you can omit any -P argument." << endl
                 << endl
                 << "    -M,--benchmark      UINT [0 ..] Default not set" << endl
                 << "                        Mining test. Used to test hashing speed." << endl
                 << "                        Specify the block number to test on." << endl
                 << endl
                 << "    -Z,--simulation     UINT [0 ..] Default not set" << endl
                 << "                        Mining test. Used to test hashing speed." << endl
                 << "                        Specify the block number to test on." << endl
                 << endl;
        }

        // Help text for API interfaces options
        if (ctx == "api")
        {
            cout << "API Interface Options :" << endl
                 << endl
                 << "    Ethminer provide an interface for monitor and or control" << endl
                 << "    Please note that information delivered by API interface" << endl
                 << "    may depend on value of --HWMON" << endl
                 << "    A single endpoint is used to accept both HTTP or plain tcp" << endl
                 << "    requests." << endl
                 << endl
                 << "    --api-bind          TEXT Default not set" << endl
                 << "                        Set the API address:port the miner should listen "
                    "on. "
                 << endl
                 << "                        Use negative port number for readonly mode" << endl
                 << "    --api-port          INT [1 .. 65535] Default not set" << endl
                 << "                        Set the API port, the miner should listen on all "
                    "bound"
                 << endl
                 << "                        addresses. Use negative numbers for readonly mode"
                 << endl
                 << "    --api-password      TEXT Default not set" << endl
                 << "                        Set the password to protect interaction with API "
                    "server. "
                 << endl
                 << "                        If not set, any connection is granted access. " << endl
                 << "                        Be advised passwords are sent unencrypted over "
                    "plain "
                    "TCP!!"
                 << endl;
        }

        if (ctx == "cl")
        {
            cout << "OpenCL Extended Options :" << endl
                 << endl
                 << "    Use this extended OpenCL arguments to fine tune the performance." << endl
                 << "    Be advised default values are best generic findings by developers" << endl
                 << endl
                 << "    --cl-devices        UINT {} Default not set" << endl
                 << "                        Space separated list of device indexes to use" << endl
                 << "                        eg --cl-devices 0 2 3" << endl
                 << "                        If not set all available CL devices will be used"
                 << endl
                 << "    --cl-global-work    UINT Default 65536" << endl
                 << "                        Set the global work size multiplier" << endl
                 << "                        Value will be adjusted to nearest power of 2" << endl
                 << "    --cl-local-work     UINT {64,128,256} Default = 128" << endl
                 << "                        Set the local work size multiplier" << endl
                 << "    --cl-nobin          FLAG" << endl
                 << "                        Use openCL kernel. Do not load binary kernel" << endl
                 << "    --cl-noexit         FLAG" << endl
                 << "                        Don't use fast exit algorithm" << endl;
        }

        if (ctx == "cu")
        {
            cout << "CUDA Extended Options :" << endl
                 << endl
                 << "    Use this extended CUDA arguments to fine tune the performance." << endl
                 << "    Be advised default values are best generic findings by developers" << endl
                 << endl
                 << "    --cu-grid-size      INT [1 .. 131072] Default = 8192" << endl
                 << "                        Set the grid size" << endl
                 << "    --cu-block-size     UINT {32,64,128,256} Default = 128" << endl
                 << "                        Set the block size" << endl
                 << "    --cu-devices        UINT {} Default not set" << endl
                 << "                        Space separated list of device indexes to use" << endl
                 << "                        eg --cu-devices 0 2 3" << endl
                 << "                        If not set all available CUDA devices will be used"
                 << endl
                 << "    --cu-parallel-hash  UINT {1,2,4,8} Default = 4" << endl
                 << "                        Set the number of hashes per kernel" << endl
                 << "    --cu-streams        INT [1 .. 99] Default = 2" << endl
                 << "                        Set the number of streams per GPU" << endl
                 << "    --cu-schedule       TEXT Default = 'sync'" << endl
                 << "                        Set the CUDA scheduler mode. Can be one of" << endl
                 << "                        'auto'  Uses a heuristic based on the number of "
                    "active "
                 << endl
                 << "                                CUDA contexts in the process (C) and the "
                    "number"
                 << endl
                 << "                                of logical processors in the system (P)"
                 << endl
                 << "                                If C > P then 'yield' else 'spin'" << endl
                 << "                        'spin'  Instructs CUDA to actively spin when "
                    "waiting"
                 << endl
                 << "                                for results from the device" << endl
                 << "                        'yield' Instructs CUDA to yield its thread when "
                    "waiting for"
                 << endl
                 << "                                for results from the device" << endl
                 << "                        'sync'  Instructs CUDA to block the CPU thread on "
                    "a "
                 << endl
                 << "                                synchronize primitive when waiting for "
                    "results"
                 << endl
                 << "                                from the device" << endl
                 << endl;
        }

        if (ctx == "cp")
        {
            cout << "CPU Extended Options :" << endl
                 << endl
                 << "    Use this extended CPU arguments"
                 << endl
                 << endl
                 << "    --cp-devices        UINT {} Default not set" << endl
                 << "                        Space separated list of device indexes to use" << endl
                 << "                        eg --cp-devices 0 2 3" << endl
                 << "                        If not set all available CPUs will be used" << endl
                 << endl;
        }

        if (ctx == "misc")
        {
            cout << "Miscellaneous Options :" << endl
                 << endl
                 << "    This set of options is valid for mining mode independently from" << endl
                 << "    OpenCL or CUDA or Mixed mining mode." << endl
                 << endl
                 << "    --display-interval  INT[1 .. 1800] Default = 5" << endl
                 << "                        Statistic display interval in seconds" << endl
                 << "    --farm-recheck      INT[1 .. 99999] Default = 500" << endl
                 << "                        Set polling interval for new work in getWork mode"
                 << endl
                 << "                        Value expressed in milliseconds" << endl
                 << "                        It has no meaning in stratum mode" << endl
                 << "    --farm-retries      INT[1 .. 99999] Default = 3" << endl
                 << "                        Set number of reconnection retries to same pool"
                 << endl
                 << "    --retry-delay       INT[1 .. 999] Default = 0" << endl
                 << "                        Delay in seconds before reconnection retry" << endl
                 << "    --failover-timeout  INT[0 .. ] Default not set" << endl
                 << "                        Sets the number of minutes ethminer can stay" << endl
                 << "                        connected to a fail-over pool before trying to" << endl
                 << "                        reconnect to the primary (the first) connection."
                 << endl
                 << "                        before switching to a fail-over connection" << endl
                 << "    --work-timeout      INT[180 .. 99999] Default = 180" << endl
                 << "                        If no new work received from pool after this" << endl
                 << "                        amount of time the connection is dropped" << endl
                 << "                        Value expressed in seconds." << endl
                 << "    --response-timeout  INT[2 .. 999] Default = 2" << endl
                 << "                        If no response from pool to a stratum message " << endl
                 << "                        after this amount of time the connection is dropped"
                 << endl
                 << "    -R,--report-hr      FLAG Notify pool of effective hashing rate" << endl
                 << "    --HWMON             INT[0 .. 2] Default = 0" << endl
                 << "                        GPU hardware monitoring level. Can be one of:" << endl
                 << "                        0 No monitoring" << endl
                 << "                        1 Monitor temperature and fan percentage" << endl
                 << "                        2 As 1 plus monitor power drain" << endl
                 << "    --exit              FLAG Stop ethminer whenever an error is encountered"
                 << endl
                 << "    --ergodicity        INT[0 .. 2] Default = 0" << endl
                 << "                        Sets how ethminer chooses the nonces segments to"
                 << endl
                 << "                        search on." << endl
                 << "                        0 A search segment is picked at startup" << endl
                 << "                        1 A search segment is picked on every pool "
                    "connection"
                 << endl
                 << "                        2 A search segment is picked on every new job" << endl
                 << endl
                 << "    --nocolor           FLAG Monochrome display log lines" << endl
                 << "    --syslog            FLAG Use syslog appropriate output (drop timestamp "
                    "and"
                 << endl
                 << "                        channel prefix)" << endl
                 << "    --stdout            FLAG Log to stdout instead of stderr" << endl
                 << "    --noeval            FLAG By-pass host software re-evaluation of GPUs"
                 << endl
                 << "                        found nonces. Trims some ms. from submission" << endl
                 << "                        time but it may increase rejected solution rate."
                 << endl
                 << "    --list-devices      FLAG Lists the detected OpenCL/CUDA devices and "
                    "exits"
                 << endl
                 << "                        Must be combined with -G or -U or -X flags" << endl
                 << "    -L,--dag-load-mode  INT[0 .. 1] Default = 0" << endl
                 << "                        Set DAG load mode. Can be one of:" << endl
                 << "                        0 Parallel load mode (each GPU independently)" << endl
                 << "                        1 Sequential load mode (one GPU after another)" << endl
                 << endl
                 << "    --tstart            UINT[30 .. 100] Default = 0" << endl
                 << "                        Suspend mining on GPU which temperature is above"
                 << endl
                 << "                        this threshold. Implies --HWMON 1" << endl
                 << "                        If not set or zero no temp control is performed"
                 << endl
                 << "    --tstop             UINT[30 .. 100] Default = 40" << endl
                 << "                        Resume mining on previously overheated GPU when "
                    "temp"
                 << endl
                 << "                        drops below this threshold. Implies --HWMON 1" << endl
                 << "                        Must be lower than --tstart" << endl
                 << "    -v,--verbosity      INT[0 .. 255] Default = 0 " << endl
                 << "                        Set output verbosity level. Use the sum of :" << endl
                 << "                        1   to log stratum json messages" << endl
                 << "                        2   to log found solutions per GPU" << endl
#ifdef DEV_BUILD
                 << "                        32  to log socket (dis)connections" << endl
                 << "                        64  to log timing of job switches" << endl
                 << "                        128 to log time for solution submission" << endl
                 << "                        256 to log program flow" << endl
#endif
                 << endl;
        }

        if (ctx == "env")
        {
            cout << "Environment variables :" << endl
                 << endl
                 << "    If you need or do feel more comfortable you can set the following" << endl
                 << "    environment variables. Please respect letter casing." << endl
                 << endl
                 << "    NO_COLOR            Set to any value to disable colored output." << endl
                 << "                        Acts the same as --nocolor command line argument"
                 << endl
                 << "    SYSLOG              Set to any value to strip timestamp, colors and "
                    "channel"
                 << endl
                 << "                        from output log." << endl
                 << "                        Acts the same as --syslog command line argument"
                 << endl
#ifndef _WIN32
                 << "    SSL_CERT_FILE       Set to the full path to of your CA certificates "
                    "file"
                 << endl
                 << "                        if it is not in standard path :" << endl
                 << "                        /etc/ssl/certs/ca-certificates.crt." << endl
#endif
                 << "    SSL_NOVERIFY        set to any value to to disable the verification "
                    "chain "
                    "for"
                 << endl
                 << "                        certificates. WARNING ! Disabling certificate "
                    "validation"
                 << endl
                 << "                        declines every security implied in connecting to a "
                    "secured"
                 << endl
                 << "                        SSL/TLS remote endpoint." << endl
                 << "                        USE AT YOU OWN RISK AND ONLY IF YOU KNOW WHAT "
                    "YOU'RE "
                    "DOING"
                 << endl;
        }

        if (ctx == "con")
        {
            cout << "Connections specifications :" << endl
                 << endl
                 << "    Whether you need to connect to a stratum pool or to make use of "
                    "getWork "
                    "polling"
                 << endl
                 << "    mode (generally used to solo mine) you need to specify the connection "
                    "making use"
                 << endl
                 << "    of -P command line argument filling up the URL. The URL is in the form "
                    ":"
                 << endl
                 << "    " << endl
                 << "    scheme://[user[.workername][:password]@]hostname:port[/...]." << endl
                 << "    " << endl
                 << "    where 'scheme' can be any of :" << endl
                 << "    " << endl
                 << "    getwork    for http getWork mode" << endl
                 << "    stratum    for tcp stratum mode" << endl
                 << "    stratums   for tcp encrypted stratum mode" << endl
                 << "    stratumss  for tcp encrypted stratum mode with strong TLS 1.2 "
                    "validation"
                 << endl
                 << endl
                 << "    Example 1: -P getwork://127.0.0.1:8545" << endl
                 << "    Example 2: "
                    "-P stratums://0x012345678901234567890234567890123.miner1@ethermine.org:5555"
                 << endl
                 << "    Example 3: "
                    "-P stratum://0x012345678901234567890234567890123.miner1@nanopool.org:9999/"
                    "john.doe%40gmail.com"
                 << endl
                 << "    Example 4: "
                    "-P stratum://0x012345678901234567890234567890123@nanopool.org:9999/miner1/"
                    "john.doe%40gmail.com"
                 << endl
                 << endl
                 << "    Please note: if your user or worker or password do contain characters"
                 << endl
                 << "    which may impair the correct parsing (namely any of . : @ # ?) you have to"
                 << endl
                 << "    enclose those values in backticks( ` ASCII 096) or Url Encode them" << endl
                 << "    Also note that backtick has a special meaning in *nix environments thus"
                 << endl
                 << "    you need to further escape those backticks with backslash." << endl
                 << endl
                 << "    Example : -P stratums://\\`account.121\\`.miner1:x@ethermine.org:5555"
                 << endl
                 << "    Example : -P stratums://account%2e121.miner1:x@ethermine.org:5555" << endl
                 << "    (In Windows backslashes are not needed)" << endl
                 << endl
                 << endl
                 << "    Common url encoded chars are " << endl
                 << "    . (dot)      %2e" << endl
                 << "    : (column)   %3a" << endl
                 << "    @ (at sign)  %40" << endl
                 << "    ? (question) %3f" << endl
                 << "    # (number)   %23" << endl
                 << "    / (slash)    %2f" << endl
                 << "    + (plus)     %2b" << endl
                 << endl
                 << "    You can add as many -P arguments as you want. Every -P specification"
                 << endl
                 << "    after the first one behaves as fail-over connection. When also the" << endl
                 << "    the fail-over disconnects ethminer passes to the next connection" << endl
                 << "    available and so on till the list is exhausted. At that moment" << endl
                 << "    ethminer restarts the connection cycle from the first one." << endl
                 << "    An exception to this behavior is ruled by the --failover-timeout" << endl
                 << "    command line argument. See 'ethminer -H misc' for details." << endl
                 << endl
                 << "    The special notation '-P exit' stops the failover loop." << endl
                 << "    When ethminer reaches this kind of connection it simply quits." << endl
                 << endl
                 << "    When using stratum mode ethminer tries to auto-detect the correct" << endl
                 << "    flavour provided by the pool. Should be fine in 99% of the cases." << endl
                 << "    Nevertheless you might want to fine tune the stratum flavour by" << endl
                 << "    any of of the following valid schemes :" << endl
                 << endl
                 << "    " << URI::KnownSchemes(ProtocolFamily::STRATUM) << endl
                 << endl
                 << "    where a scheme is made up of two parts, the stratum variant + the tcp "
                    "transport protocol"
                 << endl
                 << endl
                 << "    Stratum variants :" << endl
                 << endl
                 << "        stratum     Stratum" << endl
                 << "        stratum1    Eth Proxy compatible" << endl
                 << "        stratum2    EthereumStratum 1.0.0 (nicehash)" << endl
                 << "        stratum3    EthereumStratum 2.0.0" << endl
                 << endl
                 << "    Transport variants :" << endl
                 << endl
                 << "        tcp         Unencrypted tcp connection" << endl
                 << "        tls         Encrypted tcp connection (including deprecated TLS 1.1)"
                 << endl
                 << "        tls12       Encrypted tcp connection with TLS 1.2" << endl
                 << "        ssl         Encrypted tcp connection with TLS 1.2" << endl
                 << endl;
        }
    }

private:
    void doMiner()
    {

        new PoolManager(m_PoolSettings);
        if (m_mode != OperationMode::Simulation)
            for (auto conn : m_PoolSettings.connections)
                cnote << "Configured pool " << conn->Host() + ":" + to_string(conn->Port());

#if API_CORE

        ApiServer api(m_api_address, m_api_port, m_api_password);
        if (m_api_port)
            api.start();

#endif

        // Start PoolManager
        PoolManager::p().start();

        // Initialize display timer as sleeper with proper interval
        m_cliDisplayTimer.expires_from_now(boost::posix_time::seconds(m_cliDisplayInterval));
        m_cliDisplayTimer.async_wait(m_io_strand.wrap(boost::bind(
            &MinerCLI::cliDisplayInterval_elapsed, this, boost::asio::placeholders::error)));

        // Stay in non-busy wait till signals arrive
        unique_lock<mutex> clilock(m_climtx);
        while (g_running)
            g_shouldstop.wait(clilock);

#if API_CORE

        // Stop Api server
        if (api.isRunning())
            api.stop();

#endif
        if (PoolManager::p().isRunning())
            PoolManager::p().stop();

        cnote << "Terminated!";
        return;
    }

    // Global boost's io_service
    std::thread m_io_thread;                        // The IO service thread
    boost::asio::deadline_timer m_cliDisplayTimer;  // The timer which ticks display lines
    boost::asio::io_service::strand m_io_strand;    // A strand to serialize posts in
                                                    // multithreaded environment

    // Physical Mining Devices descriptor
    std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};

    // Mining options
    MinerType m_minerType = MinerType::Mixed;
    OperationMode m_mode = OperationMode::None;
    bool m_shouldListDevices = false;

    FarmSettings m_FarmSettings;  // Operating settings for Farm
    PoolSettings m_PoolSettings;  // Operating settings for PoolManager
    CLSettings m_CLSettings;          // Operating settings for CL Miners
    CUSettings m_CUSettings;          // Operating settings for CUDA Miners
    CPSettings m_CPSettings;          // Operating settings for CPU Miners

    //// -- Pool manager related params
    //std::vector<std::shared_ptr<URI>> m_poolConns;


    // -- CLI Interface related params
    unsigned m_cliDisplayInterval =
        5;  // Display stats/info on cli interface every this number of seconds

    // -- CLI Flow control
    mutex m_climtx;

#if API_CORE
    // -- API and Http interfaces related params
    string m_api_bind;                  // API interface binding address in form <address>:<port>
    string m_api_address = "0.0.0.0";   // API interface binding address (Default any)
    int m_api_port = 0;                 // API interface binding port
    string m_api_password;              // API interface write protection password
#endif

#if ETH_DBUS
    DBusInt dbusint;
#endif
};

int main(int argc, char** argv)
{
    // Return values
    // 0 - Normal exit
    // 1 - Invalid/Insufficient command line arguments
    // 2 - Runtime error
    // 3 - Other exceptions
    // 4 - Possible corruption

#if defined(_WIN32)
    // Need to change the code page from the default OEM code page (437) so that
    // UTF-8 characters are displayed correctly in the console
    SetConsoleOutputCP(CP_UTF8);
#endif

    // Always out release version
    auto* bi = ethminer_get_buildinfo();
    cout << endl
         << endl
         << "ethminer " << bi->project_version << endl
         << "Build: " << bi->system_name << "/" << bi->build_type << "/" << bi->compiler_id << endl
         << endl;

    if (argc < 2)
    {
        cerr << "No arguments specified. " << endl
             << "Try 'ethminer --help' to get a list of arguments." << endl
             << endl;
        return 1;
    }

    try
    {
        MinerCLI cli;

        try
        {
            // Set env vars controlling GPU driver behavior.
            setenv("GPU_MAX_HEAP_SIZE", "100");
            setenv("GPU_MAX_ALLOC_PERCENT", "100");
            setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

            // Argument validation either throws exception
            // or returns false which means do not continue
            if (!cli.validateArgs(argc, argv))
                return 0;

            if (getenv("SYSLOG"))
                g_logSyslog = true;
            if (g_logSyslog || (getenv("NO_COLOR")))
                g_logNoColor = true;

#if defined(_WIN32)
            if (!g_logNoColor)
            {
                g_logNoColor = true;
                // Set output mode to handle virtual terminal sequences
                // Only works on Windows 10, but most users should use it anyway
                HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
                if (hOut != INVALID_HANDLE_VALUE)
                {
                    DWORD dwMode = 0;
                    if (GetConsoleMode(hOut, &dwMode))
                    {
                        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                        if (SetConsoleMode(hOut, dwMode))
                            g_logNoColor = false;
                    }
                }
            }
#endif

            cli.execute();
            cout << endl << endl;
            return 0;
        }
        catch (std::invalid_argument& ex1)
        {
            cerr << "Error: " << ex1.what() << endl
                 << "Try ethminer --help to get an explained list of arguments." << endl
                 << endl;
            return 1;
        }
        catch (std::runtime_error& ex2)
        {
            cerr << "Error: " << ex2.what() << endl << endl;
            return 2;
        }
        catch (std::exception& ex3)
        {
            cerr << "Error: " << ex3.what() << endl << endl;
            return 3;
        }
        catch (...)
        {
            cerr << "Error: Unknown failure occurred. Possible memory corruption." << endl << endl;
            return 4;
        }
    }
    catch (const std::exception& ex)
    {
        cerr << "Could not initialize CLI interface " << endl
             << "Error: " << ex.what() << endl
             << endl;
        return 4;
    }
}
