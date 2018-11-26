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
#include <libpoolprotocols/PoolManager.h>
#include <libpoolprotocols/getwork/EthGetworkClient.h>
#include <libpoolprotocols/stratum/EthStratumClient.h>
#include <libpoolprotocols/testing/SimulateClient.h>

#if API_CORE
#include <libapicore/ApiServer.h>
#include <libapicore/httpServer.h>
#include <regex>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
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
        Benchmark,
        Simulation,
        Farm,
        Stratum
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
            if (PoolManager::p().isConnected())
            {
                auto solstats = Farm::f().getSolutionStats();
                {
                    ostringstream os;
                    os << Farm::f().miningProgress() << ' ';
                    if (!(g_logOptions & LOG_PER_GPU))
                        os << solstats << ' ';
                    os << Farm::f().farmLaunchedFormatted();
                    minelog << os.str();
                }

                if (g_logOptions & LOG_PER_GPU)
                {
                    ostringstream statdetails;
                    statdetails << "Solutions " << solstats << ' ';
                    for (size_t i = 0; i < Farm::f().getMiners().size(); i++)
                    {
                        if (i)
                            statdetails << " ";
                        statdetails << "gpu" << i << ":" << solstats.getString(i);
                    }
                    minelog << statdetails.str();
                }

#if ETH_DBUS
                dbusint.send(toString(mp).c_str());
#endif
            }
            else
            {
                minelog << "not-connected";
            }


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
#if API_CORE
                    "api",
#endif
                    "misc", "env"
            },
            "", true);

        bool version = false;

        app.add_option("--ergodicity", m_farmErgodicity, "", true)->check(CLI::Range(0, 2));

        app.add_flag("-V,--version", version, "Show program version");

        app.add_option("-v,--verbosity", g_logOptions, "", true)->check(CLI::Range(LOG_NEXT - 1));

        app.add_option("--farm-recheck", m_farmPollInterval, "", true)->check(CLI::Range(1, 99999));

        app.add_option("--farm-retries", m_poolMaxRetries, "", true)->check(CLI::Range(0, 99999));

        app.add_option("--work-timeout", m_poolWorkTimeout, "", true)
            ->check(CLI::Range(180, 99999));

        app.add_option("--response-timeout", m_poolRespTimeout, "", true)
            ->check(CLI::Range(2, 999));

        app.add_flag("-R,--report-hashrate,--report-hr", m_poolHashRate, "");

        app.add_option("--display-interval", m_cliDisplayInterval, "", true)
            ->check(CLI::Range(1, 1800));

        app.add_option("--HWMON", m_farmHwMonitors, "", true)->check(CLI::Range(0, 2));

        app.add_flag("--exit", g_exitOnError, "");

        vector<string> pools;
        app.add_option("-P,--pool,pool", pools, "");

        app.add_option("--failover-timeout", m_poolFlvrTimeout, "", true)
            ->check(CLI::Range(0, 999));

        app.add_flag("--nocolor", g_logNoColor, "");

        app.add_flag("--syslog", g_logSyslog, "");

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

        app.add_option("--http-bind", m_http_bind, "", true)
            ->check([this](const string& bind_arg) -> string {
                int port;
                try
                {
                    MinerCLI::ParseBind(bind_arg, this->m_http_address, port, false);
                }
                catch (const std::exception& ex)
                {
                    throw CLI::ValidationError("--http-bind", ex.what());
                }
                this->m_http_port = static_cast<uint16_t>(port);
                // not sure what to return, and the documentation doesn't say either.
                // https://github.com/CLIUtils/CLI11/issues/144
                return string("");
            });

        app.add_option("--http-port", m_http_port, "", true)->check(CLI::Range(65535));

#endif

#if ETH_ETHASHCL || ETH_ETHASHCUDA

        app.add_flag("--list-devices", m_shouldListDevices, "");

#endif

#if ETH_ETHASHCL

        int clKernel = -1;

        app.add_option("--cl-kernel", clKernel, "", true)->check(CLI::Range(2));

        app.add_option("--opencl-platform", m_oclPlatform, "", true);

        app.add_option("--opencl-device,--opencl-devices,--cl-devices", m_oclDevices, "");

        int openclThreadsPerHash = -1;
        app.add_set("--cl-parallel-hash", openclThreadsPerHash, {1, 2, 4, 8}, "", true);

        app.add_option("--cl-global-work", m_oclGWorkSize, "", true);

        app.add_set("--cl-local-work", m_oclLWorkSize, {64, 128, 192, 256}, "", true);

        app.add_flag("--cl-nobin", m_oclNoBinary, "");

#endif

#if ETH_ETHASHCUDA

        app.add_option("--cuda-grid-size,--cu-grid-size", m_cudaGridSize, "", true)
            ->check(CLI::Range(1, 131072));

        app.add_set(
            "--cuda-block-size,--cu-block-size", m_cudaBlockSize, {32, 64, 128, 256}, "", true);

        app.add_option("--cuda-devices,--cu-devices", m_cudaDevices, "");

        app.add_set(
            "--cuda-parallel-hash,--cu-parallel-hash", m_cudaParallelHash, {1, 2, 4, 8}, "", true);

        string sched = "sync";
        app.add_set(
            "--cuda-schedule,--cu-schedule", sched, {"auto", "spin", "yield", "sync"}, "", true);

        app.add_option("--cuda-streams,--cu-streams", m_cudaStreams, "", true)
            ->check(CLI::Range(1, 99));

#endif

        app.add_flag("--noeval", m_farmNoEval, "");

        app.add_option("-L,--dag-load-mode", m_farmDagLoadMode, "", true)->check(CLI::Range(1));

        app.add_option("--benchmark-warmup", m_benchmarkWarmup, "", true);

        app.add_option("--benchmark-trials", m_benchmarkTrial, "", true)->check(CLI::Range(1, 99));

        bool cl_miner = false;
        app.add_flag("-G,--opencl", cl_miner, "");

        bool cuda_miner = false;
        app.add_flag("-U,--cuda", cuda_miner, "");

        auto bench_opt = app.add_option("-M,--benchmark", m_benchmarkBlock, "", true);
        auto sim_opt = app.add_option("-Z,--simulation", m_benchmarkBlock, "", true);


        app.add_option("--tstop", m_farmTempStop, "", true)->check(CLI::Range(30, 100));
        app.add_option("--tstart", m_farmTempStart, "", true)->check(CLI::Range(30, 100));


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


#if ETH_ETHASHCL
        if (clKernel >= 0)
            warnings.push("--cl-kernel ignored. Kernel is auto-selected");
        if (openclThreadsPerHash >= 0)
            warnings.push("--cl-parallel-hash ignored. No longer applies");
#endif
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
        else
            m_minerType = MinerType::Mixed;

        /*
            Operation mode Benchmark and Simulation do not require pool definitions
            Operation mode Stratum or GetWork do need at least one
        */

        if (bench_opt->count())
        {
            m_mode = OperationMode::Benchmark;
            pools.clear();
        }
        else if (sim_opt->count())
        {
            m_mode = OperationMode::Simulation;
            pools.clear();
            pools.push_back("http://-:0");  // Fake connection
        }
        else if (!m_shouldListDevices)
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
                    if (m_mode == OperationMode::Stratum)
                        url = "stratum+tcp://-:x@exit:0";
                    if (m_mode == OperationMode::Farm)
                        url = "http://-:x@exit:0";
                }

                URI uri(url);

                if (!uri.Valid() || !uri.KnownScheme())
                {
                    std::string what = "Bad URI : " + uri.str();
                    throw std::invalid_argument(what);
                }

                if (uri.SecLevel() != dev::SecureLevel::NONE &&
                    uri.HostNameType() != dev::UriHostNameType::Dns && !getenv("SSL_NOVERIFY"))
                {
                    warnings.push(
                        "You have specified host " + uri.Host() + " with encryption enabled.");
                    warnings.push("Certificate validation will likely fail");
                }

                m_poolConns.push_back(uri);
                OperationMode mode =
                    (uri.Family() == ProtocolFamily::STRATUM ? OperationMode::Stratum :
                                                               OperationMode::Farm);

                if ((m_mode != OperationMode::None) && (m_mode != mode))
                {
                    std::string what = "Mixed stratum and getwork connections not supported.";
                    throw std::invalid_argument(what);
                }
                m_mode = mode;
            }
        }


#if ETH_ETHASHCUDA
        if (sched == "auto")
            m_cudaSchedule = 0;
        else if (sched == "spin")
            m_cudaSchedule = 1;
        else if (sched == "yield")
            m_cudaSchedule = 2;
        else if (sched == "sync")
            m_cudaSchedule = 4;
#endif

        if (m_farmTempStop && (m_farmTempStop <= m_farmTempStart))
        {
            std::string what = "-tstop must be greater than -tstart";
            throw std::invalid_argument(what);
        }
        if (m_farmTempStop && !m_farmHwMonitors)
        {
            // if we want stop mining at a specific temperature, we have to
            // monitor the temperature ==> so auto set HWMON to at least 1.
            m_farmHwMonitors = 1;
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

        // Can't proceed without any GPU
        if (!m_DevicesCollection.size())
            throw std::runtime_error("No usable mining devices found");

        // If requested list detected devices and exit
        if (m_shouldListDevices)
        {
            cout << setw(4) << " Id ";
            cout << setiosflags(ios::left) << setw(10) << "Pci Id    ";
            cout << setw(5) << "Type ";
            cout << setw(26) << "Name                      ";

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
            cout << setw(26) << "------------------------- ";

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
            std::map<string, DeviceDescriptorType>::iterator it = m_DevicesCollection.begin();
            while (it != m_DevicesCollection.end())
            {
                auto i = std::distance(m_DevicesCollection.begin(), it);
                cout << setw(3) << i << " ";
                cout << setiosflags(ios::left) << setw(10) << it->first;
                cout << setw(5);
                switch (it->second.Type)
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
                cout << setw(26) << (it->second.Name).substr(0, 24);
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
                     << FormattedMemSize(it->second.TotalMemory) << " ";
#if ETH_ETHASHCL
                if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                {
                    cout << resetiosflags(ios::left) << setw(13)
                         << FormattedMemSize(it->second.clMaxMemAlloc) << " ";
                    cout << resetiosflags(ios::left) << setw(13)
                         << FormattedMemSize(it->second.clMaxWorkGroup) << " ";
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
        if (m_cudaDevices.size() || m_oclDevices.size())
        {
#if ETH_ETHASHCUDA
            if (m_cudaDevices.size() &&
                (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed))
            {
                for (auto index : m_cudaDevices)
                {
                    if (index < m_DevicesCollection.size())
                    {
                        auto it = m_DevicesCollection.begin();
                        std::advance(it, index);
                        if (!it->second.cuDetected)
                            throw std::runtime_error("Can't CUDA subscribe a non-CUDA device.");
                        it->second.SubscriptionType = DeviceSubscriptionTypeEnum::Cuda;
                    }
                }
            }
#endif
#if ETH_ETHASHCL
            if (m_oclDevices.size() &&
                (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed))
            {
                for (auto index : m_oclDevices)
                {
                    if (index < m_DevicesCollection.size())
                    {
                        auto it = m_DevicesCollection.begin();
                        std::advance(it, index);
                        if (!it->second.clDetected)
                            throw std::runtime_error("Can't OpenCL subscribe a non-OpenCL device.");
                        if (it->second.SubscriptionType != DeviceSubscriptionTypeEnum::None)
                            throw std::runtime_error(
                                "Can't OpenCL subscribe a CUDA subscribed device.");
                        it->second.SubscriptionType = DeviceSubscriptionTypeEnum::OpenCL;
                    }
                }
            }
#endif
        }
        else
        {
            // Subscribe all detected devices
#if ETH_ETHASHCUDA
            if ((m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed))
            {
                for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
                {
                    if (!it->second.cuDetected ||
                        it->second.SubscriptionType != DeviceSubscriptionTypeEnum::None)
                        continue;
                    it->second.SubscriptionType = DeviceSubscriptionTypeEnum::Cuda;
                }
            }
#endif
#if ETH_ETHASHCL
            if ((m_minerType == MinerType::CL || m_minerType == MinerType::Mixed))
            {
                for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
                {
                    if (!it->second.clDetected ||
                        it->second.SubscriptionType != DeviceSubscriptionTypeEnum::None)
                        continue;
                    it->second.SubscriptionType = DeviceSubscriptionTypeEnum::OpenCL;
                }
            }
#endif
        }


        // Count of subscribed devices
        int subscribedDevices = 0;
        for (auto it = m_DevicesCollection.begin(); it != m_DevicesCollection.end(); it++)
        {
            if (it->second.SubscriptionType != DeviceSubscriptionTypeEnum::None)
                    subscribedDevices++;
        }

        // If no OpenCL and/or CUDA devices subscribed then throw error
        if (!subscribedDevices)
            throw std::runtime_error("No mining device selected. Aborting ...");

#if ETH_ETHASHCL
        if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
            CLMiner::configureGPU(m_oclLWorkSize, m_oclGWorkSize, m_farmDagLoadMode, m_oclNoBinary);
#endif
#if ETH_ETHASHCUDA
        if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
            CUDAMiner::configureGPU(m_cudaBlockSize, m_cudaGridSize, m_cudaStreams, m_cudaSchedule,
                m_farmDagLoadMode, m_cudaParallelHash);
#endif

        // Enable
        g_running = true;

        // Signal traps
#if defined(__linux__) || defined(__APPLE__)
        signal(SIGSEGV, MinerCLI::signalHandler);
#endif
        signal(SIGINT, MinerCLI::signalHandler);
        signal(SIGTERM, MinerCLI::signalHandler);

        // Initialize Farm
        new Farm(m_DevicesCollection, m_farmHwMonitors, m_farmNoEval);
        Farm::f().setTStartTStop(m_farmTempStart, m_farmTempStop);

        // Run proper mining mode
        switch (m_mode)
        {
        case OperationMode::Benchmark:
            doBenchmark(m_minerType, m_benchmarkWarmup, m_benchmarkTrial, m_benchmarkTrials);
            break;
        case OperationMode::Farm:
        case OperationMode::Stratum:
        case OperationMode::Simulation:
            doMiner();
            break;
        default:
            // Satisfy the compiler, but cannot happen!
            throw std::runtime_error("Program logic error. Unexpected Operation Mode.");
        }
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
#if API_CORE
             << "api,"
#endif
             << "'misc','env'}" << endl
             << "                        Display help text about one of these contexts:" << endl
             << "                        'con'  Connections and their definitions" << endl
             << "                        'test' Benchmark and simulation options" << endl
#if ETH_ETHASHCL
             << "                        'cl'   Extended OpenCL options" << endl
#endif
#if ETH_ETHASHCUDA
             << "                        'cu'   Extended CUDA options" << endl
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
                 << "    -M,--benchmark      UINT[0 ..] Default not set" << endl
                 << "                        Benchmark mining agains the given block number" << endl
                 << "                        and exits." << endl
                 << "    --benchmark-warmup  UINT Default = 15" << endl
                 << "                        Set duration in seconds of warmup for benchmark"
                 << endl
                 << "                        tests." << endl
                 << "    --benchmark-trials  INT [1 .. 99] Default = 5" << endl
                 << "                        Set the number of benchmark trials to run" << endl
                 << "    -Z,--simulation     UINT [0 ..] Default not set" << endl
                 << "                        Mining test. Used to validate kernel optimizations."
                 << endl
                 << "                        Specify a block number." << endl
                 << endl;
        }

        // Help text for API interfaces options
        if (ctx == "api")
        {
            cout << "API Interface Options :" << endl
                 << endl
                 << "    Ethminer can provide two interfaces for monitor and or control" << endl
                 << "    Please note that information delivered by API and Http interface" << endl
                 << "    may depend on value of --HWMON" << endl
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
                 << endl
                 << "    --http-bind         TEXT Default not set" << endl
                 << "                        Set the http monitoring address:port the miner "
                    "should "
                 << endl
                 << "                        listen on." << endl
                 << "    --api-port          INT [1 .. 65535] Default not set" << endl
                 << "                        Set the http port, the miner should listen on all "
                    "bound"
                 << endl
                 << "                        addresses." << endl
                 << endl;
        }

        if (ctx == "cl")
        {
            cout << "OpenCL Extended Options :" << endl
                 << endl
                 << "    Use this extended OpenCL arguments to fine tune the performance." << endl
                 << "    Be advised default values are best generic findings by developers" << endl
                 << endl
                 << "    --cl-kernel         INT [0 .. 2] Default not set" << endl
                 << "                        Select OpenCL kernel. Ignored since 0.15" << endl
                 << "    --cl-platform       UINT Default 0" << endl
                 << "                        Use OpenCL platform N" << endl
                 << "    --cl-devices        UINT {} Default not set" << endl
                 << "                        Comma separated list of device indexes to use" << endl
                 << "                        eg --cl-devices 0,2,3" << endl
                 << "                        If not set all available CL devices will be used"
                 << endl
                 << "    --cl-parallel-hash  UINT {1,2,4,8}" << endl
                 << "                        Ignored" << endl
                 << "    --cl-global-work    UINT Default 65536" << endl
                 << "                        Set the global work size multiplier" << endl
                 << "    --cl-local-work     UINT {32,64,128,256} Default = 128" << endl
                 << "                        Set the local work size multiplier" << endl
                 << "    --cl-nobin          FLAG" << endl
                 << "                        Use openCL kernel. Do not load binary kernel" << endl
                 << endl;
        }

        if (ctx == "cu")
        {
            cout << "CUDA Extended Options :" << endl
                 << endl
                 << "    Use this extended CUDA arguments to fine tune the performance." << endl
                 << "    Be advised default values are best generic findings by developers	"
                 << endl
                 << endl
                 << "    --cu-grid-size      INT [1 .. 131072] Default = 8192" << endl
                 << "                        Set the grid size" << endl
                 << "    --cu-block-size     UINT {32,64,128,256} Default = 128" << endl
                 << "                        Set the block size" << endl
                 << "    --cu-devices        UINT {} Default not set" << endl
                 << "                        Comma separated list of device indexes to use" << endl
                 << "                        eg --cu-devices 0,2,3" << endl
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
                 << "    --noeval            FLAG By-pass host software re-evaluation of GPUs"
                 << endl
                 << "                        found nonces. Trims some ms. from submission" << endl
                 << "                        time but it may increase rejected solution rate."
                 << endl
                 << "    --list-devices      FLAG Lists the detected OpenCL/CUDA devices and "
                    "exits"
                 << endl
                 << "                        Must be combined with -G or -U or -X flags" << endl
                 << "    -L,--dag-load-mode  INT[0 .. 2] Default = 0" << endl
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
                    "john.doe@gmail.com"
                 << endl
                 << "    Example 4: "
                    "-P stratum://0x012345678901234567890234567890123@nanopool.org:9999/miner1/"
                    "john.doe@gmail.com"
                 << endl
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
                 << "        stratum2    EthereumStratum 1.0.0. (nicehash)" << endl
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
    void doBenchmark(MinerType _m, unsigned _warmupDuration = 15, unsigned _trialDuration = 3,
        unsigned _trials = 5)
    {
        BlockHeader genesis;
        genesis.setNumber(m_benchmarkBlock);
        genesis.setDifficulty(u256(1) << 64);

        string platformInfo =
            (_m == MinerType::CL ? "CL" : (_m == MinerType::CUDA ? "CUDA" : "MIXED"));
        minelog << "Benchmarking on platform: " << platformInfo << " Preparing DAG for block #"
                << m_benchmarkBlock;

        map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
        sealers["opencl"] =
            Farm::SealerDescriptor{[](unsigned _index) { return new CLMiner(_index); }};
#endif
#if ETH_ETHASHCUDA
        sealers["cuda"] =
            Farm::SealerDescriptor{[](unsigned _index) { return new CUDAMiner(_index); }};
#endif

        Farm::f().setSealers(sealers);
        Farm::f().onSolutionFound([&](Solution) { return false; });
        Farm::f().start();

        WorkPackage current = WorkPackage(genesis);


        vector<uint64_t> results;
        results.reserve(_trials);
        uint64_t mean = 0;
        uint64_t innerMean = 0;
        for (unsigned i = 0; i <= _trials; ++i)
        {
            current.header = h256::random();
            current.boundary = genesis.boundary();
            Farm::f().setWork(current);
            if (!i)
                minelog << "Warming up...";
            else
                minelog << "Trial " << i << "... ";

            this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));

            if (!i)
                continue;

            auto rate = uint64_t(Farm::f().miningProgress().hashRate);
            minelog << "Hashes per second " << rate;
            results.push_back(rate);
            mean += uint64_t(rate);
        }
        sort(results.begin(), results.end());
        minelog << "min/mean/max: " << results.front() << "/" << (mean / _trials) << "/"
                << results.back() << " H/s";
        if (results.size() > 2)
        {
            for (auto it = results.begin() + 1; it != results.end() - 1; it++)
                innerMean += *it;
            innerMean /= (_trials - 2);
            minelog << "inner mean: " << innerMean << " H/s";
        }
        else
        {
            minelog << "inner mean: n/a";
        }

        return;
    }

    void doMiner()
    {
        map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
        sealers["opencl"] =
            Farm::SealerDescriptor{[](unsigned _index) { return new CLMiner(_index); }};
#endif
#if ETH_ETHASHCUDA
        sealers["cuda"] =
            Farm::SealerDescriptor{[](unsigned _index) { return new CUDAMiner(_index); }};
#endif

        PoolClient* client = nullptr;

        switch (m_mode)
        {
        case MinerCLI::OperationMode::None:
        case MinerCLI::OperationMode::Benchmark:
            throw std::runtime_error("Program logic error. Unexpected m_mode.");
            break;
        case MinerCLI::OperationMode::Simulation:
            client = new SimulateClient(20, m_benchmarkBlock);
            break;
        case MinerCLI::OperationMode::Farm:
            client = new EthGetworkClient(m_poolWorkTimeout, m_farmPollInterval);
            break;
        case MinerCLI::OperationMode::Stratum:
            client = new EthStratumClient(m_poolWorkTimeout, m_poolRespTimeout);
            break;
        default:
            // Satisfy the compiler, but cannot happen!
            throw std::runtime_error("Program logic error. Unexpected m_mode.");
        }

        Farm::f().setSealers(sealers);

        new PoolManager(
            client, m_poolMaxRetries, m_poolFlvrTimeout, m_farmErgodicity, m_poolHashRate);
        for (auto conn : m_poolConns)
        {
            PoolManager::p().addConnection(conn);
            if (m_mode != OperationMode::Simulation)
                cnote << "Configured pool " << conn.Host() + ":" + to_string(conn.Port());
        }

#if API_CORE

        ApiServer api(m_api_address, m_api_port, m_api_password);
        if (m_api_port)
            api.start();

        http_server.run(m_http_address, m_http_port, m_farmHwMonitors);

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
    std::map<std::string, DeviceDescriptorType> m_DevicesCollection = {};

    // Mining options
    MinerType m_minerType = MinerType::Mixed;
    OperationMode m_mode = OperationMode::None;
    unsigned m_miningThreads = UINT_MAX;  // TODO Safe to remove and replace with constant ?
    bool m_shouldListDevices = false;

#if ETH_ETHASHCL
    // -- OpenCL related params
    unsigned m_oclPlatform = 0;
    vector<unsigned> m_oclDevices;
    unsigned m_oclGWorkSize = CLMiner::c_defaultGlobalWorkSizeMultiplier;
    unsigned m_oclLWorkSize = CLMiner::c_defaultLocalWorkSize;
    bool m_oclNoBinary = false;
#endif

#if ETH_ETHASHCUDA
    // -- CUDA related params
    vector<unsigned> m_cudaDevices;
    unsigned m_cudaStreams = CUDAMiner::c_defaultNumStreams;
    unsigned m_cudaSchedule = 4;  // sync
    unsigned m_cudaGridSize = CUDAMiner::c_defaultGridSize;
    unsigned m_cudaBlockSize = CUDAMiner::c_defaultBlockSize;
    unsigned m_cudaParallelHash = 4;
#endif

    // -- Farm related params
    unsigned m_farmDagLoadMode = 0;  // DAG load mode : 0=parallel, 1=sequential
    bool m_farmNoEval = false;       // Whether or not ethminer should CPU re-evaluate solutions
    unsigned m_farmPollInterval =
        500;  // In getWork mode this establishes the ms. interval to check for new job
    unsigned m_farmHwMonitors =
        0;  // Farm GPU monitoring level : 0 - No monitor; 1 - Temp and Fan; 2 - Temp Fan Power
    unsigned m_farmTempStop = 0;  // Halt mining on GPU if temperature ge this threshold (Celsius)
    unsigned m_farmTempStart =
        40;  // Resume mining on GPU if temperature le this threshold (Celsius)
    unsigned m_farmErgodicity = 0;  // Sets ergodicity : 0=default, 1=per session, 2=per job

    // -- Pool manager related params
    vector<URI> m_poolConns;
    unsigned m_poolMaxRetries = 3;     // Max number of connection retries
    unsigned m_poolWorkTimeout = 180;  // If no new jobs in this number of seconds drop connection
    unsigned m_poolRespTimeout = 2;    // If no response in this number of seconds drop connection
    unsigned m_poolFlvrTimeout = 0;    // Return to primary pool after this number of minutes
    bool m_poolHashRate = false;       // Whether or not ethminer should send HR to pool

    // -- Benchmarking related params
    unsigned m_benchmarkWarmup = 15;
    unsigned m_benchmarkTrial = 3;
    unsigned m_benchmarkTrials = 5;
    unsigned m_benchmarkBlock = 0;

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
    string m_http_bind;                 // HTTP interface binding address in form <address>:<port>
    string m_http_address = "0.0.0.0";  // HTTP interface binding address (Default any)
    uint16_t m_http_port = 0;           // HTTP interface binding port
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
