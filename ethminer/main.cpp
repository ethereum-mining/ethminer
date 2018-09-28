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

using namespace std;
using namespace dev;
using namespace dev::eth;


// Global vars
bool g_running = false;
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
        (void)sig;
        dev::setThreadName("main");
        cnote << "Signal intercepted ...";
        g_running = false;
        g_shouldstop.notify_all();
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
        const char* CommonGroup = "Common Options";
#if API_CORE
        const char* APIGroup = "API Options";
#endif
#if ETH_ETHASHCL
        const char* OpenCLGroup = "OpenCL Options";
#endif
#if ETH_ETHASHCUDA
        const char* CUDAGroup = "CUDA Options";
#endif

        CLI::App app("Ethminer - GPU Ethereum miner");

        bool help = false;
        app.set_help_flag();
        app.add_flag("-h,--help", help, "Show help")->group(CommonGroup);

        bool version = false;
        app.add_flag("-V,--version", version, "Show program version")->group(CommonGroup);

        ostringstream logOptions;
        logOptions << "Set log display options. Use the summ of:"
                   << " log json messages = " << LOG_JSON
                   << ", log per GPU solutions = " << LOG_PER_GPU;
#ifdef DEV_BUILD
        logOptions << ", log connection messages = " << LOG_CONNECT
                   << ", log switch delay = " << LOG_SWITCH
                   << ", log submit delay = " << LOG_SUBMIT
                   << ", log program flow = " << LOG_PROGRAMFLOW;
#endif
        app.add_option("-v,--verbosity", g_logOptions, logOptions.str(), true)
            ->group(CommonGroup)
            ->check(CLI::Range(LOG_NEXT - 1));

        app.add_option("--farm-recheck", m_farmPollInterval,
               "Set check interval in milliseconds for changed work", true)
            ->group(CommonGroup)
            ->check(CLI::Range(1, 99999));

        app.add_option(
               "--farm-retries", m_poolMaxRetries, "Set number of reconnection retries", true)
            ->group(CommonGroup)
            ->check(CLI::Range(0, 99999));

        app.add_option("--work-timeout", m_poolWorkTimeout,
               "Set disconnect timeout in seconds of working on the same job", true)
            ->group(CommonGroup)
            ->check(CLI::Range(180, 99999));

        app.add_option("--response-timeout", m_poolRespTimeout,
               "Set disconnect timeout in seconds for pool responses", true)
            ->group(CommonGroup)
            ->check(CLI::Range(2, 999));

        app.add_flag("-R,--report-hashrate", m_poolHashRate, "Report current hashrate to pool")
            ->group(CommonGroup);

        app.add_option("--display-interval", m_cliDisplayInterval,
               "Set mining stats log interval in seconds.", true)
            ->group(CommonGroup)
            ->check(CLI::Range(1, 1800));

        app.add_option("--HWMON", m_farmHwMonitors,
               "0 - No monitoring; "
               "1 - Monitor GPU temp and fan; "
               "2 - Monitor GPU temp fan and power drain; ",
               true)
            ->group(CommonGroup)
            ->check(CLI::Range(0, 2));

        app.add_flag(
               "--exit", m_farmExitOnErrors, "Stops the miner whenever an error is encountered")
            ->group(CommonGroup);

        vector<string> pools;
        app.add_option(
               "-P,--pool,pool", pools, "Specify one or more pool URLs. See below for URL syntax")
            ->group(CommonGroup);

        app.add_option("--failover-timeout", m_poolFlvrTimeout,
               "Set the amount of time in minutes to stay on a failover pool before trying to "
               "reconnect to primary. If = 0 then no switch back.",
               true)
            ->group(CommonGroup)
            ->check(CLI::Range(0, 999));

        app.add_flag("--nocolor", g_logNoColor, "Display monochrome log")->group(CommonGroup);

        app.add_flag("--syslog", g_logSyslog,
               "Use syslog appropriate log output (drop timestamp and channel prefix)")
            ->group(CommonGroup);

#if API_CORE
        app.add_option("--api-bind", m_api_bind,
               "Set the API address:port the miner should listen to. Use negative port number for "
               "readonly mode",
               true)
            ->group(APIGroup)
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
        app.add_option("--api-port", m_api_port,
               "Set the API port, the miner should listen to. Use 0 to disable. Use negative "
               "numbers for readonly mode",
               true)
            ->group(APIGroup)
            ->check(CLI::Range(-65535, 65535));

        app.add_option("--api-password", m_api_password,
               "Set the password to protect interaction with API server. If not set, any "
               "connection is granted access. "
               "Be advised passwords are sent unencrypted over plain TCP!!")
            ->group(APIGroup);

        app.add_option("--http-bind", m_http_bind,
               "Set the web API address:port the miner should listen to.", true)
            ->group(APIGroup)
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

        app.add_option("--http-port", m_http_port,
               "Set the web API port, the miner should listen to. Use 0 to disable. Data shown "
               "depends on hwmon setting",
               true)
            ->group(APIGroup)
            ->check(CLI::Range(65535));

#endif

#if ETH_ETHASHCL || ETH_ETHASHCUDA

        app.add_flag("--list-devices", m_shouldListDevices,
               "List the detected OpenCL/CUDA devices and exit. Should be combined with -G, -U, or "
               "-X flag")
            ->group(CommonGroup);

#endif

#if ETH_ETHASHCL

        int clKernel = -1;
        app.add_option("--cl-kernel", clKernel, "Ignored parameter. Kernel is auto-selected.", true)
            ->group(OpenCLGroup)
            ->check(CLI::Range(2));

        app.add_option("--opencl-platform", m_oclPlatform, "Use OpenCL platform n", true)
            ->group(OpenCLGroup);

        app.add_option("--opencl-device,--opencl-devices", m_oclDevices,
               "Select list of devices to mine on (default: use all available)")
            ->group(OpenCLGroup);

        int openclThreadsPerHash = -1;
        app.add_set(
               "--cl-parallel-hash", openclThreadsPerHash, {1, 2, 4, 8}, "ignored parameter", true)
            ->group(OpenCLGroup);

        app.add_option(
               "--cl-global-work", m_oclGWorkSize, "Set the global work size multipler.", true)
            ->group(OpenCLGroup);

        app.add_set("--cl-local-work", m_oclLWorkSize, {64, 128, 192, 256},
               "Set the local work size", true)
            ->group(OpenCLGroup);

        app.add_flag(
               "--cl-only", m_oclNoBinary, "Use opencl kernel. Don't attempt to load binary kernel")
            ->group(OpenCLGroup);
#endif

#if ETH_ETHASHCUDA

        app.add_option("--cuda-grid-size", m_cudaGridSize, "Set the grid size", true)
            ->group(CUDAGroup)
            ->check(CLI::Range(1, 131072));

        app.add_set(
               "--cuda-block-size", m_cudaBlockSize, {32, 64, 128, 256}, "Set the block size", true)
            ->group(CUDAGroup);

        app.add_option("--cuda-devices", m_cudaDevices,
               "Select list of devices to mine on (default: use all available)")
            ->group(CUDAGroup);

        app.add_set("--cuda-parallel-hash", m_cudaParallelHash, {1, 2, 4, 8},
               "Set the number of hashes per kernel", true)
            ->group(CUDAGroup);

        string sched = "sync";
        app.add_set("--cuda-schedule", sched, {"auto", "spin", "yield", "sync"},
               "Set the scheduler mode.", true)
            ->group(CUDAGroup);

        app.add_option("--cuda-streams", m_cudaStreams, "Set the number of streams", true)
            ->group(CUDAGroup)
            ->check(CLI::Range(1, 99));

#endif
        app.add_flag(
               "--noeval", m_farmNoEval, "Bypass host software re-evaluation of GPU solutions")
            ->group(CommonGroup);

        app.add_option("-L,--dag-load-mode", m_farmDagLoadMode,
               "Set the DAG load mode. 0=parallel, 1=sequential, 2=single."
               "  parallel    - load DAG on all GPUs at the same time"
               "  sequential  - load DAG on GPUs one after another. Use this when the miner "
               "crashes during DAG generation"
               "  single      - generate DAG on device, then copy to other devices. Implies "
               "--dag-single-dev"
               "  ",
               true)
            ->group(CommonGroup)
            ->check(CLI::Range(2));

        app.add_option("--dag-single-dev", m_farmDagCreateDevice,
               "Set the DAG creation device in single mode", true)
            ->group(CommonGroup);

        app.add_option("--benchmark-warmup", m_benchmarkWarmup,
               "Set the duration in seconds of warmup for the benchmark tests", true)
            ->group(CommonGroup);

        app.add_option("--benchmark-trial", m_benchmarkTrial,
               "Set the number of benchmark trials to run", true)
            ->group(CommonGroup)
            ->check(CLI::Range(1, 99));

        bool cl_miner = false;
        app.add_flag("-G,--opencl", cl_miner, "When mining use the GPU via OpenCL")
            ->group(CommonGroup);

        bool cuda_miner = false;
        app.add_flag("-U,--cuda", cuda_miner, "When mining use the GPU via CUDA")
            ->group(CommonGroup);

        bool mixed_miner = false;
        app.add_flag(
               "-X,--cuda-opencl", mixed_miner, "When mining with mixed AMD(OpenCL) and CUDA GPUs")
            ->group(CommonGroup);

        auto bench_opt = app.add_option("-M,--benchmark", m_benchmarkBlock,
            "Benchmark mining and exit; Specify block number to benchmark against specific DAG",
            true);
        bench_opt->group(CommonGroup);

        auto sim_opt = app.add_option("-Z,--simulation", m_benchmarkBlock,
            "Mining test. Used to validate kernel optimizations. Specify block number", true);
        sim_opt->group(CommonGroup);

        app.add_option("--tstop", m_farmTempStop,
               "Stop mining on a GPU if temperature exceeds value. 0 is disabled, valid: 30..100",
               true)
            ->group(CommonGroup)
            ->check(CLI::Range(30, 100));

        app.add_option("--tstart", m_farmTempStart,
               "Restart mining on a GPU if the temperature drops below, valid: 30..100", true)
            ->group(CommonGroup)
            ->check(CLI::Range(30, 100));

        ostringstream ssHelp;
        ssHelp
            << "Pool URL Specification:" << endl
            << "    URL takes the form: scheme://user[.workername][:password]@hostname:port[/...]."
            << endl
            << "    where scheme can be any of:" << endl
            << "    getwork     for getWork mode" << endl
            << "    stratum     for stratum mode" << endl
            << "    stratums    for secure stratum mode" << endl
            << "    stratumss   for secure stratum mode with strong TLS12 verification" << endl
            << "    for a complete list if available schemes, see below" << endl
            << endl
            << "    Example 1:"
               "    stratums://0x012345678901234567890234567890123.miner1@ethermine.org:5555"
            << endl
            << "    Example 2:"
               "    stratum://0x012345678901234567890234567890123.miner1@nanopool.org:9999/"
               "john.doe@gmail.com"
            << endl
            << "    Example 3:"
               "    stratum://0x012345678901234567890234567890123@nanopool.org:9999/miner1/"
               "john.doe@gmail.com"
            << endl
            << endl
            << "    There are three main variants of the stratum protocol. If you are not sure"
            << endl
            << "    which one your pool needs, try one of the 3 schemes above and ethminer" << endl
            << "    will try to detect the correct variant automatically. If you know your" << endl
            << "    pool's requirements, the following are supported." << endl
            << "    Schemes: " << URI::KnownSchemes(ProtocolFamily::STRATUM) << endl
            << "    Where a scheme is made up of two parts, the stratum variant + the transport"
            << endl
            << "    protocol." << endl
            << "    stratum variants:" << endl
            << "      stratum -   Stratum" << endl
            << "      stratum1 -  Eth proxy" << endl
            << "      stratum2 -  Eth stratum (nicehash)" << endl
            << "    transports:" << endl
            << "      tcp -       Unencrypted connection" << endl
            << "      tls -       Encrypted with tls (including deprecated tls 1.1)" << endl
            << "      tls12,ssl - Encrypted with tls 1.2 or later" << endl
            << endl
#if ETH_ETHASHCUDA
            << "Cuda scheduler modes" << endl
            << "    auto  - Uses a heuristic based on the number of active CUDA contexts in the "
            << endl
            << "            process C and the number of logical processors in the system P." << endl
            << "            If C > P then yield else spin." << endl
            << "    spin  - Instruct CUDA to actively spin when waiting for results from the "
               "device."
            << endl
            << "    yield - Instruct CUDA to yield its thread when waiting for results from the "
            << endl
            << "            device." << endl
            << "    sync  - Instruct CUDA to block the CPU thread on a synchronization primitive "
            << endl
            << "            when waiting for the results from the device." << endl
            << endl
#endif
            << "Environment Variables:" << endl
            << "    NO_COLOR - set to any value to disable color output. Unset to re-enable "
               "color output."
            << endl
            << "    SYSLOG   - set to any value to strip time and disable color from output, "
               "for logging under systemd"
#ifndef _WIN32
            << endl
            << "    SSL_CERT_FILE - full path to your CA certificates file if elsewhere than "
               "/etc/ssl/certs/ca-certificates.crt"
#endif
            ;
        app.footer(ssHelp.str());

        // Exception handling is held at higher level
        app.parse(argc, argv);
        if (help)
        {
            cerr << endl << app.help() << endl;
            return false;
        }
        else if (version)
        {
            auto* bi = ethminer_get_buildinfo();
            cerr << "\nethminer " << bi->project_version << "\nBuild: " << bi->system_name << "/"
                 << bi->build_type << "/" << bi->compiler_id << "\n\n";
            return false;
        }


#if ETH_ETHASHCL
        if (clKernel >= 0)
            clog << "--cl-kernel ignored. Kernel is auto-selected\n";
        if (openclThreadsPerHash >= 0)
            clog << "--cl-parallel-hash ignored. No longer applies\n";
#endif
#ifndef DEV_BUILD
        if ((g_logOptions & LOG_CONNECT) || (g_logOptions & LOG_SWITCH) ||
            (g_logOptions & LOG_SUBMIT) || (g_logOptions & LOG_PROGRAMFLOW))
        {
            clog << "Some log options ignored. Compile with -DDEVBUILD=ON\n";
        }
#endif


        if (!cl_miner && !cuda_miner && !mixed_miner && !bench_opt->count() && !sim_opt->count())
            throw std::invalid_argument("One of - G, -U, -X, -M, or -Z must be specified");

        if (cl_miner)
            m_minerType = MinerType::CL;
        else if (cuda_miner)
            m_minerType = MinerType::CUDA;
        else if (mixed_miner)
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
                    std::string what = "Bad URI : " + uri.String();
                    throw std::invalid_argument(what);
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


#if ETH_ETHASHCL
        m_oclDeviceCount = m_oclDevices.size();
#endif

#if ETH_ETHASHCUDA
        m_cudaDeviceCount = m_cudaDevices.size();
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

        return true;
    }

    void execute()
    {
        auto* bi = ethminer_get_buildinfo();

        if (m_shouldListDevices)
        {
            cout << "\nethminer " << bi->project_version << "\nBuild: " << bi->system_name << "/"
                 << bi->build_type << "/" << bi->compiler_id << endl;

            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
            {
#if ETH_ETHASHCL

                CLMiner::listDevices();
#else
                throw std::runtime_error(
                    "Selected OpenCL device listing without having compiled with -DETHASHCL=ON");
#endif
            }
            if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
            {
#if ETH_ETHASHCUDA
                CUDAMiner::listDevices();
#else
                throw std::runtime_error(
                    "Selected CUDA device listing without having compiled with -DETHASHCUDA=ON");
#endif
            }
            cout << endl;
            return;
        }

        minelog << "ethminer " << bi->project_version;
        minelog << "Build: " << bi->system_name << "/" << bi->build_type;

        if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
        {
#if ETH_ETHASHCL
            if (m_oclDeviceCount > 0)
            {
                CLMiner::setDevices(m_oclDevices, m_oclDeviceCount);
                m_miningThreads = m_oclDeviceCount;
            }

            if (!CLMiner::configureGPU(m_oclLWorkSize, m_oclGWorkSize, m_oclPlatform, 0,
                    m_farmDagLoadMode, m_farmDagCreateDevice, m_farmExitOnErrors, m_oclNoBinary))
            {
                throw std::runtime_error("Unable to initialize OpenCL GPU(s)");
            }

            CLMiner::setNumInstances(m_miningThreads);
#else
            throw std::runtime_error(
                "Selected OpenCL mining without having compiled with -DETHASHCL=ON");
#endif
        }
        if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
        {
#if ETH_ETHASHCUDA
            try
            {
                if (m_cudaDeviceCount > 0)
                {
                    CUDAMiner::setDevices(m_cudaDevices, m_cudaDeviceCount);
                    m_miningThreads = m_cudaDeviceCount;
                }
                CUDAMiner::setNumInstances(m_miningThreads);
            }
            catch (std::runtime_error const& err)
            {
                std::string what = "CUDA error : ";
                what.append(err.what());
                throw std::runtime_error(what);
            }

            if (!CUDAMiner::configureGPU(m_cudaBlockSize, m_cudaGridSize, m_cudaStreams,
                    m_cudaSchedule, m_farmDagLoadMode, m_farmDagCreateDevice, m_farmExitOnErrors))
            {
                throw std::runtime_error("Unable to initialize CUDA GPU(s)");
            }

            CUDAMiner::setParallelHash(m_cudaParallelHash);
#else
            throw std::runtime_error(
                "Selected CUDA mining without having compiled with -DETHASHCUDA=ON");
#endif
        }


        // Enable
        g_running = true;

        // Signal traps
        signal(SIGINT, MinerCLI::signalHandler);
        signal(SIGTERM, MinerCLI::signalHandler);

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
            throw std::runtime_error("Program logic error. Unexpected m_mode.");
        }
    }

private:
    void doBenchmark(MinerType _m, unsigned _warmupDuration = 15, unsigned _trialDuration = 3,
        unsigned _trials = 5)
    {
        BlockHeader genesis;
        genesis.setNumber(m_benchmarkBlock);
        genesis.setDifficulty(u256(1) << 64);


        map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
        sealers["opencl"] = Farm::SealerDescriptor{
            &CLMiner::instances, [](unsigned _index) { return new CLMiner(_index); }};
#endif
#if ETH_ETHASHCUDA
        sealers["cuda"] = Farm::SealerDescriptor{
            &CUDAMiner::instances, [](unsigned _index) { return new CUDAMiner(_index); }};
#endif
        new Farm(m_farmHwMonitors, m_farmNoEval);
        Farm::f().setSealers(sealers);
        Farm::f().setTStartTStop(m_farmTempStart, m_farmTempStop);
        Farm::f().onSolutionFound([&](Solution) { return false; });

        string platformInfo = _m == MinerType::CL ? "CL" : "CUDA";
        minelog << "Benchmarking on platform: " << platformInfo << " Preparing DAG for block #"
                << m_benchmarkBlock;

        if (_m == MinerType::CL)
            Farm::f().start("opencl", false);
        else if (_m == MinerType::CUDA)
            Farm::f().start("cuda", false);

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
        sealers["opencl"] = Farm::SealerDescriptor{
            &CLMiner::instances, [](unsigned _index) { return new CLMiner(_index); }};
#endif
#if ETH_ETHASHCUDA
        sealers["cuda"] = Farm::SealerDescriptor{
            &CUDAMiner::instances, [](unsigned _index) { return new CUDAMiner(_index); }};
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
            client = new EthGetworkClient(m_farmPollInterval, m_poolHashRate);
            break;
        case MinerCLI::OperationMode::Stratum:
            client = new EthStratumClient(m_poolWorkTimeout, m_poolRespTimeout, m_poolHashRate);
            break;
        default:
            // Satisfy the compiler, but cannot happen!
            throw std::runtime_error("Program logic error. Unexpected m_mode.");
        }

        new Farm(m_farmHwMonitors, m_farmNoEval);
        Farm::f().setSealers(sealers);
        Farm::f().setTStartTStop(m_farmTempStart, m_farmTempStop);

        new PoolManager(client, m_minerType, m_poolMaxRetries, m_poolFlvrTimeout);
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
        {
            g_shouldstop.wait(clilock);
        }


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
    boost::asio::deadline_timer m_cliDisplayTimer;  // A dummy timer to keep io_service with
                                                    // something to do and prevent io shutdown
    boost::asio::io_service::strand m_io_strand;    // A strand to serialize posts in multithreaded
                                                    // environment

    // Mining options
    MinerType m_minerType = MinerType::Mixed;
    OperationMode m_mode = OperationMode::None;
    unsigned m_miningThreads = UINT_MAX;  // TODO remove ?
    bool m_shouldListDevices = false;

#if ETH_ETHASHCL
    // -- OpenCL related params
    unsigned m_oclPlatform = 0;
    unsigned m_oclDeviceCount = 0;
    vector<unsigned> m_oclDevices;
    unsigned m_oclGWorkSize = CLMiner::c_defaultGlobalWorkSizeMultiplier;
    unsigned m_oclLWorkSize = CLMiner::c_defaultLocalWorkSize;
    bool m_oclNoBinary = false;
#endif

#if ETH_ETHASHCUDA
    // -- CUDA related params
    unsigned m_cudaDeviceCount = 0;
    vector<unsigned> m_cudaDevices;
    unsigned m_cudaStreams = CUDAMiner::c_defaultNumStreams;
    unsigned m_cudaSchedule = 4;  // sync
    unsigned m_cudaGridSize = CUDAMiner::c_defaultGridSize;
    unsigned m_cudaBlockSize = CUDAMiner::c_defaultBlockSize;
    unsigned m_cudaParallelHash = 4;
#endif

    // -- Farm related params
    unsigned m_farmDagLoadMode = 0;  // DAG load mode : 0=parallel, 1=sequential, 2=single
    unsigned m_farmDagCreateDevice =
        0;  // Ordinal index of GPU creating DAG (Implies m_farmDagLoadMode == 2
    bool m_farmExitOnErrors =
        false;                  // Whether or not ethminer should exit on mining threads errors
    bool m_farmNoEval = false;  // Whether or not ethminer should CPU re-evaluate solutions
    unsigned m_farmPollInterval =
        500;  // In getWork mode this establishes the ms. interval to check for new job
    unsigned m_farmHwMonitors =
        0;  // Farm GPU monitoring level : 0 - No monitor; 1 - Temp and Fan; 2 - Temp Fan Power
    unsigned m_farmTempStop = 0;  // Halt mining on GPU if temperature ge this threshold (Celsius)
    unsigned m_farmTempStart =
        40;  // Resume mining on GPU if temperature le this threshold (Celsius)

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

    if (argc < 2)
    {
        cerr << "\n"
             << "No arguments specified. " << endl
             << "Try 'ethminer --help' to get a list of arguments."
             << "\n\n";
        return 0;
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
            return 0;
        }
        catch (std::invalid_argument& ex1)
        {
            cerr << "\n"
                 << "Error: " << ex1.what() << "\n"
                 << "Try ethminer --help to get an explained list of arguments."
                 << "\n\n";
            ;
            return 1;
        }
        catch (std::runtime_error& ex2)
        {
            cerr << "\n"
                 << "Error: " << ex2.what() << "\n\n";
            return 2;
        }
        catch (std::exception& ex3)
        {
            cerr << "\n"
                 << "Error: " << ex3.what() << "\n\n";
            return 3;
        }
        catch (...)
        {
            cerr << "\n"
                 << "Error: Unknown failure occurred. Possible memory corruption."
                 << "\n\n";
            return 4;
        }
    }
    catch (const std::exception& ex)
    {
        cerr << "\n"
             << "Could not initialize CLI interface " << endl
             << "Error: " << ex.what() << "\n\n";
        return 4;
    }
}
