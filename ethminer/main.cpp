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
#endif

using namespace std;
using namespace dev;
using namespace dev::eth;

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

bool g_got_exit_signal = false;

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

    MinerCLI() : m_io_work_timer(g_io_service), m_io_strand(g_io_service)
    {
        // Post first deadline timer to give io_service
        // initial work
        m_io_work_timer.expires_from_now(boost::posix_time::seconds(60));
        m_io_work_timer.async_wait(m_io_strand.wrap(
            boost::bind(&MinerCLI::io_work_timer_handler, this, boost::asio::placeholders::error)));

        // Start io_service in it's own thread
        m_io_thread = std::thread{boost::bind(&boost::asio::io_service::run, &g_io_service)};

        // Io service is now live and running
        // All components using io_service should post to reference of g_io_service
        // and should not start/stop or even join threads (which heavily time consuming)
    }

    void io_work_timer_handler(const boost::system::error_code& ec)
    {
        if (!ec)
        {
            // This does absolutely nothing aside resubmitting timer
            // ensuring io_service's queue has always something to do
            m_io_work_timer.expires_from_now(boost::posix_time::seconds(120));
            m_io_work_timer.async_wait(m_io_strand.wrap(boost::bind(
                &MinerCLI::io_work_timer_handler, this, boost::asio::placeholders::error)));
        }
    }

    void stop_io_service()
    {
        // Here we stop all io_service's related activities
        g_io_service.stop();
        m_io_thread.join();
    }

    static void signalHandler(int sig)
    {
        (void)sig;
        g_got_exit_signal = true;
    }
#if API_CORE
    static bool ParseBind(const std::string& inaddr, std::string& outaddr, int& outport,
        bool advertise_negative_port, std::string& errstr)
    {
        std::string addr = inaddr;
        {
            std::string portstr = addr.substr(
                (addr.find_last_of(":") == std::string::npos) ? 0 : addr.find_last_of(":"));
            addr.resize(addr.length() - portstr.length());
            if (!portstr.empty() && portstr[0] == ':')
            {
                portstr = portstr.substr(1);
            }
            bool out_of_int_range = false;
            try
            {
                outport = std::stoi(portstr);
            }
            catch (const std::out_of_range&)
            {
                out_of_int_range = true;
            }
            catch (...)
            {
                errstr = std::string("unable to extract port from string: \"") + portstr +
                         std::string("\"");
                return false;
            };
            if (out_of_int_range || abs(outport) < 1 || abs(outport) > 0xFFFF)
            {
                errstr =
                    std::string("port out of range: ") + portstr +
                    (advertise_negative_port ?
                            std::string(" (must be a non-zero value between -65535 and 65535)") :
                            std::string(" (must be between 1-65535)"));
                return false;
            }
            if (portstr.length() != std::to_string(outport).length())
            {
                errstr = std::string("invalid characters found after port specification: \"") +
                         portstr + std::string("\"");
                return false;
            }
        }
        if (addr.empty())
        {
            addr = "0.0.0.0";
        }
        boost::system::error_code ec;
        boost::asio::ip::address address = boost::asio::ip::address::from_string(addr, ec);
        if (ec)
        {
            errstr = std::string("invalid ip address: \"") + addr +
                     std::string("\" - parsing error: ") + ec.message();
            return false;
        }
        outaddr = addr;
        try
        {
            boost::asio::io_service io_service;
            boost::asio::ip::tcp::acceptor a(
                io_service, boost::asio::ip::tcp::endpoint(address, abs(outport)));
        }
        catch (const std::exception& ex)
        {
            errstr = std::string("unable to bind to ") + addr + std::string(":") +
                     std::to_string(abs(outport)) + std::string(" - error message: ") +
                     std::string(ex.what());
            return false;
        }
        return true;
    }
#endif
    void ParseCommandLine(int argc, char** argv)
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

        app.add_option("--farm-recheck", m_farmRecheckPeriod,
               "Set check interval in milliseconds for changed work", true)
            ->group(CommonGroup)
            ->check(CLI::Range(1, 99999));

        app.add_option(
               "--farm-retries", m_maxFarmRetries, "Set number of reconnection retries", true)
            ->group(CommonGroup)
            ->check(CLI::Range(0, 99999));

        app.add_option("--work-timeout", m_worktimeout,
               "Set disconnect timeout in seconds of working on the same job", true)
            ->group(CommonGroup)
            ->check(CLI::Range(180, 99999));

        app.add_option("--response-timeout", m_responsetimeout,
               "Set disconnect timeout in seconds for pool responses", true)
            ->group(CommonGroup)
            ->check(CLI::Range(2, 999));

        app.add_flag("-R,--report-hashrate", m_report_hashrate, "Report current hashrate to pool")
            ->group(CommonGroup);

        app.add_option("--display-interval", m_displayInterval,
               "Set mining stats log interval in seconds", true)
            ->group(CommonGroup)
            ->check(CLI::Range(1, 99999));

        unsigned hwmon;
        auto hwmon_opt = app.add_option("--HWMON", hwmon,
            "0 - Displays gpu temp, fan percent. 1 - and power usage."
            " Note for Linux: The program uses sysfs for power, which requires running with root "
            "privileges.");
        hwmon_opt->group(CommonGroup)->check(CLI::Range(1));

        app.add_flag("--exit", m_exit, "Stops the miner whenever an error is encountered")
            ->group(CommonGroup);

        vector<string> pools;
        app.add_option(
               "-P,--pool,pool", pools, "Specify one or more pool URLs. See below for URL syntax")
            ->group(CommonGroup);

        app.add_option("--failover-timeout", m_failovertimeout,
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
                string errormsg;
                if (!MinerCLI::ParseBind(
                        bind_arg, this->m_api_address, this->m_api_port, true, errormsg))
                {
                    throw CLI::ValidationError("--api-bind", errormsg);
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
                string errormsg;
                int port;
                if (!MinerCLI::ParseBind(bind_arg, this->m_http_address, port, false, errormsg))
                {
                    throw CLI::ValidationError("--http-bind", errormsg);
                }
                if (port < 0)
                {
                    throw CLI::ValidationError("--http-bind",
                        "the web API does not have read/write modes, specify a positive port "
                        "number between 1-65535");
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


        app.add_option("--opencl-platform", m_openclPlatform, "Use OpenCL platform n", true)
            ->group(OpenCLGroup);

        app.add_option("--opencl-device,--opencl-devices", m_openclDevices,
               "Select list of devices to mine on (default: use all available)")
            ->group(OpenCLGroup);

        int openclThreadsPerHash = -1;
        app.add_set(
               "--cl-parallel-hash", openclThreadsPerHash, {1, 2, 4, 8}, "ignored parameter", true)
            ->group(OpenCLGroup);

        app.add_option("--cl-global-work", m_globalWorkSizeMultiplier,
               "Set the global work size multipler.", true)
            ->group(OpenCLGroup);

        app.add_set("--cl-local-work", m_localWorkSize, {64, 128, 192, 256},
               "Set the local work size", true)
            ->group(OpenCLGroup);

        app.add_flag(
               "--cl-only", m_noBinary, "Use opencl kernel. Don't attempt to load binary kernel")
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

        app.add_option("--cuda-streams", m_numStreams, "Set the number of streams", true)
            ->group(CUDAGroup)
            ->check(CLI::Range(1, 99));

#endif
        app.add_flag("--noeval", m_noEval, "Bypass host software re-evaluation of GPU solutions")
            ->group(CommonGroup);

        app.add_option("-L,--dag-load-mode", m_dagLoadMode,
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

        app.add_option("--dag-single-dev", m_dagCreateDevice,
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

        app.add_option("--tstop", m_tstop,
               "Stop mining on a GPU if temperature exceeds value. 0 is disabled, valid: 30..100",
               true)
            ->group(CommonGroup)
            ->check(CLI::Range(30, 100));

        app.add_option("--tstart", m_tstart,
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

        try
        {
            app.parse(argc, argv);
            if (help)
            {
                cerr << endl << app.help() << endl;
                exit(0);
            }
            else if (version)
            {
                auto* bi = ethminer_get_buildinfo();
                cerr << "\nethminer " << bi->project_version << "\nBuild: " << bi->system_name
                     << "/" << bi->build_type << "/" << bi->compiler_id << "\n\n";
                exit(0);
            }
        }
        catch (const CLI::ParseError& e)
        {
            cerr << endl << e.what() << "\n\n";
            exit(-1);
        }

#if ETH_ETHASHCL
        if (clKernel >= 0)
            clog << "--cl-kernel ignored. Kernel is auto-selected\n";
        if (openclThreadsPerHash >= 0)
            clog << "--cl-parallel-hash ignored. No longer applies\n";
#endif

        if (hwmon_opt->count())
        {
            m_show_hwmonitors = true;
            if (hwmon)
                m_show_power = true;
        }

        if (!cl_miner && !cuda_miner && !mixed_miner && !bench_opt->count() && !sim_opt->count())
        {
            cerr << endl
                 << "One of -G, -U, -X, -M, or -Z must be specified"
                 << "\n\n";
            exit(-1);
        }

        if (cl_miner)
            m_minerType = MinerType::CL;
        else if (cuda_miner)
            m_minerType = MinerType::CUDA;
        else if (mixed_miner)
            m_minerType = MinerType::Mixed;
        if (bench_opt->count())
            m_mode = OperationMode::Benchmark;
        else if (sim_opt->count())
            m_mode = OperationMode::Simulation;

        for (auto url : pools)
        {
            if (url == "exit")  // add fake scheme and port to 'exit' url
                url = "stratum+tcp://-:x@exit:0";
            URI uri(url);
            if (!uri.Valid())
            {
                cerr << endl << "Bad endpoint address: " << url << "\n\n";
                exit(-1);
            }
            if (!uri.KnownScheme())
            {
                cerr << endl << "Unknown URI scheme " << uri.Scheme() << "\n\n";
                exit(-1);
            }
            m_endpoints.push_back(uri);

            OperationMode mode = OperationMode::None;
            switch (uri.Family())
            {
            case ProtocolFamily::STRATUM:
                mode = OperationMode::Stratum;
                break;
            case ProtocolFamily::GETWORK:
                mode = OperationMode::Farm;
                break;
            }
            if ((m_mode != OperationMode::None) && (m_mode != mode))
            {
                cerr << endl
                     << "Mixed stratum and getwork endpoints not supported."
                     << "\n\n";
                exit(-1);
            }
            m_mode = mode;
        }

        if ((m_mode == OperationMode::None) && !m_shouldListDevices)
        {
            cerr << endl
                 << "At least one pool URL must be specified"
                 << "\n\n";
            exit(-1);
        }

#if ETH_ETHASHCL
        m_openclDeviceCount = m_openclDevices.size();
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

        if (m_tstop && (m_tstop <= m_tstart))
        {
            cerr << endl
                 << "tstop must be greater than tstart"
                 << "\n\n";
            exit(-1);
        }
        if (m_tstop && !m_show_hwmonitors)
        {
            // if we want stop mining at a specific temperature, we have to
            // monitor the temperature ==> so auto enable HWMON.
            m_show_hwmonitors = true;
        }
    }

    void execute()
    {
        if (m_shouldListDevices)
        {
#if ETH_ETHASHCL
            if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
                CLMiner::listDevices();
#endif
#if ETH_ETHASHCUDA
            if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
                CUDAMiner::listDevices();
#endif
            stop_io_service();
            exit(0);
        }

        auto* build = ethminer_get_buildinfo();
        minelog << "ethminer " << build->project_version;
        minelog << "Build: " << build->system_name << "/" << build->build_type;

        if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
        {
#if ETH_ETHASHCL
            if (m_openclDeviceCount > 0)
            {
                CLMiner::setDevices(m_openclDevices, m_openclDeviceCount);
                m_miningThreads = m_openclDeviceCount;
            }

            if (!CLMiner::configureGPU(m_localWorkSize, m_globalWorkSizeMultiplier,
                    m_openclPlatform, 0, m_dagLoadMode, m_dagCreateDevice, m_noEval, m_exit,
                    m_noBinary))
            {
                stop_io_service();
                exit(1);
            };

            CLMiner::setNumInstances(m_miningThreads);
#else
            cerr << endl
                 << "Selected GPU mining without having compiled with -DETHASHCL=ON"
                 << "\n\n";
            exit(1);
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
                cwarn << "CUDA error: " << err.what();
                stop_io_service();
                exit(1);
            }

            if (!CUDAMiner::configureGPU(m_cudaBlockSize, m_cudaGridSize, m_numStreams,
                    m_cudaSchedule, m_dagLoadMode, m_dagCreateDevice, m_noEval, m_exit))
            {
                stop_io_service();
                exit(1);
            }

            CUDAMiner::setParallelHash(m_cudaParallelHash);
#else
            cerr << endl
                 << "CUDA support disabled. Configure project build with -DETHASHCUDA=ON"
                 << "\n\n";
            stop_io_service();
            exit(1);
#endif
        }

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
            cerr << endl
                 << "Program logic error"
                 << "\n\n";
            exit(-1);
        }
    }

private:
    void doBenchmark(MinerType _m, unsigned _warmupDuration = 15, unsigned _trialDuration = 3,
        unsigned _trials = 5)
    {
        BlockHeader genesis;
        genesis.setNumber(m_benchmarkBlock);
        genesis.setDifficulty(u256(1) << 64);

        new Farm(m_show_hwmonitors, m_show_power);
        map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
        sealers["opencl"] = Farm::SealerDescriptor{
            &CLMiner::instances, [](unsigned _index) { return new CLMiner(_index); }};
#endif
#if ETH_ETHASHCUDA
        sealers["cuda"] = Farm::SealerDescriptor{
            &CUDAMiner::instances, [](unsigned _index) { return new CUDAMiner(_index); }};
#endif
        Farm::f().setSealers(sealers);
        Farm::f().onSolutionFound([&](Solution) {
            return false;
        });

        Farm::f().setTStartTStop(m_tstart, m_tstop);

        string platformInfo = _m == MinerType::CL ? "CL" : "CUDA";
        cout << "Benchmarking on platform: " << platformInfo << endl;

        cout << "Preparing DAG for block #" << m_benchmarkBlock << endl;
        // genesis.prep();

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
                cout << "Warming up..." << endl;
            else
                cout << "Trial " << i << "... " << flush << endl;
            this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));

            auto mp = Farm::f().miningProgress();
            if (!i)
                continue;
            auto rate = uint64_t(mp.hashRate);

            cout << rate << endl;
            results.push_back(rate);
            mean += uint64_t(rate);
        }
        sort(results.begin(), results.end());
        cout << "min/mean/max: " << results.front() << "/" << (mean / _trials) << "/"
             << results.back() << " H/s" << endl;
        if (results.size() > 2)
        {
            for (auto it = results.begin() + 1; it != results.end() - 1; it++)
                innerMean += *it;
            innerMean /= (_trials - 2);
            cout << "inner mean: " << innerMean << " H/s" << endl;
        }
        else
            cout << "inner mean: n/a" << endl;
        stop_io_service();
        exit(0);
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

        if (m_mode == OperationMode::Stratum)
        {
            client = new EthStratumClient(m_worktimeout, m_responsetimeout, m_report_hashrate);
        }
        else if (m_mode == OperationMode::Farm)
        {
            client = new EthGetworkClient(m_farmRecheckPeriod, m_report_hashrate);
        }
        else if (m_mode == OperationMode::Simulation)
        {
            client = new SimulateClient(20, m_benchmarkBlock);
        }
        else
        {
            cwarn << "Invalid OperationMode";
            exit(1);
        }

        // Should not happen!
        if (!client)
        {
            cwarn << "Invalid PoolClient";
            stop_io_service();
            exit(1);
        }

        // sealers, m_minerType
        new Farm(m_show_hwmonitors, m_show_power);
        Farm::f().setSealers(sealers);

        new PoolManager(client, m_minerType, m_maxFarmRetries, m_failovertimeout);

        Farm::f().setTStartTStop(m_tstart, m_tstop);

        // If we are in simulation mode we add a fake connection
        if (m_mode == OperationMode::Simulation)
        {
            URI con(URI("http://-:0"));
            PoolManager::p().clearConnections();
            PoolManager::p().addConnection(con);
        }
        else
        {
            for (auto conn : m_endpoints)
            {
                cnote << "Configured pool " << conn.Host() + ":" + to_string(conn.Port());
                PoolManager::p().addConnection(conn);
            }
        }

#if API_CORE

        ApiServer api(m_api_address, m_api_port, m_api_password);
        api.start();

        http_server.run(m_http_address, m_http_port, m_show_hwmonitors, m_show_power);

#endif

        // Start PoolManager
        PoolManager::p().start();

        unsigned interval = m_displayInterval;

        // Run CLI in loop
        while (!g_got_exit_signal && PoolManager::p().isRunning())
        {
            // Wait at the beginning of the loop to give some time
            // services to start properly. Otherwise we get a "not-connected"
            // message immediately

            // Split m_displayInterval in 1 second parts to optain a faster exit
            this_thread::sleep_for(chrono::seconds(1));
            interval--;
            if (interval)
                continue;
            interval = m_displayInterval;

            // Display current stats of the farm if it's connected
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
        }

#if API_CORE

        // Stop Api server
        api.stop();

#endif

        PoolManager::p().stop();
        stop_io_service();

        cnote << "Terminated!";
        exit(0);
    }

    /// Operating mode.
    OperationMode m_mode = OperationMode::None;

    /// Global boost's io_service
    std::thread m_io_thread;                      // The IO service thread
    boost::asio::deadline_timer m_io_work_timer;  // A dummy timer to keep io_service with something
                                                  // to do and prevent io shutdown
    boost::asio::io_service::strand m_io_strand;  // A strand to serialize posts in multithreaded
                                                  // environment

    /// Mining options
    MinerType m_minerType = MinerType::Mixed;
    unsigned m_openclPlatform = 0;
    unsigned m_miningThreads = UINT_MAX;
    bool m_shouldListDevices = false;
#if ETH_ETHASHCL
    unsigned m_openclDeviceCount = 0;
    vector<unsigned> m_openclDevices;
    unsigned m_globalWorkSizeMultiplier = CLMiner::c_defaultGlobalWorkSizeMultiplier;
    unsigned m_localWorkSize = CLMiner::c_defaultLocalWorkSize;
    bool m_noBinary = false;
#endif
#if ETH_ETHASHCUDA
    unsigned m_cudaDeviceCount = 0;
    vector<unsigned> m_cudaDevices;
    unsigned m_numStreams = CUDAMiner::c_defaultNumStreams;
    unsigned m_cudaSchedule = 4;  // sync
    unsigned m_cudaGridSize = CUDAMiner::c_defaultGridSize;
    unsigned m_cudaBlockSize = CUDAMiner::c_defaultBlockSize;
    unsigned m_cudaParallelHash = 4;
#endif
    bool m_noEval = false;
    unsigned m_dagLoadMode = 0;  // parallel
    unsigned m_dagCreateDevice = 0;
    bool m_exit = false;
    /// Benchmarking params
    unsigned m_benchmarkWarmup = 15;
    unsigned m_benchmarkTrial = 3;
    unsigned m_benchmarkTrials = 5;
    unsigned m_benchmarkBlock = 0;

    vector<URI> m_endpoints;

    unsigned m_maxFarmRetries = 3;
    unsigned m_farmRecheckPeriod = 500;
    unsigned m_displayInterval = 5;

    // Number of seconds to wait before triggering a no work timeout from pool
    unsigned m_worktimeout = 180;
    // Number of seconds to wait before triggering a response timeout from pool
    unsigned m_responsetimeout = 2;
    // Number of minutes to wait on a failover pool before trying to go back to primary. In minutes
    // !!
    unsigned m_failovertimeout = 0;

    bool m_show_hwmonitors = false;
    bool m_show_power = false;

    unsigned m_tstop = 0;
    unsigned m_tstart = 40;

#if API_CORE
    string m_api_bind;
    string m_api_address = "0.0.0.0";
    int m_api_port = 0;
    string m_api_password;
    string m_http_bind;
    string m_http_address = "0.0.0.0";
    uint16_t m_http_port = 0;
#endif

    bool m_report_hashrate = false;

#if ETH_DBUS
    DBusInt dbusint;
#endif
};

int main(int argc, char** argv)
{
    try
    {
        // Set env vars controlling GPU driver behavior.
        setenv("GPU_MAX_HEAP_SIZE", "100");
        setenv("GPU_MAX_ALLOC_PERCENT", "100");
        setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

        MinerCLI m;

        m.ParseCommandLine(argc, argv);

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

        m.execute();
    }
    catch (std::exception& ex)
    {
        cerr << "Error: " << ex.what() << "\n\n";
        return -1;
    }

    return 0;
}
