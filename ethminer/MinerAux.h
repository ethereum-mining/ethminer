#pragma once

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
/** @file MinerAux.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * CLI module for mining.
 */

#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <random>
#include <set>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>
#include <thread>

#include <libethcore/Exceptions.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Farm.h>
#include <ethminer-buildinfo.h>
#if ETH_ETHASHCL
#include <libethash-cl/CLMiner.h>
#endif
#if ETH_ETHASHCUDA
#include <libethash-cuda/CUDAMiner.h>
#endif
#include <libpoolprotocols/PoolManager.h>
#include <libpoolprotocols/stratum/EthStratumClient.h>
#include <libpoolprotocols/getwork/EthGetworkClient.h>
#include <libpoolprotocols/testing/SimulateClient.h>

#if ETH_DBUS
#include "DBusInt.h"
#endif
#if API_CORE
#include <libapicore/Api.h>
#include <libapicore/httpServer.h>
#endif

#include <CLI/CLI.hpp>

using namespace std;
using namespace dev;
using namespace dev::eth;


class BadArgument: public Exception {};
struct MiningChannel: public LogChannel
{
	static const char* name() { return EthGreen "  m"; }
	static const int verbosity = 2;
	static const bool debug = false;
};
#define minelog clog(MiningChannel)

inline std::string toJS(unsigned long _n)
{
	std::string h = toHex(toCompactBigEndian(_n, 1));
	// remove first 0, if it is necessary;
	std::string res = h[0] != '0' ? h : h.substr(1);
	return "0x" + res;
}

bool g_running = false;

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

	MinerCLI() : 
		m_io_work(m_io_service), 
		m_io_work_timer(m_io_service),
		m_io_strand(m_io_service)
	{
		// Post first deadline timer to give io_service
		// initial work
		m_io_work_timer.expires_from_now(boost::posix_time::seconds(60));
		m_io_work_timer.async_wait(m_io_strand.wrap(boost::bind(&MinerCLI::io_work_timer_handler, this, boost::asio::placeholders::error)));

		// Start io_service in it's own thread
		m_io_thread = std::thread{ boost::bind(&boost::asio::io_service::run, &m_io_service) };

		// Io service is now live and running
		// All components using io_service should post to reference of m_io_service
		// and should not start/stop or even join threads (which heavily time consuming)


	}

	void io_work_timer_handler(const boost::system::error_code& ec) {

		if (!ec) {

			// This does absolutely nothing aside resubmitting timer
			// ensuring io_service's queue has always something to do
			m_io_work_timer.expires_from_now(boost::posix_time::seconds(120));
			m_io_work_timer.async_wait(m_io_strand.wrap(boost::bind(&MinerCLI::io_work_timer_handler, this, boost::asio::placeholders::error)));
		}

	}

	void stop_io_service() {

		// Here we stop all io_service's related activities
		m_io_service.stop();
		m_io_thread.join();

	}

	static void signalHandler(int sig)
	{
		(void)sig;
		g_running = false;
	}

	void ParseCommandLine (int argc, char** argv)
	{

		const char* CommonGroup = "Common Options";
		const char* APIGroup =    "API Options";
		const char* OpenCLGroup = "OpenCL Options";
		const char* CUDAGroup =   "CUDA Options";

		CLI::App app("Ethminer - GPU Ethereum miner");

		bool help = false;
		app.set_help_flag();
		app.add_flag("-h,--help", help, "Show help")->group(CommonGroup);

		bool version = false;
		app.add_flag("-V,--version", version, "Display program version")->group(CommonGroup);

		app.add_option("-v,--verbosity", g_logVerbosity,
			"Log verbosity level")
			->group(CommonGroup)
			->check(CLI::Range(9))
			->set_type_name("<n>");

		app.add_option("--farm-recheck", m_farmRecheckPeriod,
			"Leave n ms between checks for changed work (default: 500)")
			->group(CommonGroup)
			->check(CLI::Range(1, 99999))
			->set_type_name("<n>");

		app.add_option("--farm-retries", m_maxFarmRetries,
			"Number of retries until switch to failover (default: 3)")
			->group(CommonGroup)
			->check(CLI::Range(1, 99999))
			->set_type_name("<n>");

		app.add_option("--stratum-email", m_email,
			"Email address used in eth-proxy (optional)")
			->group(CommonGroup)
			->set_type_name("<s>");

		app.add_option("--work-timeout", m_worktimeout,
			"reconnect/failover after n seconds of working on the same (stratum) job (default: 180)")
			->group(CommonGroup)
			->check(CLI::Range(1, 99999))
			->set_type_name("<n>");

		app.add_option("--response-timeout", m_responsetimeout,
			"reconnect/failover after n seconds delay for response from (stratum) pool (minimum: 2, default: 2)")
			->group(CommonGroup)
			->check(CLI::Range(2, 999))
			->set_type_name("<n>");

		app.add_flag("-R,--report-hashrate", m_report_stratum_hashrate,
			"Report current hashrate to pool")
			->group(CommonGroup);

		app.add_option("--display-interval", m_displayInterval,
			"Set mining stats display interval in seconds (default: 5)")
			->group(CommonGroup)
			->check(CLI::Range(1, 99999))
			->set_type_name("<n>");

		unsigned hwmon;
		auto hwmon_opt = app.add_option("--HWMON", hwmon,
			"0 - Displays gpu temp, fan percent. 1 - and power usage."
			" Note: In linux, the program uses sysfs, which may require running with root privileges.");
			hwmon_opt->group(CommonGroup)
				->check(CLI::Range(1))
				->set_type_name("<0|1>");

		app.add_flag("--exit", m_exit,
			"Stops the miner whenever an error is encountered")
			->group(CommonGroup);

		vector<string> pools;
		app.add_option("-P,--pool", pools,
			"Specify one or more pool URLs")
			->group(CommonGroup)
			->set_type_name("<url>");
	
#if API_CORE

		app.add_option("--api-port", m_api_port,
			"The api port, the miner should listen to. Use 0 to disable. Use negative numbers for readonly mode (default: 0)")
			->group(APIGroup)
			->check(CLI::Range(-32767, 32767))
			->set_type_name("<n>");

		app.add_option("--http-port", m_http_port,
			"The web api port, the miner should listen to. Use 0 to disable. (default: 0). Data shown depends on hwmon setting.")
			->group(APIGroup)
			->check(CLI::Range(1, 32767))
			->set_type_name("<n>");

#endif

#if ETH_ETHASHCL || ETH_ETHASHCUDA

		app.add_flag("--list-devices", m_shouldListDevices,
			"List the detected OpenCL/CUDA devices and exit. Should be combined with -G, -U, or -X flag")
			->group(CommonGroup);

#endif
		stringstream ssHelp;

#if ETH_ETHASHCL

		app.add_option("--opencl-platform", m_openclPlatform,
			"When mining using -G/--opencl use OpenCL platform n (default: 0)")
			->group(OpenCLGroup)
			->set_type_name("<n>");

		app.add_option("--opencl-device,--opencl-devices", m_openclDevices,
			"Select which OpenCL devices to mine on. Default is to use all")
			->group(OpenCLGroup)
			->set_type_name("<n>");

		app.add_set("--cl-parallel-hash", m_openclThreadsPerHash, {1, 2, 4, 8 },
			"Define how many threads to associate per hash (default: 8)")
			->group(OpenCLGroup)
			->set_type_name("<1|2|4|8>");

		app.add_option("--cl-kernel", m_openclSelectedKernel,
			"Select opencl kernel. (0: stable kernel, 1: experimental kernel)")
			->group(OpenCLGroup)
			->check(CLI::Range(1))
			->set_type_name("<0|1>");

		ssHelp.str("");
		ssHelp <<
			"The OpenCL global work size multipler of the local work size. (default: " <<
			CLMiner::c_defaultGlobalWorkSizeMultiplier << ')';
		app.add_option("--cl-global-work", m_globalWorkSizeMultiplier, ssHelp.str())
			->group(OpenCLGroup)
			->check(CLI::Range(1, 999999999))
			->set_type_name("<n>");

		ssHelp.str("");
		ssHelp <<
			"the OpenCL local work size (default: " << CLMiner::c_defaultLocalWorkSize << ')';
		app.add_option("--cl-local-work", m_localWorkSize, ssHelp.str())
			->group(OpenCLGroup)
			->check(CLI::Range(32, 99999))
			->set_type_name("<n>");

#endif

#if ETH_ETHASHCUDA

		ssHelp.str("");
		ssHelp <<
			"the CUDA grid size (default: " << toString(CUDAMiner::c_defaultGridSize) << ')';
		app.add_option("--cuda-grid-size", m_cudaGridSize, ssHelp.str())
			->group(CUDAGroup)
			->check(CLI::Range(1, 999999999))
			->set_type_name("<n>");

		ssHelp.str("");
		ssHelp <<
			"the CUDA block size (default: " << toString(CUDAMiner::c_defaultBlockSize) << ')';
		app.add_option("--cuda-block-size", m_cudaBlockSize, ssHelp.str())
			->group(CUDAGroup)
			->check(CLI::Range(1, 999999999))
			->set_type_name("<n>");

		app.add_option("--cuda-devices", m_cudaDevices,
			"Select which CUDA devices to mine on. Default is to use all")
			->group(CUDAGroup)
			->set_type_name("<n>");

		app.add_option("--cuda-parallel-hash", m_cudaParallelHash,
			"Define how many hashes to calculate in a kernel (default: 4)")
			->group(CUDAGroup)
			->check(CLI::Range(1, 8))
			->set_type_name("<n>");

		string sched = "sync";
		app.add_set("--cuda-schedule", sched, {"auto", "spin", "yield", "sync"},
			"CUDA scheduler mode")
			->group(CUDAGroup)
			->set_type_name("<auto|spin|yield|sync>");

		ssHelp.str("");
		ssHelp <<
			"The number of CUDA streams (default: " << toString(CUDAMiner::c_defaultNumStreams)	<< ')';
		app.add_option("--cuda-streams", m_numStreams, ssHelp.str())
			->group(CUDAGroup)
			->check(CLI::Range(1, 99))
			->set_type_name("<n>");

#endif
		app.add_flag("--noeval", m_noEval,
			"Bypass host software re-evaluation of GPU solutions")
			->group(CommonGroup);

		app.add_option("-L,--dag-load-mode", m_dagLoadMode,
			"DAG load mode. 0=parallel, 1=sequential, 2=sequential (default: 0)")
			->group(CommonGroup)
			->check(CLI::Range(2))
			->set_type_name("<0|1|2>");

		app.add_option("--dag-single-dev", m_dagCreateDevice,
			"Device to create DAG in single mode (default: 0)")
			->group(CommonGroup)
			->set_type_name("<n>");

		app.add_option("--benchmark-warmup", m_benchmarkWarmup,
			"Set the duration in seconds of warmup for the benchmark tests (default: 3)")
			->group(CommonGroup)
			->set_type_name("<n>");

		app.add_option("--benchmark-trial", m_benchmarkTrial,
			"Set the number of benchmark trials to run (default: 5)")
			->group(CommonGroup)
			->check(CLI::Range(1, 99))
			->set_type_name("<n>");

		bool cl_miner = false;
		app.add_flag("-G,--opencl", cl_miner,
			"When mining use the GPU via OpenCL")
			->group(CommonGroup);

		bool cuda_miner = false;
		app.add_flag("-U,--cuda", cuda_miner,
			"When mining use the GPU via CUDA")
			->group(CommonGroup);

		bool mixed_miner = false;
		app.add_flag("-X,--cuda-opencl", mixed_miner,
			"When mining with mixed AMD and CUDA GPUs")
			->group(CommonGroup);

		auto bench_opt = app.add_option("-M,--benchmark", m_benchmarkBlock,
			"Benchmark for mining and exit; Specify block number to benchmark against specific DAG");
			bench_opt->group(CommonGroup)
				->set_type_name("<n>");

		auto sim_opt = app.add_option("-Z,--simulation", m_benchmarkBlock,
			"Mining test. Used to validate kernel optimizations. Specify block number");
			sim_opt->group(CommonGroup)
				->set_type_name("<n>");

		app.add_option("-t,--mining-threads", m_miningThreads,
			"Limit number of CPU/GPU miners")
			->group(CommonGroup)
			->check(CLI::Range(1, 99))
			->set_type_name("<n>");

		app.add_option("--tstop", m_tstop,
			"Stop mining on a GPU if temperature exceeds value (valid: 30..100)")
			->group(CommonGroup)
			->check(CLI::Range(30, 100))
			->set_type_name("<n>");

		app.add_option("--tstart", m_tstart,
			"Restart mining on a GPU if the temperature drops below (default: 40, valid: 30..100)")
			->group(CommonGroup)
			->check(CLI::Range(30, 100))
			->set_type_name("<n>");

		ssHelp.str("");
		ssHelp
            << "Pool URL Specification:" << endl
            << "    URL takes the form: scheme://user[:password]@hostname:port[/emailaddress]." << endl
            << "    for getwork use one of the following schemes:" << endl
            << "      " << URI::KnownSchemes(ProtocolFamily::GETWORK) << endl
            << "    for stratum use one of the following schemes: "<< endl
            << "      " << URI::KnownSchemes(ProtocolFamily::STRATUM) << endl
            << "    Example 1 : stratum+ssl://0x012345678901234567890234567890123.miner1@ethermine.org:5555" << endl
            << "    Example 2 : stratum1+tcp://0x012345678901234567890234567890123.miner1@nanopool.org:9999/john.doe@gmail.com" << endl
            << "    Example 3 : stratum1+tcp://0x012345678901234567890234567890123@nanopool.org:9999/miner1/john.doe@gmail.com"
			<< endl << endl
			<< "Environment Variables:" << endl
     		<< "    NO_COLOR - set to any value to disable color output. Unset to re-enable color output." << endl
     		<< "    SYSLOG   - set to any value to strip time and disable color from output, for logging under systemd";
		app.set_footer(ssHelp.str());

    	try {
        	app.parse(argc, argv);
			if (help) {
				cerr << endl << app.help() << endl;
				exit(0);
			}
			else if (version) {
			    auto* bi = ethminer_get_buildinfo();
    			cerr << "\nethminer " << bi->project_version << "\nBuild: " << bi->system_name << "/"
         			<< bi->build_type << "/" << bi->compiler_id << "\n\n";
    				exit(0);
			}
    	} catch(const CLI::ParseError &e) {
			cerr << endl << e.what() << endl << endl;
			exit(-1);
    	}

		if (hwmon_opt->count()) {
			m_show_hwmonitors = true;
			if (hwmon)
				m_show_power = true;
		}

		for (auto url : pools) {
			if (url == "exit") // add fake scheme and port to 'exit' url
				url = "stratum+tcp://-:x@exit:0";
			URI uri;
			try {
				uri = url;
			}
			catch (...) {
				cerr << "Bad endpoint address: " << url << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
			if (!uri.KnownScheme())
			{
				cerr << "Unknown URI scheme " << uri.Scheme() << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
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
				cerr << "Mixed stratum and getwork endpoints not supported." << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
			m_mode = mode;
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

		if (!cl_miner && !cuda_miner && !mixed_miner && !bench_opt->count() && !sim_opt->count())
		{
			cerr << endl << "One of -G, -U, -X, -M, or -Z must be specified" << endl << endl;
			exit(-1);
		}

		if (cl_miner)
			m_minerType = MinerType::CL;
		else if (cuda_miner)
			m_minerType = MinerType::CUDA;
		else if (mixed_miner)
			m_minerType = MinerType::Mixed;
		else if (bench_opt->count())
			m_mode = OperationMode::Benchmark;
		else if (sim_opt->count())
			m_mode = OperationMode::Simulation;
		if (m_tstop && (m_tstop <= m_tstart))
		{
			cerr << endl << "tstop must be greater than tstart" << endl << endl;
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

			CLMiner::setCLKernel(m_openclSelectedKernel);
			CLMiner::setThreadsPerHash(m_openclThreadsPerHash);

			if (!CLMiner::configureGPU(
				m_localWorkSize,
				m_globalWorkSizeMultiplier,
				m_openclPlatform,
				0,
				m_dagLoadMode,
				m_dagCreateDevice,
				m_noEval,
				m_exit
			)) {
				stop_io_service();
				exit(1);
			};

			CLMiner::setNumInstances(m_miningThreads);
#else
			cerr << "Selected GPU mining without having compiled with -DETHASHCL=1" << endl;
			exit(1);
#endif
		}
		if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
		{
#if ETH_ETHASHCUDA
			if (m_cudaDeviceCount > 0)
			{
				CUDAMiner::setDevices(m_cudaDevices, m_cudaDeviceCount);
				m_miningThreads = m_cudaDeviceCount;
			}

			CUDAMiner::setNumInstances(m_miningThreads);
			if (!CUDAMiner::configureGPU(
				m_cudaBlockSize,
				m_cudaGridSize,
				m_numStreams,
				m_cudaSchedule,
				m_dagLoadMode,
				m_dagCreateDevice,
				m_noEval,
				m_exit
			))
			{
				stop_io_service();
				exit(1);
			}

			CUDAMiner::setParallelHash(m_cudaParallelHash);
#else
			cerr << "CUDA support disabled. Configure project build with -DETHASHCUDA=ON" << endl;
			stop_io_service();
			exit(1);
#endif
		}

		g_running = true;
		signal(SIGINT, MinerCLI::signalHandler);
		signal(SIGTERM, MinerCLI::signalHandler);

		if (m_mode == OperationMode::Benchmark)
			doBenchmark(m_minerType, m_benchmarkWarmup, m_benchmarkTrial, m_benchmarkTrials);
		else if (m_mode == OperationMode::Farm || m_mode == OperationMode::Stratum || m_mode == OperationMode::Simulation)
			doMiner();
	}

private:

	void doBenchmark(MinerType _m, unsigned _warmupDuration = 15, unsigned _trialDuration = 3, unsigned _trials = 5)
	{
		BlockHeader genesis;
		genesis.setNumber(m_benchmarkBlock);
		genesis.setDifficulty(u256(1) << 64);

		Farm f(m_io_service);
		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor{
			&CLMiner::instances, [](FarmFace& _farm, unsigned _index){ return new CLMiner(_farm, _index); }
		};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{
			&CUDAMiner::instances, [](FarmFace& _farm, unsigned _index){ return new CUDAMiner(_farm, _index); }
		};
#endif
		f.setSealers(sealers);
		f.onSolutionFound([&](Solution) { return false; });

		f.setTStartTStop(m_tstart, m_tstop);

		string platformInfo = _m == MinerType::CL ? "CL" : "CUDA";
		cout << "Benchmarking on platform: " << platformInfo << endl;

		cout << "Preparing DAG for block #" << m_benchmarkBlock << endl;
		//genesis.prep();

		if (_m == MinerType::CL)
			f.start("opencl", false);
		else if (_m == MinerType::CUDA)
			f.start("cuda", false);

		WorkPackage current = WorkPackage(genesis);
		

		vector<uint64_t> results;
		results.reserve(_trials);
		uint64_t mean = 0;
		uint64_t innerMean = 0;
		for (unsigned i = 0; i <= _trials; ++i)
		{
			current.header = h256::random();
			current.boundary = genesis.boundary();
			f.setWork(current);	
			if (!i)
				cout << "Warming up..." << endl;
			else
				cout << "Trial " << i << "... " << flush <<endl;
			this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));

			auto mp = f.miningProgress();
			if (!i)
				continue;
			auto rate = mp.rate();

			cout << rate << endl;
			results.push_back(rate);
			mean += rate;
		}
		sort(results.begin(), results.end());
		cout << "min/mean/max: " << results.front() << "/" << (mean / _trials) << "/" << results.back() << " H/s" << endl;
		if (results.size() > 2) {
			for (auto it = results.begin()+1; it != results.end()-1; it++)
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
		sealers["opencl"] = Farm::SealerDescriptor{&CLMiner::instances, [](FarmFace& _farm, unsigned _index){ return new CLMiner(_farm, _index); }};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{&CUDAMiner::instances, [](FarmFace& _farm, unsigned _index){ return new CUDAMiner(_farm, _index); }};
#endif

		PoolClient *client = nullptr;

		if (m_mode == OperationMode::Stratum) {
			client = new EthStratumClient(m_io_service, m_worktimeout, m_responsetimeout, m_email, m_report_stratum_hashrate);
		}
		else if (m_mode == OperationMode::Farm) {
			client = new EthGetworkClient(m_farmRecheckPeriod);
		}
		else if (m_mode == OperationMode::Simulation) {
			client = new SimulateClient(20, m_benchmarkBlock);
		}
		else {
			cwarn << "Invalid OperationMode";
			exit(1);
		}

		// Should not happen!
		if (!client) {
			cwarn << "Invalid PoolClient";
			stop_io_service();
			exit(1);
		}

		//sealers, m_minerType
		Farm f(m_io_service);
		f.setSealers(sealers);

		PoolManager mgr(client, f, m_minerType, m_maxFarmRetries);

		f.setTStartTStop(m_tstart, m_tstop);

		// If we are in simulation mode we add a fake connection
		if (m_mode == OperationMode::Simulation) {
			URI con(URI("http://-:0"));
			mgr.clearConnections();
			mgr.addConnection(con);
		}
		else {
			for (auto conn : m_endpoints)
                mgr.addConnection(conn);
		}

#if API_CORE
		Api api(this->m_api_port, f);
        http_server.run(m_http_port, &f, m_show_hwmonitors, m_show_power);
#endif

		// Start PoolManager
		mgr.start();

		// Run CLI in loop
		while (g_running && mgr.isRunning()) {
			if (mgr.isConnected()) {
				auto mp = f.miningProgress(m_show_hwmonitors, m_show_power);
				minelog << mp << f.getSolutionStats() << f.farmLaunchedFormatted();

#if ETH_DBUS
				dbusint.send(toString(mp).data());
#endif
			}
			else {
				minelog << "not-connected";
			}
			this_thread::sleep_for(chrono::seconds(m_displayInterval));
		}

		mgr.stop();
		stop_io_service();

		cnote << "Terminated !";
		exit(0);
	}

	/// Operating mode.
	OperationMode m_mode = OperationMode::None;

	/// Global boost's io_service
	std::thread m_io_thread;									// The IO service thread
	boost::asio::io_service m_io_service;						// The IO service itself
	boost::asio::io_service::work m_io_work;					// The IO work which prevents io_service.run() to return on no work thus terminating thread
	boost::asio::deadline_timer m_io_work_timer;				// A dummy timer to keep io_service with something to do and prevent io shutdown
	boost::asio::io_service::strand m_io_strand;				// A strand to serialize posts in multithreaded environment

	/// Mining options
	MinerType m_minerType = MinerType::Mixed;
	unsigned m_openclPlatform = 0;
	unsigned m_miningThreads = UINT_MAX;
	bool m_shouldListDevices = false;
#if ETH_ETHASHCL
	unsigned m_openclSelectedKernel = 0;  ///< A numeric value for the selected OpenCL kernel
	unsigned m_openclDeviceCount = 0;
	vector<unsigned> m_openclDevices;
	unsigned m_openclThreadsPerHash = 8;
	unsigned m_globalWorkSizeMultiplier = CLMiner::c_defaultGlobalWorkSizeMultiplier;
	unsigned m_localWorkSize = CLMiner::c_defaultLocalWorkSize;
#endif
#if ETH_ETHASHCUDA
	unsigned m_cudaDeviceCount = 0;
	vector<unsigned> m_cudaDevices;
	unsigned m_numStreams = CUDAMiner::c_defaultNumStreams;
	unsigned m_cudaSchedule = 4; // sync
	unsigned m_cudaGridSize = CUDAMiner::c_defaultGridSize;
	unsigned m_cudaBlockSize = CUDAMiner::c_defaultBlockSize;
	unsigned m_cudaParallelHash    = 4;
#endif
	bool m_noEval = false;
	unsigned m_dagLoadMode = 0; // parallel
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

	bool m_show_hwmonitors = false;
	bool m_show_power = false;

	unsigned m_tstop = 0;
	unsigned m_tstart = 40;

#if API_CORE
	int m_api_port = 0;
	unsigned m_http_port = 0;
#endif

	bool m_report_stratum_hashrate = false;
	string m_email;

#if ETH_DBUS
	DBusInt dbusint;
#endif
};
