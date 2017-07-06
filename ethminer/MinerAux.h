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

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/optional.hpp>

#include <libethcore/Exceptions.h>
#include <libdevcore/SHA3.h>
#include <libethcore/EthashAux.h>
#include <libethcore/EthashCUDAMiner.h>
#include <libethcore/EthashGPUMiner.h>
#include <libethcore/Farm.h>
#if ETH_ETHASHCL
#include <libethash-cl/ethash_cl_miner.h>
#endif
#if ETH_ETHASHCUDA
#include <libethash-cuda/ethash_cuda_miner.h>
#endif
#include <jsonrpccpp/client/connectors/httpclient.h>
#include "FarmClient.h"
#if ETH_STRATUM
#include <libstratum/EthStratumClient.h>
#include <libstratum/EthStratumClientV2.h>
#endif
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;

#undef RETURN

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

	MinerCLI(OperationMode _mode = OperationMode::None): mode(_mode) {}

	bool interpretOption(int& i, int argc, char** argv)
	{
		string arg = argv[i];
		if ((arg == "-F" || arg == "--farm") && i + 1 < argc)
		{
			mode = OperationMode::Farm;
			m_farmURL = argv[++i];
			m_activeFarmURL = m_farmURL;
		}
		else if ((arg == "-FF" || arg == "-FS" || arg == "--farm-failover" || arg == "--stratum-failover") && i + 1 < argc)
		{
			string url = argv[++i];

			if (mode == OperationMode::Stratum)
			{
				size_t p = url.find_last_of(":");
				if (p > 0)
				{
					m_farmFailOverURL = url.substr(0, p);
					if (p + 1 <= url.length())
						m_fport = url.substr(p + 1);
				}
				else
				{
					m_farmFailOverURL = url;
				}
			}
			else
			{
				m_farmFailOverURL = url;
			}
		}
		else if (arg == "--farm-recheck" && i + 1 < argc)
			try {
				m_farmRecheckSet = true;
				m_farmRecheckPeriod = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--farm-retries" && i + 1 < argc)
			try {
				m_maxFarmRetries = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
#if ETH_STRATUM
		else if ((arg == "-S" || arg == "--stratum") && i + 1 < argc)
		{
			mode = OperationMode::Stratum;
			string url = string(argv[++i]);
			size_t p = url.find_last_of(":");
			if (p > 0)
			{
				m_farmURL = url.substr(0, p);
				if (p + 1 <= url.length())
					m_port = url.substr(p+1);
			}
			else
			{
				m_farmURL = url;
			}
		}
		else if ((arg == "-O" || arg == "--userpass") && i + 1 < argc)
		{
			string userpass = string(argv[++i]);
			size_t p = userpass.find_first_of(":");
			m_user = userpass.substr(0, p);
			if (p + 1 <= userpass.length())
				m_pass = userpass.substr(p+1);
		}
		else if ((arg == "-SC" || arg == "--stratum-client") && i + 1 < argc)
		{
			try {
				m_stratumClientVersion = atoi(argv[++i]);
				if (m_stratumClientVersion > 2) m_stratumClientVersion = 2;
				else if (m_stratumClientVersion < 1) m_stratumClientVersion = 1;
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if ((arg == "-SP" || arg == "--stratum-protocol") && i + 1 < argc)
		{
			try {
				m_stratumProtocol = atoi(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if ((arg == "-SE" || arg == "--stratum-email") && i + 1 < argc)
		{
			try {
				m_email = string(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if ((arg == "-FO" || arg == "--failover-userpass") && i + 1 < argc)
		{
			string userpass = string(argv[++i]);
			size_t p = userpass.find_first_of(":");
			m_fuser = userpass.substr(0, p);
			if (p + 1 <= userpass.length())
				m_fpass = userpass.substr(p + 1);
		}
		else if ((arg == "-u" || arg == "--user") && i + 1 < argc)
		{
			m_user = string(argv[++i]);
		}
		else if ((arg == "-p" || arg == "--pass") && i + 1 < argc)
		{
			m_pass = string(argv[++i]);
		}
		else if ((arg == "-o" || arg == "--port") && i + 1 < argc)
		{
			m_port = string(argv[++i]);
		}
		else if ((arg == "-fu" || arg == "--failover-user") && i + 1 < argc)
		{
			m_fuser = string(argv[++i]);
		}
		else if ((arg == "-fp" || arg == "--failover-pass") && i + 1 < argc)
		{
			m_fpass = string(argv[++i]);
		}
		else if ((arg == "-fo" || arg == "--failover-port") && i + 1 < argc)
		{
			m_fport = string(argv[++i]);
		}
		else if ((arg == "--work-timeout") && i + 1 < argc)
		{
			m_worktimeout = atoi(argv[++i]);
		}

#endif
#if ETH_ETHASHCL
		else if (arg == "--opencl-platform" && i + 1 < argc)
			try {
				m_openclPlatform = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--opencl-devices" || arg == "--opencl-device")
			while (m_openclDeviceCount < 16 && i + 1 < argc)
			{
				try
				{
					m_openclDevices[m_openclDeviceCount] = stol(argv[++i]);
					++m_openclDeviceCount;
				}
				catch (...)
				{
					i--;
					break;
				}
			}
#endif
#if ETH_ETHASHCL || ETH_ETHASHCUDA
		else if ((arg == "--cl-global-work" || arg == "--cuda-grid-size")  && i + 1 < argc)
			try {
				m_globalWorkSizeMultiplier = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if ((arg == "--cl-local-work" || arg == "--cuda-block-size") && i + 1 < argc)
			try {
				m_localWorkSize = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--list-devices")
			m_shouldListDevices = true;
		else if ((arg == "--cl-extragpu-mem" || arg == "--cuda-extragpu-mem") && i + 1 < argc)
			m_extraGPUMemory = 1000000 * stol(argv[++i]);
#endif
#if ETH_ETHASHCUDA
		else if (arg == "--cuda-devices")
		{
			while (m_cudaDeviceCount < 16 && i + 1 < argc)
			{
				try
				{
					m_cudaDevices[m_cudaDeviceCount] = stol(argv[++i]);
					++m_cudaDeviceCount;
				}
				catch (...)
				{
					i--;
					break;
				}
			}
		}
                else if (arg == "--cuda-parallel-hash" && i + 1 < argc)
                {
                        try {
                                m_parallelHash = stol(argv[++i]);
                                if (m_parallelHash == 0 || m_parallelHash > 8)
                                {
                                    throw BadArgument();
                                }
                        }
                        catch (...)
                        {
                                cerr << "Bad " << arg << " option: " << argv[i] << endl;
                                BOOST_THROW_EXCEPTION(BadArgument());
                        }
                }
		else if (arg == "--cuda-schedule" && i + 1 < argc)
		{
			string mode = argv[++i];
			if (mode == "auto") m_cudaSchedule = 0;
			else if (mode == "spin") m_cudaSchedule = 1;
			else if (mode == "yield") m_cudaSchedule = 2;
			else if (mode == "sync") m_cudaSchedule = 4;
			else
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if (arg == "--cuda-streams" && i + 1 < argc)
			m_numStreams = stol(argv[++i]);
#endif
		else if ((arg == "-L" || arg == "--dag-load-mode") && i + 1 < argc)
		{
			string mode = argv[++i];
			if (mode == "parallel") m_dagLoadMode = DAG_LOAD_MODE_PARALLEL;
			else if (mode == "sequential") m_dagLoadMode = DAG_LOAD_MODE_SEQUENTIAL;
			else if (mode == "single")
			{
				m_dagLoadMode = DAG_LOAD_MODE_SINGLE;
				m_dagCreateDevice = stol(argv[++i]);
			}
			else
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else if (arg == "--benchmark-warmup" && i + 1 < argc)
			try {
				m_benchmarkWarmup = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--benchmark-trial" && i + 1 < argc)
			try {
				m_benchmarkTrial = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "--benchmark-trials" && i + 1 < argc)
			try
			{
				m_benchmarkTrials = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		else if (arg == "-G" || arg == "--opencl")
			m_minerType = MinerType::CL;
		else if (arg == "-U" || arg == "--cuda")
		{
			m_minerType = MinerType::CUDA;
		}
		else if (arg == "-X" || arg == "--cuda-opencl")
		{
			m_minerType = MinerType::Mixed;
		}
		else if (arg == "-M" || arg == "--benchmark")
		{
			mode = OperationMode::Benchmark;
			if (i + 1 < argc)
			{
				string m = boost::to_lower_copy(string(argv[++i]));
				try
				{
					m_benchmarkBlock = stol(m);
				}
				catch (...)
				{
					if (argv[i][0] == 45) { // check next arg
						i--;
					}
					else {
						cerr << "Bad " << arg << " option: " << argv[i] << endl;
						BOOST_THROW_EXCEPTION(BadArgument());
					}
				}
			}
		}
		else if (arg == "-Z" || arg == "--simulation") {
			mode = OperationMode::Simulation;
			if (i + 1 < argc)
			{
				string m = boost::to_lower_copy(string(argv[++i]));
				try
				{
					m_benchmarkBlock = stol(m);
				}
				catch (...)
				{
					if (argv[i][0] == 45) { // check next arg
						i--;
					}
					else {
						cerr << "Bad " << arg << " option: " << argv[i] << endl;
						BOOST_THROW_EXCEPTION(BadArgument());
					}
				}
			}
		}
		else if ((arg == "-t" || arg == "--mining-threads") && i + 1 < argc)
		{
			try
			{
				m_miningThreads = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				BOOST_THROW_EXCEPTION(BadArgument());
			}
		}
		else
			return false;
		return true;
	}

	void execute()
	{
		if (m_shouldListDevices)
		{
#if ETH_ETHASHCL
			if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
				EthashGPUMiner::listDevices();
#endif
#if ETH_ETHASHCUDA
			if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
				EthashCUDAMiner::listDevices();
#endif
			exit(0);
		}

		if (m_minerType == MinerType::CL || m_minerType == MinerType::Mixed)
		{
#if ETH_ETHASHCL
			if (m_openclDeviceCount > 0)
			{
				EthashGPUMiner::setDevices(m_openclDevices, m_openclDeviceCount);
				m_miningThreads = m_openclDeviceCount;
			}

			if (!EthashGPUMiner::configureGPU(
					m_localWorkSize,
					m_globalWorkSizeMultiplier,
					m_openclPlatform,
					m_openclDevice,
					m_extraGPUMemory,
					0,
					m_dagLoadMode,
					m_dagCreateDevice
				))
				exit(1);
			EthashGPUMiner::setNumInstances(m_miningThreads);
#else
			cerr << "Selected GPU mining without having compiled with -DETHASHCL=1" << endl;
			exit(1);
#endif
		}
		else if (m_minerType == MinerType::CUDA || m_minerType == MinerType::Mixed)
		{
#if ETH_ETHASHCUDA
			if (m_cudaDeviceCount > 0)
			{
				EthashCUDAMiner::setDevices(m_cudaDevices, m_cudaDeviceCount);
				m_miningThreads = m_cudaDeviceCount;
			}

			EthashCUDAMiner::setNumInstances(m_miningThreads);
			if (!EthashCUDAMiner::configureGPU(
				m_localWorkSize,
				m_globalWorkSizeMultiplier,
				m_numStreams,
				m_extraGPUMemory,
				m_cudaSchedule,
				0,
				m_dagLoadMode,
				m_dagCreateDevice
				))
				exit(1);

			EthashCUDAMiner::setParallelHash(m_parallelHash);
#else
			cerr << "CUDA support disabled. Configure project build with -DETHASHCUDA=ON" << endl;
			exit(1);
#endif
		}
		if (mode == OperationMode::Benchmark)
			doBenchmark(m_minerType, m_benchmarkWarmup, m_benchmarkTrial, m_benchmarkTrials);
		else if (mode == OperationMode::Farm)
			doFarm(m_minerType, m_activeFarmURL, m_farmRecheckPeriod);
		else if (mode == OperationMode::Simulation)
			doSimulation(m_minerType);
#if ETH_STRATUM
		else if (mode == OperationMode::Stratum)
			doStratum();
#endif
	}

	static void streamHelp(ostream& _out)
	{
		_out
			<< "Work farming mode:" << endl
			<< "    -F,--farm <url>  Put into mining farm mode with the work server at URL (default: http://127.0.0.1:8545)" << endl
			<< "    -FF,-FO, --farm-failover, --stratum-failover <url> Failover getwork/stratum URL (default: disabled)" << endl
			<< "	--farm-retries <n> Number of retries until switch to failover (default: 3)" << endl
#if ETH_STRATUM
			<< "	-S, --stratum <host:port>  Put into stratum mode with the stratum server at host:port" << endl
			<< "	-FS, --failover-stratum <host:port>  Failover stratum server at host:port" << endl
			<< "    -O, --userpass <username.workername:password> Stratum login credentials" << endl
			<< "    -FO, --failover-userpass <username.workername:password> Failover stratum login credentials (optional, will use normal credentials when omitted)" << endl
			<< "    --work-timeout <n> reconnect/failover after n seconds of working on the same (stratum) job. Defaults to 180. Don't set lower than max. avg. block time" << endl
			<< "    -SC, --stratum-client <n>  Stratum client version. Defaults to 1 (async client). Use 2 to use the new synchronous client." << endl
			<< "    -SP, --stratum-protocol <n> Choose which stratum protocol to use:" << endl
			<< "        0: official stratum spec: ethpool, ethermine, coinotron, mph, nanopool (default)" << endl
			<< "        1: eth-proxy compatible: dwarfpool, f2pool, nanopool" << endl
			<< "        2: EthereumStratum/1.0.0: nicehash" << endl
			<< "    -SE, --stratum-email <s> Email address used in eth-proxy (optional)" << endl
#endif
#if ETH_STRATUM
			<< "    --farm-recheck <n>  Leave n ms between checks for changed work (default: 500). When using stratum, use a high value (i.e. 2000) to get more stable hashrate output" << endl
#endif
			<< endl
			<< "Benchmarking mode:" << endl
			<< "    -M [<n>],--benchmark [<n>] Benchmark for mining and exit; Optionally specify block number to benchmark against specific DAG." << endl
			<< "    --benchmark-warmup <seconds>  Set the duration of warmup for the benchmark tests (default: 3)." << endl
			<< "    --benchmark-trial <seconds>  Set the duration for each trial for the benchmark tests (default: 3)." << endl
			<< "    --benchmark-trials <n>  Set the duration of warmup for the benchmark tests (default: 5)." << endl
			<< "Simulation mode:" << endl
			<< "    -Z [<n>],--simulation [<n>] Mining test mode. Used to validate kernel optimizations. Optionally specify block number." << endl
			<< "Mining configuration:" << endl
			<< "    -G,--opencl  When mining use the GPU via OpenCL." << endl
			<< "    -U,--cuda  When mining use the GPU via CUDA." << endl
			<< "    -X,--cuda-opencl Use OpenCL + CUDA in a system with mixed AMD/Nvidia cards. May require setting --opencl-platform 1" << endl
			<< "    --opencl-platform <n>  When mining using -G/--opencl use OpenCL platform n (default: 0)." << endl
			<< "    --opencl-device <n>  When mining using -G/--opencl use OpenCL device n (default: 0)." << endl
			<< "    --opencl-devices <0 1 ..n> Select which OpenCL devices to mine on. Default is to use all" << endl
			<< "    -t, --mining-threads <n> Limit number of CPU/GPU miners to n (default: use everything available on selected platform)" << endl
			<< "    --list-devices List the detected OpenCL/CUDA devices and exit. Should be combined with -G or -U flag" << endl
			<< "    -L, --dag-load-mode <mode> DAG generation mode." << endl
			<< "        parallel    - load DAG on all GPUs at the same time (default)" << endl
			<< "        sequential  - load DAG on GPUs one after another. Use this when the miner crashes during DAG generation" << endl
			<< "        single <n>  - generate DAG on device n, then copy to other devices" << endl
#if ETH_ETHASHCL
			<< "    --cl-extragpu-mem Set the memory (in MB) you believe your GPU requires for stuff other than mining. default: 0" << endl
			<< "    --cl-local-work Set the OpenCL local work size. Default is " << toString(ethash_cl_miner::c_defaultLocalWorkSize) << endl
			<< "    --cl-global-work Set the OpenCL global work size as a multiple of the local work size. Default is " << toString(ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier) << " * " << toString(ethash_cl_miner::c_defaultLocalWorkSize) << endl
#endif
#if ETH_ETHASHCUDA
			<< "    --cuda-extragpu-mem Set the memory (in MB) you believe your GPU requires for stuff other than mining. Windows rendering e.t.c.." << endl
			<< "    --cuda-block-size Set the CUDA block work size. Default is " << toString(ethash_cuda_miner::c_defaultBlockSize) << endl
			<< "    --cuda-grid-size Set the CUDA grid size. Default is " << toString(ethash_cuda_miner::c_defaultGridSize) << endl
			<< "    --cuda-streams Set the number of CUDA streams. Default is " << toString(ethash_cuda_miner::c_defaultNumStreams) << endl
			<< "    --cuda-schedule <mode> Set the schedule mode for CUDA threads waiting for CUDA devices to finish work. Default is 'sync'. Possible values are:" << endl
			<< "        auto  - Uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical processors in the system P. If C > P, then yield else spin." << endl
			<< "        spin  - Instruct CUDA to actively spin when waiting for results from the device." << endl
			<< "        yield - Instruct CUDA to yield its thread when waiting for results from the device." << endl
			<< "        sync  - Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the results from the device." << endl
			<< "    --cuda-devices <0 1 ..n> Select which CUDA GPUs to mine on. Default is to use all" << endl
			<< "    --cuda-parallel-hash <1 2 ..8> Define how many hashes to calculate in a kernel, can be scaled to achive better performance. Default=4" << endl
#endif
			;
	}

private:

	void doBenchmark(MinerType _m, unsigned _warmupDuration = 15, unsigned _trialDuration = 3, unsigned _trials = 5)
	{
		BlockHeader genesis;
		genesis.setNumber(m_benchmarkBlock);
		genesis.setDifficulty(1 << 18);
		cdebug << genesis.boundary();

		Farm f;
		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor{&EthashGPUMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{ &EthashCUDAMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		f.setSealers(sealers);
		f.onSolutionFound([&](Solution) { return false; });

		string platformInfo = _m == MinerType::CL ? "CL" : "CUDA";
		cout << "Benchmarking on platform: " << platformInfo << endl;

		cout << "Preparing DAG for block #" << m_benchmarkBlock << endl;
		//genesis.prep();

		genesis.setDifficulty(u256(1) << 63);
		f.setWork(genesis);
		if (_m == MinerType::CL)
			f.start("opencl", false);
		else if (_m == MinerType::CUDA)
			f.start("cuda", false);

		map<uint64_t, WorkingProgress> results;
		uint64_t mean = 0;
		uint64_t innerMean = 0;
		for (unsigned i = 0; i <= _trials; ++i)
		{
			if (!i)
				cout << "Warming up..." << endl;
			else
				cout << "Trial " << i << "... " << flush;
			this_thread::sleep_for(chrono::seconds(i ? _trialDuration : _warmupDuration));

			auto mp = f.miningProgress();
			f.resetMiningProgress();
			if (!i)
				continue;
			auto rate = mp.rate();

			cout << rate << endl;
			results[rate] = mp;
			mean += rate;
		}
		f.stop();
		int j = -1;
		for (auto const& r: results)
			if (++j > 0 && j < (int)_trials - 1)
				innerMean += r.second.rate();
		innerMean /= (_trials - 2);
		cout << "min/mean/max: " << results.begin()->second.rate() << "/" << (mean / _trials) << "/" << results.rbegin()->second.rate() << " H/s" << endl;
		cout << "inner mean: " << innerMean << " H/s" << endl;

		exit(0);
	}

	void doSimulation(MinerType _m, int difficulty = 20)
	{
		BlockHeader genesis;
		genesis.setNumber(m_benchmarkBlock);
		genesis.setDifficulty(1 << 18);
		cdebug << genesis.boundary();

		Farm f;
		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor{ &EthashGPUMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashGPUMiner(ci); } };
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{ &EthashCUDAMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		f.setSealers(sealers);

		string platformInfo = _m == MinerType::CL ? "CL" : "CUDA";
		cout << "Running mining simulation on platform: " << platformInfo << endl;

		cout << "Preparing DAG for block #" << m_benchmarkBlock << endl;
		//genesis.prep();

		genesis.setDifficulty(u256(1) << difficulty);
		f.setWork(genesis);

		if (_m == MinerType::CL)
			f.start("opencl", false);
		else if (_m == MinerType::CUDA)
			f.start("cuda", false);

		int time = 0;

		WorkPackage current = WorkPackage(genesis);
		while (true) {
			bool completed = false;
			Solution solution;
			f.onSolutionFound([&](Solution sol)
			{
				solution = sol;
				return completed = true;
			});
			for (unsigned i = 0; !completed; ++i)
			{
				auto mp = f.miningProgress();
				f.resetMiningProgress();

				cnote << "Mining on difficulty " << difficulty << " " << mp;
				this_thread::sleep_for(chrono::milliseconds(1000));
				time++;
			}
			cnote << "Difficulty:" << difficulty << "  Nonce:" << solution.nonce.hex();
			if (EthashAux::eval(current.seedHash, current.headerHash, solution.nonce).value < current.boundary)
			{
				cnote << "SUCCESS: GPU gave correct result!";
			}
			else
				cwarn << "FAILURE: GPU gave incorrect result!";

			if (time < 12)
				difficulty++;
			else if (time > 18)
				difficulty--;
			time = 0;
			genesis.setDifficulty(u256(1) << difficulty);
			genesis.noteDirty();

			h256 hh;
			std::random_device engine;
			hh.randomize(engine);

			current.headerHash = hh;
			current.boundary = genesis.boundary();
			minelog << "Generated random work package:";
			minelog << "  Header-hash:" << current.headerHash.hex();
			minelog << "  Seedhash:" << current.seedHash.hex();
			minelog << "  Target: " << h256(current.boundary).hex();
			f.setWork(current);

		}
	}


	void doFarm(MinerType _m, string & _remote, unsigned _recheckPeriod)
	{
		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor{&EthashGPUMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashGPUMiner(ci); }};
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{ &EthashCUDAMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		(void)_m;
		(void)_remote;
		(void)_recheckPeriod;
		jsonrpc::HttpClient client(m_farmURL);
		:: FarmClient rpc(client);
		jsonrpc::HttpClient failoverClient(m_farmFailOverURL);
		::FarmClient rpcFailover(failoverClient);

		FarmClient * prpc = &rpc;

		h256 id = h256::random();
		Farm f;
		f.setSealers(sealers);
		if (_m == MinerType::CL)
			f.start("opencl", false);
		else if (_m == MinerType::CUDA)
			f.start("cuda", false);
		WorkPackage current;
		std::mutex x_current;
		while (m_running)
			try
			{
				bool completed = false;
				Solution solution;
				f.onSolutionFound([&](Solution sol)
				{
					solution = sol;
					return completed = true;
				});
				for (unsigned i = 0; !completed; ++i)
				{
					auto mp = f.miningProgress();
					f.resetMiningProgress();
					if (current)
						minelog << "Mining on PoWhash" << "#" + (current.headerHash.hex().substr(0, 8)) << ": " << mp << f.getSolutionStats();
					else
						minelog << "Getting work package...";

					auto rate = mp.rate();

					try
					{
						prpc->eth_submitHashrate(toJS(rate), "0x" + id.hex());
					}
					catch (jsonrpc::JsonRpcException const& _e)
					{
						cwarn << "Failed to submit hashrate.";
						cwarn << boost::diagnostic_information(_e);
					}

					Json::Value v = prpc->eth_getWork();
					h256 hh(v[0].asString());
					h256 newSeedHash(v[1].asString());

					if (hh != current.headerHash)
					{
						x_current.lock();
						current.headerHash = hh;
						current.seedHash = newSeedHash;
						current.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);
						minelog << "Got work package: #" + current.headerHash.hex().substr(0,8);
						f.setWork(current);
						x_current.unlock();
					}
					this_thread::sleep_for(chrono::milliseconds(_recheckPeriod));
				}
				cnote << "Solution found; Submitting to" << _remote << "...";
				cnote << "  Nonce:" << solution.nonce.hex();
				cnote << "  headerHash:" << solution.headerHash.hex();
				cnote << "  mixHash:" << solution.mixHash.hex();
				if (EthashAux::eval(solution.seedHash, solution.headerHash, solution.nonce).value < solution.boundary)
				{
					bool ok = prpc->eth_submitWork("0x" + toString(solution.nonce), "0x" + toString(solution.headerHash), "0x" + toString(solution.mixHash));
					if (ok) {
						cnote << "B-) Submitted and accepted.";
						f.acceptedSolution(false);
					}
					else {
						cwarn << ":-( Not accepted.";
						f.rejectedSolution(false);
					}
					//exit(0);
				}
				else {
					f.failedSolution();
					cwarn << "FAILURE: GPU gave incorrect result!";
				}
				current.reset();
			}
			catch (jsonrpc::JsonRpcException&)
			{
				if (m_maxFarmRetries > 0)
				{
					for (auto i = 3; --i; this_thread::sleep_for(chrono::seconds(1)))
						cerr << "JSON-RPC problem. Probably couldn't connect. Retrying in " << i << "... \r";
					cerr << endl;
				}
				else
				{
					cerr << "JSON-RPC problem. Probably couldn't connect." << endl;
				}
				if (m_farmFailOverURL != "")
				{
					m_farmRetries++;
					if (m_farmRetries > m_maxFarmRetries)
					{
						if (_remote == "exit")
						{
							m_running = false;
						}
						else if (_remote == m_farmURL) {
							_remote = m_farmFailOverURL;
							prpc = &rpcFailover;
						}
						else {
							_remote = m_farmURL;
							prpc = &rpc;
						}
						m_farmRetries = 0;
					}

				}
			}
		exit(0);
	}

#if ETH_STRATUM
	void doStratum()
	{
		map<string, Farm::SealerDescriptor> sealers;
#if ETH_ETHASHCL
		sealers["opencl"] = Farm::SealerDescriptor{ &EthashGPUMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashGPUMiner(ci); } };
#endif
#if ETH_ETHASHCUDA
		sealers["cuda"] = Farm::SealerDescriptor{ &EthashCUDAMiner::instances, [](Miner::ConstructionInfo ci){ return new EthashCUDAMiner(ci); } };
#endif
		if (!m_farmRecheckSet)
			m_farmRecheckPeriod = m_defaultStratumFarmRecheckPeriod;

		Farm f;

		// this is very ugly, but if Stratum Client V2 tunrs out to be a success, V1 will be completely removed anyway
		if (m_stratumClientVersion == 1) {
			EthStratumClient client(&f, m_minerType, m_farmURL, m_port, m_user, m_pass, m_maxFarmRetries, m_worktimeout, m_stratumProtocol, m_email);
			if (m_farmFailOverURL != "")
			{
				if (m_fuser != "")
				{
					client.setFailover(m_farmFailOverURL, m_fport, m_fuser, m_fpass);
				}
				else
				{
					client.setFailover(m_farmFailOverURL, m_fport);
				}
			}
			f.setSealers(sealers);

			f.onSolutionFound([&](Solution sol)
			{
				if (client.isConnected()) {
					client.submit(sol);
				}
				else {
					cwarn << "Can't submit solution: Not connected";
				}
				return false;
			});

			while (client.isRunning())
			{
				auto mp = f.miningProgress();
				f.resetMiningProgress();
				if (client.isConnected())
				{
					if (client.current())
					{
						minelog << "Mining on PoWhash" << "#" + (client.currentHeaderHash().hex().substr(0, 8)) << ": " << mp << f.getSolutionStats();
					}
					else if (client.waitState() == MINER_WAIT_STATE_WORK)
					{
						minelog << "Waiting for work package...";
					}
				}
				this_thread::sleep_for(chrono::milliseconds(m_farmRecheckPeriod));
			}
		}
		else if (m_stratumClientVersion == 2) {
			EthStratumClientV2 client(&f, m_minerType, m_farmURL, m_port, m_user, m_pass, m_maxFarmRetries, m_worktimeout, m_stratumProtocol, m_email);
			if (m_farmFailOverURL != "")
			{
				if (m_fuser != "")
				{
					client.setFailover(m_farmFailOverURL, m_fport, m_fuser, m_fpass);
				}
				else
				{
					client.setFailover(m_farmFailOverURL, m_fport);
				}
			}
			f.setSealers(sealers);

			f.onSolutionFound([&](Solution sol)
			{
				client.submit(sol);
				return false;
			});

			while (client.isRunning())
			{
				auto mp = f.miningProgress();
				f.resetMiningProgress();
				if (client.isConnected())
				{
					if (client.current())
					{
						minelog << "Mining on PoWhash" << "#" + (client.currentHeaderHash().hex().substr(0, 8)) << ": " << mp << f.getSolutionStats();
					}
					else if (client.waitState() == MINER_WAIT_STATE_WORK)
					{
						minelog << "Waiting for work package...";
					}
				}
				this_thread::sleep_for(chrono::milliseconds(m_farmRecheckPeriod));
			}
		}

	}
#endif

	/// Operating mode.
	OperationMode mode;

	/// Mining options
	bool m_running = true;
	MinerType m_minerType = MinerType::Mixed;
	unsigned m_openclPlatform = 0;
	unsigned m_openclDevice = 0;
	unsigned m_miningThreads = UINT_MAX;
	bool m_shouldListDevices = false;
#if ETH_ETHASHCL
	unsigned m_openclDeviceCount = 0;
	unsigned m_openclDevices[16];
#if !ETH_ETHASHCUDA
	unsigned m_globalWorkSizeMultiplier = ethash_cl_miner::c_defaultGlobalWorkSizeMultiplier;
	unsigned m_localWorkSize = ethash_cl_miner::c_defaultLocalWorkSize;
#endif
#endif
#if ETH_ETHASHCUDA
	unsigned m_globalWorkSizeMultiplier = ethash_cuda_miner::c_defaultGridSize;
	unsigned m_localWorkSize = ethash_cuda_miner::c_defaultBlockSize;
	unsigned m_cudaDeviceCount = 0;
	unsigned m_cudaDevices[16];
	unsigned m_numStreams = ethash_cuda_miner::c_defaultNumStreams;
	unsigned m_cudaSchedule = 4; // sync
#endif
	// default value was 350MB of GPU memory for other stuff (windows system rendering, e.t.c.)
	unsigned m_extraGPUMemory = 0;// 350000000; don't assume miners run desktops...
	unsigned m_dagLoadMode = 0; // parallel
	unsigned m_dagCreateDevice = 0;
	/// Benchmarking params
	unsigned m_benchmarkWarmup = 15;
	unsigned m_parallelHash    = 4;
	unsigned m_benchmarkTrial = 3;
	unsigned m_benchmarkTrials = 5;
	unsigned m_benchmarkBlock = 0;
	/// Farm params
	string m_farmURL = "http://127.0.0.1:8545";
	string m_farmFailOverURL = "";


	string m_activeFarmURL = m_farmURL;
	unsigned m_farmRetries = 0;
	unsigned m_maxFarmRetries = 3;
	unsigned m_farmRecheckPeriod = 500;
	unsigned m_defaultStratumFarmRecheckPeriod = 2000;
	bool m_farmRecheckSet = false;
	int m_worktimeout = 180;

#if ETH_STRATUM
	int m_stratumClientVersion = 1;
	int m_stratumProtocol = STRATUM_PROTOCOL_STRATUM;
	string m_user;
	string m_pass;
	string m_port;
	string m_fuser = "";
	string m_fpass = "";
	string m_email = "";
#endif
	string m_fport = "";
};
