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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include <signal.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>

#include <libdevcrypto/FileSystem.h>
#include <libevmcore/Instruction.h>
#include <libdevcore/StructuredLogger.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/EthashAux.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <libethereum/All.h>
#include <libwebthree/WebThree.h>
#if ETH_JSONRPC || !ETH_TRUE
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#endif
#include "BuildInfo.h"
#if ETH_JSONRPC || !ETH_TRUE
#include "PhoneHome.h"
#include "Farm.h"
#endif
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
using namespace boost::algorithm;
using dev::eth::Instruction;

#undef RETURN

bool isTrue(std::string const& _m)
{
	return _m == "on" || _m == "yes" || _m == "true" || _m == "1";
}

bool isFalse(std::string const& _m)
{
	return _m == "off" || _m == "no" || _m == "false" || _m == "0";
}

void help()
{
	cout
		<< "Usage ethminer [OPTIONS]" << endl
		<< "Options:" << endl << endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "Work farming mode:" << endl
		<< "    -F,--farm <url>  Put into mining farm mode with the work server at URL (default: http://127.0.0.1:8080)" << endl
		<< "    --farm-recheck <n>  Leave n ms between checks for changed work (default: 500)." << endl
#endif
		<< "Benchmarking mode:" << endl
		<< "    -M,--benchmark  Benchmark for mining and exit; use with --cpu and --opencl." << endl
		<< "    --benchmark-warmup <seconds>  Set the duration of warmup for the benchmark tests (default: 3)." << endl
		<< "    --benchmark-trial <seconds>  Set the duration for each trial for the benchmark tests (default: 3)." << endl
		<< "    --benchmark-trials <n>  Set the duration of warmup for the benchmark tests (default: 5)." << endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "    --phone-home <on/off>  When benchmarking, publish results (default: on)" << endl
#endif
		<< "DAG creation mode:" << endl
		<< "    -D,--create-dag <number>  Create the DAG in preparation for mining on given block and exit." << endl
		<< "General Options:" << endl
		<< "    -C,--cpu  When mining, use the CPU." << endl
		<< "    -G,--opencl  When mining use the GPU via OpenCL." << endl
		<< "    --opencl-platform <n>  When mining using -G/--opencl use OpenCL platform n (default: 0)." << endl
		<< "    --opencl-device <n>  When mining using -G/--opencl use OpenCL device n (default: 0)." << endl
		<< "    -t, --mining-threads <n> Limit number of CPU/GPU miners to n (default: use everything available on selected platform)" << endl
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 8)." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
		;
		exit(0);
}

string credits()
{
	std::ostringstream cout;
	cout
		<< "Ethereum (++) " << dev::Version << endl
		<< "  Code by Gav Wood et al, (c) 2013, 2014, 2015." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;
	return cout.str();
}

void version()
{
	cout << "eth version " << dev::Version << endl;
	cout << "eth network protocol version: " << dev::eth::c_protocolVersion << endl;
	cout << "Client database version: " << dev::eth::c_databaseVersion << endl;
	cout << "Build: " << DEV_QUOTED(ETH_BUILD_PLATFORM) << "/" << DEV_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
}

void doInitDAG(unsigned _n)
{
	BlockInfo bi;
	bi.number = _n;
	cout << "Initializing DAG for epoch beginning #" << (bi.number / 30000 * 30000) << " (seedhash " << bi.seedHash().abridged() << "). This will take a while." << endl;
	Ethash::prep(bi);
	exit(0);
}

enum class OperationMode
{
	DAGInit,
	Benchmark,
	Farm
};

enum class MinerType
{
	CPU,
	GPU
};

void doBenchmark(MinerType _m, bool _phoneHome, unsigned _warmupDuration = 15, unsigned _trialDuration = 3, unsigned _trials = 5)
{
	BlockInfo genesis = CanonBlockChain::genesis();
	genesis.difficulty = 1 << 18;
	cdebug << genesis.boundary();

	GenericFarm<Ethash> f;
	f.onSolutionFound([&](ProofOfWork::Solution) { return false; });

	string platformInfo = _m == MinerType::CPU ? ProofOfWork::CPUMiner::platformInfo() : _m == MinerType::GPU ? ProofOfWork::GPUMiner::platformInfo() : "";
	cout << "Benchmarking on platform: " << platformInfo << endl;

	cout << "Preparing DAG..." << endl;
	Ethash::prep(genesis);

	genesis.difficulty = u256(1) << 63;
	genesis.noteDirty();
	f.setWork(genesis);
	if (_m == MinerType::CPU)
		f.startCPU();
	else if (_m == MinerType::GPU)
		f.startGPU();

	map<uint64_t, MiningProgress> results;
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
		if (i > 1 && i < 5)
			innerMean += rate;
	}
	f.stop();
	cout << "min/mean/max: " << results.begin()->second.rate() << "/" << (mean / _trials) << "/" << results.rbegin()->second.rate() << " H/s" << endl;
	cout << "inner mean: " << (innerMean / (_trials - 2)) << " H/s" << endl;

	(void)_phoneHome;
#if ETH_JSONRPC || !ETH_TRUE
	if (_phoneHome)
	{
		cout << "Phoning home to find world ranking..." << endl;
		jsonrpc::HttpClient client("http://gav.ethdev.com:3000/benchmark");
		PhoneHome rpc(client);
		try
		{
			unsigned ranking = rpc.report_benchmark(platformInfo, innerMean);
			cout << "Ranked: " << ranking << " of all benchmarks." << endl;
		}
		catch (...)
		{
			cout << "Error phoning home. ET is sad." << endl;
		}
	}
#endif
	exit(0);
}

struct HappyChannel: public LogChannel  { static const char* name() { return ":-D"; } static const int verbosity = 1; };
struct SadChannel: public LogChannel { static const char* name() { return ":-("; } static const int verbosity = 1; };

void doFarm(MinerType _m, string const& _remote, unsigned _recheckPeriod)
{
	(void)_m;
	(void)_remote;
	(void)_recheckPeriod;
#if ETH_JSONRPC || !ETH_TRUE
	jsonrpc::HttpClient client(_remote);
	Farm rpc(client);
	GenericFarm<Ethash> f;
	if (_m == MinerType::CPU)
		f.startCPU();
	else if (_m == MinerType::GPU)
		f.startGPU();

	ProofOfWork::WorkPackage current;
	while (true)
		try
		{
			bool completed = false;
			ProofOfWork::Solution solution;
			f.onSolutionFound([&](ProofOfWork::Solution sol)
			{
				solution = sol;
				return completed = true;
			});
			for (unsigned i = 0; !completed; ++i)
			{
				if (current)
					cnote << "Mining on PoWhash" << current.headerHash << ": " << f.miningProgress();
				else
					cnote << "Getting work package...";
				Json::Value v = rpc.eth_getWork();
				h256 hh(v[0].asString());
				if (hh != current.headerHash)
				{
					current.headerHash = hh;
					current.seedHash = h256(v[1].asString());
					current.boundary = h256(fromHex(v[2].asString()), h256::AlignRight);
					cnote << "Got work package:" << current.headerHash << " < " << current.boundary;
					f.setWork(current);
				}
				this_thread::sleep_for(chrono::milliseconds(_recheckPeriod));
			}
			cnote << "Solution found; submitting [" << solution.nonce << "," << current.headerHash << "," << solution.mixHash << "] to" << _remote << "...";
			bool ok = rpc.eth_submitWork("0x" + toString(solution.nonce), "0x" + toString(current.headerHash), "0x" + toString(solution.mixHash));
			if (ok)
				clog(HappyChannel) << "Submitted and accepted.";
			else
				clog(SadChannel) << "Not accepted.";
			current.reset();
		}
		catch (jsonrpc::JsonRpcException&)
		{
			for (auto i = 3; --i; this_thread::sleep_for(chrono::seconds(1)))
				cerr << "JSON-RPC problem. Probably couldn't connect. Retrying in " << i << "... \r";
			cerr << endl;
		}
#endif
	exit(0);
}

int main(int argc, char** argv)
{
	// Init defaults
	Defaults::get();

	/// Operating mode.
	OperationMode mode = OperationMode::Farm;

	/// Mining options
	MinerType minerType = MinerType::CPU;
	unsigned openclPlatform = 0;
	unsigned openclDevice = 0;
	unsigned miningThreads = UINT_MAX;

	/// DAG initialisation param.
	unsigned initDAG = 0;

	/// Benchmarking params
	bool phoneHome = true;
	unsigned benchmarkWarmup = 3;
	unsigned benchmarkTrial = 3;
	unsigned benchmarkTrials = 5;

	/// Farm params
	string farmURL = "http://127.0.0.1:8080";
	unsigned farmRecheckPeriod = 500;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if ((arg == "-F" || arg == "--farm") && i + 1 < argc)
		{
			mode = OperationMode::Farm;
			farmURL = argv[++i];
		}
		else if (arg == "--farm-recheck" && i + 1 < argc)
			try {
				farmRecheckPeriod = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--opencl-platform" && i + 1 < argc)
			try {
				openclPlatform = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--opencl-device" && i + 1 < argc)
			try {
				openclDevice = stol(argv[++i]);
				miningThreads = 1;
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--phone-home" && i + 1 < argc)
		{
			string m = argv[++i];
			if (isTrue(m))
				phoneHome = true;
			else if (isFalse(m))
				phoneHome = false;
			else
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "--benchmark-warmup" && i + 1 < argc)
			try {
				benchmarkWarmup = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--benchmark-trial" && i + 1 < argc)
			try {
				benchmarkTrial = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--benchmark-trials" && i + 1 < argc)
			try {
				benchmarkTrials = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "-C" || arg == "--cpu")
			minerType = MinerType::CPU;
		else if (arg == "-G" || arg == "--opencl")
			minerType = MinerType::GPU;
		else if ((arg == "-D" || arg == "--create-dag") && i + 1 < argc)
		{
			string m = boost::to_lower_copy(string(argv[++i]));
			mode = OperationMode::DAGInit;
			try
			{
				initDAG = stol(m);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				return -1;
			}
		}
		else if ((arg == "-w" || arg == "--check-pow") && i + 4 < argc)
		{
			string m;
			try
			{
				BlockInfo bi;
				m = boost::to_lower_copy(string(argv[++i]));
				h256 powHash(m);
				m = boost::to_lower_copy(string(argv[++i]));
				h256 seedHash;
				if (m.size() == 64 || m.size() == 66)
					seedHash = h256(m);
				else
					seedHash = EthashAux::seedHash(stol(m));
				m = boost::to_lower_copy(string(argv[++i]));
				bi.difficulty = u256(m);
				auto boundary = bi.boundary();
				m = boost::to_lower_copy(string(argv[++i]));
				bi.nonce = h64(m);
				auto r = EthashAux::eval(seedHash, powHash, bi.nonce);
				bool valid = r.value < boundary;
				cout << (valid ? "VALID :-)" : "INVALID :-(") << endl;
				cout << r.value << (valid ? " < " : " >= ") << boundary << endl;
				cout << "  where " << boundary << " = 2^256 / " << bi.difficulty << endl;
				cout << "  and " << r.value << " = ethash(" << powHash << ", " << bi.nonce << ")" << endl;
				cout << "  with seed as " << seedHash << endl;
				if (valid)
					cout << "(mixHash = " << r.mixHash << ")" << endl;
				cout << "SHA3( light(seed) ) = " << sha3(EthashAux::light(seedHash)->data()) << endl;
				exit(0);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "-M" || arg == "--benchmark")
			mode = OperationMode::Benchmark;
		else if ((arg == "-t" || arg == "--mining-threads") && i + 1 < argc)
		{
			try {
				miningThreads = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
		else if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc)
			g_logVerbosity = atoi(argv[++i]);
		else if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-V" || arg == "--version")
			version();
		else
		{
			cerr << "Invalid argument: " << arg << endl;
			exit(-1);
		}
	}

	if (minerType == MinerType::CPU)
		ProofOfWork::CPUMiner::setNumInstances(miningThreads);
	else if (minerType == MinerType::GPU)
	{
		ProofOfWork::GPUMiner::setDefaultPlatform(openclPlatform);
		ProofOfWork::GPUMiner::setDefaultDevice(openclDevice);
		ProofOfWork::GPUMiner::setNumInstances(miningThreads);
	}

	if (mode == OperationMode::DAGInit)
		doInitDAG(initDAG);

	if (mode == OperationMode::Benchmark)
		doBenchmark(minerType, phoneHome, benchmarkWarmup, benchmarkTrial, benchmarkTrials);

	if (mode == OperationMode::Farm)
		doFarm(minerType, farmURL, farmRecheckPeriod);

	return 0;
}

