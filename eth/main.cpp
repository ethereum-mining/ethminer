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
#if ETH_READLINE || !ETH_TRUE
#include <readline/readline.h>
#include <readline/history.h>
#endif
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

void interactiveHelp()
{
	cout
		<< "Commands:" << endl
		<< "    netstart <port>  Starts the network subsystem on a specific port." << endl
		<< "    netstop  Stops the network subsystem." << endl
		<< "    jsonstart <port>  Starts the JSON-RPC server." << endl
		<< "    jsonstop  Stops the JSON-RPC server." << endl
		<< "    connect <addr> <port>  Connects to a specific peer." << endl
		<< "    verbosity (<level>)  Gets or sets verbosity level." << endl
		<< "    setetherprice <p>  Resets the ether price." << endl
		<< "    setpriority <p>  Resets the transaction priority." << endl
		<< "    minestart  Starts mining." << endl
		<< "    minestop  Stops mining." << endl
		<< "    mineforce <enable>  Forces mining, even when there are no transactions." << endl
		<< "    address  Gives the current address." << endl
		<< "    secret  Gives the current secret" << endl
		<< "    block  Gives the current block height." << endl
		<< "    balance  Gives the current balance." << endl
		<< "    transact  Execute a given transaction." << endl
		<< "    send  Execute a given transaction with current secret." << endl
		<< "    contract  Create a new contract with current secret." << endl
		<< "    peers  List the peers that are connected" << endl
#if ETH_FATDB || !ETH_TRUE
		<< "    listaccounts  List the accounts on the network." << endl
		<< "    listcontracts  List the contracts on the network." << endl
#endif
		<< "    setsecret <secret>  Set the secret to the hex secret key." << endl
		<< "    setaddress <addr>  Set the coinbase (mining payout) address." << endl
		<< "    exportconfig <path>  Export the config (.RLP) to the path provided." << endl
		<< "    importconfig <path>  Import the config (.RLP) from the path provided." << endl
		<< "    inspect <contract>  Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    dumptrace <block> <index> <filename> <format>  Dumps a transaction trace" << endl << "to <filename>. <format> should be one of pretty, standard, standard+." << endl
		<< "    dumpreceipt <block> <index>  Dumps a transation receipt." << endl
		<< "    exit  Exits the application." << endl;
}

void help()
{
	cout
		<< "Usage eth [OPTIONS]" << endl
		<< "Options:" << endl << endl
		<< "Client mode (default):" << endl
		<< "    -o,--mode <full/peer>  Start a full node or a peer node (default: full)." << endl
		<< "    -i,--interactive  Enter interactive mode (default: non-interactive)." << endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "    -j,--json-rpc  Enable JSON-RPC server (default: off)." << endl
		<< "    --json-rpc-port <n>  Specify JSON-RPC server port (implies '-j', default: " << SensibleHttpPort << ")." << endl
#endif
		<< "    -K,--kill  First kill the blockchain." << endl
		<< "    -R,--rebuild  Rebuild the blockchain from the existing database." << endl
		<< "    -s,--secret <secretkeyhex>  Set the secret key for use with send command (default: auto)." << endl
		<< "    -S,--session-secret <secretkeyhex>  Set the secret key for use with send command, for this session only." << endl
		<< "Client transacting:" << endl
		<< "    -B,--block-fees <n>  Set the block fee profit in the reference unit e.g. ¢ (default: 15)." << endl
		<< "    -e,--ether-price <n>  Set the ether price in the reference unit e.g. ¢ (default: 30.679)." << endl
		<< "    -P,--priority <0 - 100>  Default % priority of a transaction (default: 50)." << endl
		<< "Client mining:" << endl
		<< "    -a,--address <addr>  Set the coinbase (mining payout) address to addr (default: auto)." << endl
		<< "    -m,--mining <on/off/number>  Enable mining, optionally for a specified number of blocks (default: off)" << endl
		<< "    -f,--force-mining  Mine even when there are no transactions to mine (default: off)" << endl
		<< "    -C,--cpu  When mining, use the CPU." << endl
		<< "    -G,--opencl  When mining use the GPU via OpenCL." << endl
		<< "    --opencl-platform <n>  When mining using -G/--opencl use OpenCL platform n (default: 0)." << endl
		<< "    --opencl-device <n>  When mining using -G/--opencl use OpenCL device n (default: 0)." << endl
		<< "    -t, --mining-threads <n> Limit number of CPU/GPU miners to n (default: use everything available on selected platform)" << endl
		<< "Client networking:" << endl
		<< "    --client-name <name>  Add a name to your client's version string (default: blank)." << endl
		<< "    -b,--bootstrap  Connect to the default Ethereum peerserver." << endl
		<< "    -x,--peers <number>  Attempt to connect to given number of peers (default: 5)." << endl
		<< "    --public-ip <ip>  Force public ip to given (default: auto)." << endl
		<< "    --listen-ip <ip>(:<port>)  Listen on the given IP for incoming connections (default: 0.0.0.0)." << endl
		<< "    --listen <port>  Listen on the given port for incoming connections (default: 30303)." << endl
		<< "    -r,--remote <host>(:<port>)  Connect to remote host (default: none)." << endl
		<< "    --port <port>  Connect to remote port (default: 30303)." << endl
		<< "    --network-id <n> Only connect to other hosts with this network id (default:0)." << endl
		<< "    --upnp <on/off>  Use UPnP for NAT (default: on)." << endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "Work farming mode:" << endl
		<< "    -F,--farm <url>  Put into mining farm mode with the work server at URL. Use with -G/--opencl." << endl
		<< "    --farm-recheck <n>  Leave n ms between checks for changed work (default: 500)." << endl
#endif
		<< "Ethash verify mode:" << endl
		<< "    -w,--check-pow <headerHash> <seedHash> <difficulty> <nonce>  Check PoW credentials for validity." << endl
		<< "Benchmarking mode:" << endl
		<< "    -M,--benchmark  Benchmark for mining and exit; use with --cpu and --opencl." << endl
		<< "    --benchmark-warmup <seconds>  Set the duration of warmup for the benchmark tests (default: 3)." << endl
		<< "    --benchmark-trial <seconds>  Set the duration for each trial for the benchmark tests (default: 3)." << endl
		<< "    --benchmark-trials <n>  Set the duration of warmup for the benchmark tests (default: 5)." << endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "    --phone-home <on/off>  When benchmarking, publish results (default: on)" << endl
#endif
		<< "DAG creation mode:" << endl
		<< "    -D,--create-dag <this/next/number>  Create the DAG in preparation for mining on given block and exit." << endl
		<< "Import/export modes:" << endl
		<< "    -I,--import <file>  Import file as a concatenated series of blocks and exit." << endl
		<< "    -E,--export <file>  Export file as a concatenated series of blocks and exit." << endl
		<< "    --from <n>  Export only from block n; n may be a decimal, a '0x' prefixed hash, or 'latest'." << endl
		<< "    --to <n>  Export only to block n (inclusive); n may be a decimal, a '0x' prefixed hash, or 'latest'." << endl
		<< "    --only <n>  Equivalent to --export-from n --export-to n." << endl
		<< "General Options:" << endl
		<< "    -d,--db-path <path>  Load database from path (default: " << getDataDir() << ")" << endl
#if ETH_EVMJIT || !ETH_TRUE
		<< "    -J,--jit  Enable EVM JIT (default: off)." << endl
#endif
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 8)." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
		;
		exit(0);
}

string credits(bool _interactive = false)
{
	std::ostringstream cout;
	cout
		<< "Ethereum (++) " << dev::Version << endl
		<< "  Code by Gav Wood et al, (c) 2013, 2014, 2015." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;

	if (_interactive)
		cout
			<< "Type 'netstart 30303' to start networking" << endl
			<< "Type 'connect " << Host::pocHost() << " 30303' to connect" << endl
			<< "Type 'exit' to quit" << endl << endl;
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

Address c_config = Address("ccdeac59d35627b7de09332e819d5159e7bb7250");
string pretty(h160 _a, dev::eth::State const& _st)
{
	string ns;
	h256 n;
	if (h160 nameReg = (u160)_st.storage(c_config, 0))
		n = _st.storage(nameReg, (u160)(_a));
	if (n)
	{
		std::string s((char const*)n.data(), 32);
		if (s.find_first_of('\0') != string::npos)
			s.resize(s.find_first_of('\0'));
		ns = " " + s;
	}
	return ns;
}

bool g_exit = false;

void sighandler(int)
{
	g_exit = true;
}

enum class NodeMode
{
	PeerServer,
	Full
};

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
	Node,
	Import,
	Export,
	DAGInit,
	Benchmark,
	Farm
};

enum class Format
{
	Binary,
	Hex,
	Human
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
	cout << "\x1b[30mEthBlack\x1b[0m" << endl;
	cout << "\x1b[90mEthCoal\x1b[0m" << endl;
	cout << "\x1b[37mEthGray\x1b[0m" << endl;
	cout << "\x1b[97mEthWhite\x1b[0m" << endl;
	cout << "\x1b[31mEthRed\x1b[0m" << endl;
	cout << "\x1b[32mEthGreen\x1b[0m" << endl;
	cout << "\x1b[33mEthYellow\x1b[0m" << endl;
	cout << "\x1b[34mEthBlue\x1b[0m" << endl;
	cout << "\x1b[35mEthPurple\x1b[0m" << endl;
	cout << "\x1b[36mEthCyan\x1b[0m" << endl;
	// High Intensity
	cout << "\x1b[91mEthRedI\x1b[0m" << endl;
	cout << "\x1b[92mEthLime\x1b[0m" << endl;
	cout << "\x1b[93mEthYellowI\x1b[0m" << endl;
	cout << "\x1b[94mEthBlueI\x1b[0m" << endl;
	cout << "\x1b[95mEthPurpleI\x1b[0m" << endl;
	cout << "\x1b[96mEthCyanI\x1b[0m" << endl;

	// Bold
	cout << "\x1b[1;30mEthBlackB\x1b[0m" << endl;
	cout << "\x1b[1;90mEthCoalB\x1b[0m" << endl;
	cout << "\x1b[1;37mEthGrayB\x1b[0m" << endl;
	cout << "\x1b[1;97mEthWhiteB\x1b[0m" << endl;
	cout << "\x1b[1;31mEthRedB\x1b[0m" << endl;
	cout << "\x1b[1;32mEthGreenB\x1b[0m" << endl;
	cout << "\x1b[1;33mEthYellowB\x1b[0m" << endl;
	cout << "\x1b[1;34mEthBlueB\x1b[0m" << endl;
	cout << "\x1b[1;35mEthPurpleB\x1b[0m" << endl;
	cout << "\x1b[1;36mEthCyanB\x1b[0m" << endl;
	// Bold High Intensity
	cout << "\x1b[1;91mEthRedBI\x1b[0m" << endl;
	cout << "\x1b[1;92mEthGreenBI\x1b[0m" << endl;
	cout << "\x1b[1;93mEthYellowBI\x1b[0m" << endl;
	cout << "\x1b[1;94mEthBlueBI\x1b[0m" << endl;
	cout << "\x1b[1;95mEthPurpleBI\x1b[0m" << endl;
	cout << "\x1b[1;96mEthCyanBI\x1b[0m" << endl;

	// Background
	cout << "\x1b[40mEthBlackOn\x1b[0m" << endl;
	cout << "\x1b[100mEthCoalOn\x1b[0m" << endl;
	cout << "\x1b[47mEthGrayOn\x1b[0m" << endl;
	cout << "\x1b[107mEthWhiteOn\x1b[0m" << endl;
	cout << "\x1b[41mEthRedOn\x1b[0m" << endl;
	cout << "\x1b[42mEthGreenOn\x1b[0m" << endl;
	cout << "\x1b[43mEthYellowOn\x1b[0m" << endl;
	cout << "\x1b[44mEthBlueOn\x1b[0m" << endl;
	cout << "\x1b[45mEthPurpleOn\x1b[0m" << endl;
	cout << "\x1b[46mEthCyanOn\x1b[0m" << endl;
	// High Intensity backgrounds
	cout << "\x1b[101mEthRedOnI\x1b[0m" << endl;
	cout << "\x1b[102mEthGreenOnI\x1b[0m" << endl;
	cout << "\x1b[103mEthYellowOnI\x1b[0m" << endl;
	cout << "\x1b[104mEthBlueOnI\x1b[0m" << endl;
	cout << "\x1b[105mEthPurpleOnI\x1b[0m" << endl;
	cout << "\x1b[106mEthCyanOnI\x1b[0m" << endl;

	// Underline
	cout << "\x1b[4;30mEthBlackU\x1b[0m" << endl;
	cout << "\x1b[4;31mEthRedU\x1b[0m" << endl;
	cout << "\x1b[4;32mEthGreenU\x1b[0m" << endl;
	cout << "\x1b[4;33mEthYellowU\x1b[0m" << endl;
	cout << "\x1b[4;34mEthBlueU\x1b[0m" << endl;
	cout << "\x1b[4;35mEthPurpleU\x1b[0m" << endl;
	cout << "\x1b[4;36mEthCyanU\x1b[0m" << endl;
	cout << "\x1b[4;37mEthWhiteU\x1b[0m" << endl;

	// Init defaults
	Defaults::get();

	/// Operating mode.
	OperationMode mode = OperationMode::Node;
	string dbPath;

	/// Mining options
	MinerType minerType = MinerType::CPU;
	unsigned openclPlatform = 0;
	unsigned openclDevice = 0;
	unsigned miningThreads = UINT_MAX;

	/// File name for import/export.
	string filename;

	/// Hashes/numbers for export range.
	string exportFrom = "1";
	string exportTo = "latest";
	Format exportFormat = Format::Binary;

	/// DAG initialisation param.
	unsigned initDAG = 0;

	/// General params for Node operation
	NodeMode nodeMode = NodeMode::Full;
	bool interactive = false;
#if ETH_JSONRPC
	int jsonrpc = -1;
#endif
	bool upnp = true;
	WithExisting killChain = WithExisting::Trust;
	bool jit = false;

	/// Networking params.
	string clientName;
	string listenIP;
	unsigned short listenPort = 30303;
	string publicIP;
	string remoteHost;
	unsigned short remotePort = 30303;
	unsigned peers = 5;
	bool bootstrap = false;
	unsigned networkId = 0;

	/// Mining params
	unsigned mining = 0;
	bool forceMining = false;
	KeyPair sigKey = KeyPair::create();
	Secret sessionSecret;
	Address coinbase = sigKey.address();

	/// Structured logging params
	bool structuredLogging = false;
	string structuredLoggingFormat = "%Y-%m-%dT%H:%M:%S";

	/// Transaction params
	TransactionPriority priority = TransactionPriority::Medium;
	double etherPrice = 30.679;
	double blockFees = 15.0;

	/// Benchmarking params
	bool phoneHome = true;
	unsigned benchmarkWarmup = 3;
	unsigned benchmarkTrial = 3;
	unsigned benchmarkTrials = 5;

	/// Farm params
	string farmURL = "http://127.0.0.1:8080";
	unsigned farmRecheckPeriod = 500;

	string configFile = getDataDir() + "/config.rlp";
	bytes b = contents(configFile);
	if (b.size())
	{
		RLP config(b);
		sigKey = KeyPair(config[0].toHash<Secret>());
		coinbase = config[1].toHash<Address>();
	}

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "--listen-ip" && i + 1 < argc)
			listenIP = argv[++i];
		else if ((arg == "-l" || arg == "--listen" || arg == "--listen-port") && i + 1 < argc)
		{
			if (arg == "-l")
				cerr << "-l is DEPRECATED. It will be removed for the Frontier. Use --listen-port instead." << endl;
			listenPort = (short)atoi(argv[++i]);
		}
		else if ((arg == "-u" || arg == "--public-ip" || arg == "--public") && i + 1 < argc)
		{
			if (arg == "-u")
				cerr << "-u is DEPRECATED. It will be removed for the Frontier. Use --public-ip instead." << endl;
			publicIP = argv[++i];
		}
		else if ((arg == "-r" || arg == "--remote") && i + 1 < argc)
			remoteHost = argv[++i];
		else if ((arg == "-p" || arg == "--port") && i + 1 < argc)
		{
			if (arg == "-p")
				cerr << "-p is DEPRECATED. It will be removed for the Frontier. Use --port instead (or place directly as host:port)." << endl;
			remotePort = (short)atoi(argv[++i]);
		}
		else if ((arg == "-I" || arg == "--import") && i + 1 < argc)
		{
			mode = OperationMode::Import;
			filename = argv[++i];
		}
		else if ((arg == "-E" || arg == "--export") && i + 1 < argc)
		{
			mode = OperationMode::Export;
			filename = argv[++i];
		}
		else if ((arg == "-F" || arg == "--farm") && i + 1 < argc)
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
		else if (arg == "--format" && i + 1 < argc)
		{
			string m = argv[++i];
			if (m == "binary")
				exportFormat = Format::Binary;
			else if (m == "hex")
				exportFormat = Format::Hex;
			else if (m == "human")
				exportFormat = Format::Human;
			else
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "--to" && i + 1 < argc)
			exportTo = argv[++i];
		else if (arg == "--from" && i + 1 < argc)
			exportFrom = argv[++i];
		else if (arg == "--only" && i + 1 < argc)
			exportTo = exportFrom = argv[++i];
		else if ((arg == "-n" || arg == "-u" || arg == "--upnp") && i + 1 < argc)
		{
			if (arg == "-n")
				cerr << "-n is DEPRECATED. It will be removed for the Frontier. Use --upnp instead." << endl;
			string m = argv[++i];
			if (isTrue(m))
				upnp = true;
			else if (isFalse(m))
				upnp = false;
			else
			{
				cerr << "Bad " << arg << " option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "--network-id" && i + 1 < argc)
			try {
				networkId = stol(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
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
		else if (arg == "-K" || arg == "--kill-blockchain" || arg == "--kill")
			killChain = WithExisting::Kill;
		else if (arg == "-R" || arg == "--rebuild")
			killChain = WithExisting::Verify;
		else if ((arg == "-c" || arg == "--client-name") && i + 1 < argc)
		{
			if (arg == "-c")
				cerr << "-c is DEPRECATED. It will be removed for the Frontier. Use --client-name instead." << endl;
			clientName = argv[++i];
		}
		else if ((arg == "-a" || arg == "--address" || arg == "--coinbase-address") && i + 1 < argc)
			try {
				coinbase = h160(fromHex(argv[++i], WhenError::Throw));
			}
			catch (BadHexCharacter&)
			{
				cerr << "Bad hex in " << arg << " option: " << argv[i] << endl;
				return -1;
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
		else if ((arg == "-s" || arg == "--secret") && i + 1 < argc)
			sigKey = KeyPair(h256(fromHex(argv[++i])));
		else if ((arg == "-S" || arg == "--session-secret") && i + 1 < argc)
			sessionSecret = h256(fromHex(argv[++i]));
		else if (arg == "--structured-logging-format" && i + 1 < argc)
			structuredLoggingFormat = string(argv[++i]);
		else if (arg == "--structured-logging")
			structuredLogging = true;
		else if ((arg == "-d" || arg == "--path" || arg == "--db-path") && i + 1 < argc)
			dbPath = argv[++i];
		else if ((arg == "-D" || arg == "--create-dag") && i + 1 < argc)
		{
			string m = boost::to_lower_copy(string(argv[++i]));
			mode = OperationMode::DAGInit;
			if (m == "next")
				initDAG = PendingBlock;
			else if (m == "this")
				initDAG = LatestBlock;
			else
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
		else if ((arg == "-B" || arg == "--block-fees") && i + 1 < argc)
		{
			try
			{
				blockFees = stof(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
		else if ((arg == "-e" || arg == "--ether-price") && i + 1 < argc)
		{
			try
			{
				etherPrice = stof(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
		else if ((arg == "-P" || arg == "--priority") && i + 1 < argc)
		{
			string m = boost::to_lower_copy(string(argv[++i]));
			if (m == "lowest")
				priority = TransactionPriority::Lowest;
			else if (m == "low")
				priority = TransactionPriority::Low;
			else if (m == "medium" || m == "mid" || m == "default" || m == "normal")
				priority = TransactionPriority::Medium;
			else if (m == "high")
				priority = TransactionPriority::High;
			else if (m == "highest")
				priority = TransactionPriority::Highest;
			else
				try {
					priority = (TransactionPriority)(max(0, min(100, stoi(m))) * 8 / 100);
				}
				catch (...) {
					cerr << "Unknown " << arg << " option: " << m << endl;
					return -1;
				}
		}
		else if ((arg == "-m" || arg == "--mining") && i + 1 < argc)
		{
			string m = argv[++i];
			if (isTrue(m))
				mining = ~(unsigned)0;
			else if (isFalse(m))
				mining = 0;
			else
				try {
					mining = stoi(m);
				}
				catch (...) {
					cerr << "Unknown " << arg << " option: " << m << endl;
					return -1;
				}
		}
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
		else if (arg == "-b" || arg == "--bootstrap")
			bootstrap = true;
		else if (arg == "-f" || arg == "--force-mining")
			forceMining = true;
		else if (arg == "-i" || arg == "--interactive")
			interactive = true;
#if ETH_JSONRPC
		else if ((arg == "-j" || arg == "--json-rpc"))
			jsonrpc = jsonrpc == -1 ? SensibleHttpPort : jsonrpc;
		else if (arg == "--json-rpc-port" && i + 1 < argc)
			jsonrpc = atoi(argv[++i]);
#endif
		else if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc)
			g_logVerbosity = atoi(argv[++i]);
		else if ((arg == "-x" || arg == "--peers") && i + 1 < argc)
			peers = atoi(argv[++i]);
		else if ((arg == "-o" || arg == "--mode") && i + 1 < argc)
		{
			string m = argv[++i];
			if (m == "full")
				nodeMode = NodeMode::Full;
			else if (m == "peer")
				nodeMode = NodeMode::PeerServer;
			else
			{
				cerr << "Unknown mode: " << m << endl;
				return -1;
			}
		}
#if ETH_EVMJIT
		else if (arg == "-J" || arg == "--jit")
		{
			jit = true;
		}
#endif
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

	{
		RLPStream config(2);
		config << sigKey.secret() << coinbase;
		writeFile(configFile, config.out());
	}

	if (sessionSecret)
		sigKey = KeyPair(sessionSecret);

	

	if (minerType == MinerType::CPU)
		ProofOfWork::CPUMiner::setNumInstances(miningThreads);
	else if (minerType == MinerType::GPU)
	{
		ProofOfWork::GPUMiner::setDefaultPlatform(openclPlatform);
		ProofOfWork::GPUMiner::setDefaultDevice(openclDevice);
		ProofOfWork::GPUMiner::setNumInstances(miningThreads);
	}

	// Two codepaths is necessary since named block require database, but numbered
	// blocks are superuseful to have when database is already open in another process.
	if (mode == OperationMode::DAGInit && !(initDAG == LatestBlock || initDAG == PendingBlock))
		doInitDAG(initDAG);

	if (mode == OperationMode::Benchmark)
		doBenchmark(minerType, phoneHome, benchmarkWarmup, benchmarkTrial, benchmarkTrials);

	if (mode == OperationMode::Farm)
		doFarm(minerType, farmURL, farmRecheckPeriod);

	if (!clientName.empty())
		clientName += "/";

	StructuredLogger::get().initialize(structuredLogging, structuredLoggingFormat);
	VMFactory::setKind(jit ? VMKind::JIT : VMKind::Interpreter);
	auto netPrefs = publicIP.empty() ? NetworkPreferences(listenIP ,listenPort, upnp) : NetworkPreferences(publicIP, listenIP ,listenPort, upnp);
	auto nodesState = contents((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp");
	std::string clientImplString = "++eth/" + clientName + "v" + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) + (jit ? "/JIT" : "");
	dev::WebThreeDirect web3(
		clientImplString,
		dbPath,
		killChain,
		nodeMode == NodeMode::Full ? set<string>{"eth", "shh"} : set<string>(),
		netPrefs,
		&nodesState);
	
	if (mode == OperationMode::DAGInit)
		doInitDAG(web3.ethereum()->blockChain().number() + (initDAG == PendingBlock ? 30000 : 0));

	auto toNumber = [&](string const& s) -> unsigned {
		if (s == "latest")
			return web3.ethereum()->number();
		if (s.size() == 64 || (s.size() == 66 && s.substr(0, 2) == "0x"))
			return web3.ethereum()->blockChain().number(h256(s));
		try {
			return stol(s);
		}
		catch (...)
		{
			cerr << "Bad block number/hash option: " << s << endl;
			exit(-1);
		}
	};

	if (mode == OperationMode::Export)
	{
		ofstream fout(filename, std::ofstream::binary);
		ostream& out = (filename.empty() || filename == "--") ? cout : fout;

		unsigned last = toNumber(exportTo);
		for (unsigned i = toNumber(exportFrom); i <= last; ++i)
		{
			bytes block = web3.ethereum()->blockChain().block(web3.ethereum()->blockChain().numberHash(i));
			switch (exportFormat)
			{
			case Format::Binary: out.write((char const*)block.data(), block.size()); break;
			case Format::Hex: out << toHex(block) << endl; break;
			case Format::Human: out << RLP(block) << endl; break;
			default:;
			}
		}
		return 0;
	}

	if (mode == OperationMode::Import)
	{
		ifstream fin(filename, std::ifstream::binary);
		istream& in = (filename.empty() || filename == "--") ? cin : fin;
		unsigned alreadyHave = 0;
		unsigned good = 0;
		unsigned futureTime = 0;
		unsigned unknownParent = 0;
		unsigned bad = 0;
		while (in.peek() != -1)
		{
			bytes block(8);
			in.read((char*)block.data(), 8);
			block.resize(RLP(block, RLP::LaisezFaire).actualSize());
			in.read((char*)block.data() + 8, block.size() - 8);
			switch (web3.ethereum()->injectBlock(block))
			{
			case ImportResult::Success: good++; break;
			case ImportResult::AlreadyKnown: alreadyHave++; break;
			case ImportResult::UnknownParent: unknownParent++; break;
			case ImportResult::FutureTime: futureTime++; break;
			default: bad++; break;
			}
		}
		cout << (good + bad + futureTime + unknownParent + alreadyHave) << " total: " << good << " ok, " << alreadyHave << " got, " << futureTime << " future, " << unknownParent << " unknown parent, " << bad << " malformed." << endl;
		return 0;
	}

	cout << credits();
	web3.setIdealPeerCount(peers);
	std::shared_ptr<eth::BasicGasPricer> gasPricer = make_shared<eth::BasicGasPricer>(u256(double(ether / 1000) / etherPrice), u256(blockFees * 1000));
	eth::Client* c = nodeMode == NodeMode::Full ? web3.ethereum() : nullptr;
	StructuredLogger::starting(clientImplString, dev::Version);
	if (c)
	{
		c->setGasPricer(gasPricer);
		c->setForceMining(forceMining);
		c->setTurboMining(minerType == MinerType::GPU);
		c->setAddress(coinbase);
		c->setNetworkId(networkId);
	}

	cout << "Transaction Signer: " << sigKey.address() << endl;
	cout << "Mining Benefactor: " << coinbase << endl;
	web3.startNetwork();

	if (bootstrap)
		web3.addNode(p2p::NodeId(), Host::pocHost());
	if (remoteHost.size())
		web3.addNode(p2p::NodeId(), remoteHost + ":" + toString(remotePort));

#if ETH_JSONRPC
	shared_ptr<WebThreeStubServer> jsonrpcServer;
	unique_ptr<jsonrpc::AbstractServerConnector> jsonrpcConnector;
	if (jsonrpc > -1)
	{
		jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonrpc, "", "", SensibleHttpThreads));
		jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, vector<KeyPair>({sigKey})));
		jsonrpcServer->StartListening();
	}
#endif

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	if (interactive)
	{
		string logbuf;
		string l;
		while (!g_exit)
		{
			g_logPost = [](std::string const& a, char const*) { cout << "\r           \r" << a << endl << "Press Enter" << flush; };
			cout << logbuf << "Press Enter" << flush;
			std::getline(cin, l);
			logbuf.clear();
			g_logPost = [&](std::string const& a, char const*) { logbuf += a + "\n"; };

#if ETH_READLINE
			if (l.size())
				add_history(l.c_str());
			if (auto c = readline("> "))
			{
				l = c;
				free(c);
			}
			else
				break;
#else
			string l;
			cout << "> " << flush;
			std::getline(cin, l);
#endif
			istringstream iss(l);
			string cmd;
			iss >> cmd;
			boost::to_lower(cmd);
			if (cmd == "netstart")
			{
				iss >> netPrefs.listenPort;
				web3.setNetworkPreferences(netPrefs);
				web3.startNetwork();
			}
			else if (cmd == "connect")
			{
				string addrPort;
				iss >> addrPort;
				web3.addNode(p2p::NodeId(), addrPort);
			}
			else if (cmd == "netstop")
			{
				web3.stopNetwork();
			}
			else if (c && cmd == "minestart")
			{
				c->startMining();
			}
			else if (c && cmd == "minestop")
			{
				c->stopMining();
			}
			else if (c && cmd == "mineforce")
			{
				string enable;
				iss >> enable;
				c->setForceMining(isTrue(enable));
			}
			else if (c && cmd == "setblockfees")
			{
				iss >> blockFees;
				try
				{
					gasPricer->setRefBlockFees(u256(blockFees * 1000));
				}
				catch (Overflow const& _e)
				{
					cout << boost::diagnostic_information(_e);
				}

				cout << "Block fees: " << blockFees << endl;
			}
			else if (c && cmd == "setetherprice")
			{
				iss >> etherPrice;
				if (etherPrice == 0)
					cout << "ether price cannot be set to zero" << endl;
				else
				{
					try
					{
						gasPricer->setRefPrice(u256(double(ether / 1000) / etherPrice));
					}
					catch (Overflow const& _e)
					{
						cout << boost::diagnostic_information(_e);
					}
				}
				cout << "ether Price: " << etherPrice << endl;
			}
			else if (c && cmd == "setpriority")
			{
				string m;
				iss >> m;
				boost::to_lower(m);
				if (m == "lowest")
					priority = TransactionPriority::Lowest;
				else if (m == "low")
					priority = TransactionPriority::Low;
				else if (m == "medium" || m == "mid" || m == "default" || m == "normal")
					priority = TransactionPriority::Medium;
				else if (m == "high")
					priority = TransactionPriority::High;
				else if (m == "highest")
					priority = TransactionPriority::Highest;
				else
					try {
						priority = (TransactionPriority)(max(0, min(100, stoi(m))) * 8 / 100);
					}
					catch (...) {
						cerr << "Unknown priority: " << m << endl;
					}
				cout << "Priority: " << (int)priority << "/8" << endl;
			}
			else if (cmd == "verbosity")
			{
				if (iss.peek() != -1)
					iss >> g_logVerbosity;
				cout << "Verbosity: " << g_logVerbosity << endl;
			}
#if ETH_JSONRPC
			else if (cmd == "jsonport")
			{
				if (iss.peek() != -1)
					iss >> jsonrpc;
				cout << "JSONRPC Port: " << jsonrpc << endl;
			}
			else if (cmd == "jsonstart")
			{
				if (jsonrpc < 0)
					jsonrpc = SensibleHttpPort;
				jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonrpc, "", "", SensibleHttpThreads));
				jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, vector<KeyPair>({sigKey})));
				jsonrpcServer->StartListening();
			}
			else if (cmd == "jsonstop")
			{
				if (jsonrpcServer.get())
					jsonrpcServer->StopListening();
				jsonrpcServer.reset();
			}
#endif
			else if (cmd == "address")
			{
				cout << "Current address:" << endl << sigKey.address() << endl;
			}
			else if (cmd == "secret")
			{
				cout << "Secret Key: " << sigKey.secret() << endl;
			}
			else if (c && cmd == "block")
			{
				cout << "Current block: " <<c->blockChain().details().number << endl;
			}
			else if (cmd == "peers")
			{
				for (auto it: web3.peers())
					cout << it.host << ":" << it.port << ", " << it.clientVersion << ", "
						<< std::chrono::duration_cast<std::chrono::milliseconds>(it.lastPing).count() << "ms"
						<< endl;
			}
			else if (c && cmd == "balance")
			{
				cout << "Current balance: " << formatBalance( c->balanceAt(sigKey.address())) << " = " <<c->balanceAt(sigKey.address()) << " wei" << endl;
			}
			else if (c && cmd == "transact")
			{
				auto const& bc =c->blockChain();
				auto h = bc.currentHash();
				auto blockData = bc.block(h);
				BlockInfo info(blockData);
				if (iss.peek() != -1)
				{
					string hexAddr;
					u256 amount;
					u256 gasPrice;
					u256 gas;
					string sechex;
					string sdata;

					iss >> hexAddr >> amount >> gasPrice >> gas >> sechex >> sdata;

					if (!gasPrice)
						gasPrice = gasPricer->bid(priority);

					cnote << "Data:";
					cnote << sdata;
					bytes data = dev::eth::parseData(sdata);
					cnote << "Bytes:";
					string sbd = asString(data);
					bytes bbd = asBytes(sbd);
					stringstream ssbd;
					ssbd << bbd;
					cnote << ssbd.str();
					int ssize = sechex.length();
					int size = hexAddr.length();
					u256 minGas = (u256)Transaction::gasRequired(data, 0);
					if (size < 40)
					{
						if (size > 0)
							cwarn << "Invalid address length:" << size;
					}
					else if (gas < minGas)
						cwarn << "Minimum gas amount is" << minGas;
					else if (ssize < 40)
					{
						if (ssize > 0)
							cwarn << "Invalid secret length:" << ssize;
					}
					else
					{
						try
						{
							Secret secret = h256(fromHex(sechex));
							Address dest = h160(fromHex(hexAddr));
							c->submitTransaction(secret, amount, dest, data, gas, gasPrice);
						}
						catch (BadHexCharacter& _e)
						{
							cwarn << "invalid hex character, transaction rejected";
							cwarn << boost::diagnostic_information(_e);
						}
						catch (...)
						{
							cwarn << "transaction rejected";
						}
					}
				}
				else
					cwarn << "Require parameters: submitTransaction ADDRESS AMOUNT GASPRICE GAS SECRET DATA";
			}
#if ETH_FATDB
			else if (c && cmd == "listcontracts")
			{
				auto acs =c->addresses();
				string ss;
				for (auto const& i: acs)
					if ( c->codeAt(i, PendingBlock).size())
					{
						ss = toString(i) + " : " + toString( c->balanceAt(i)) + " [" + toString((unsigned) c->countAt(i)) + "]";
						cout << ss << endl;
					}
			}
			else if (c && cmd == "listaccounts")
			{
				auto acs =c->addresses();
				string ss;
				for (auto const& i: acs)
					if ( c->codeAt(i, PendingBlock).empty())
					{
						ss = toString(i) + " : " + toString( c->balanceAt(i)) + " [" + toString((unsigned) c->countAt(i)) + "]";
						cout << ss << endl;
					}
			}
#endif
			else if (c && cmd == "send")
			{
				if (iss.peek() != -1)
				{
					string hexAddr;
					u256 amount;

					iss >> hexAddr >> amount;
					int size = hexAddr.length();
					if (size < 40)
					{
						if (size > 0)
							cwarn << "Invalid address length:" << size;
					}
					else
					{
						auto const& bc =c->blockChain();
						auto h = bc.currentHash();
						auto blockData = bc.block(h);
						BlockInfo info(blockData);
						u256 minGas = (u256)Transaction::gasRequired(bytes(), 0);
						try
						{
							Address dest = h160(fromHex(hexAddr, WhenError::Throw));
							c->submitTransaction(sigKey.secret(), amount, dest, bytes(), minGas);
						}
						catch (BadHexCharacter& _e)
						{
							cwarn << "invalid hex character, transaction rejected";
							cwarn << boost::diagnostic_information(_e);
						}
						catch (...)
						{
							cwarn << "transaction rejected";
						}
					}
				}
				else
					cwarn << "Require parameters: send ADDRESS AMOUNT";
			}
			else if (c && cmd == "contract")
			{
				auto const& bc =c->blockChain();
				auto h = bc.currentHash();
				auto blockData = bc.block(h);
				BlockInfo info(blockData);
				if (iss.peek() != -1)
				{
					u256 endowment;
					u256 gas;
					u256 gasPrice;
					string sinit;
					iss >> endowment >> gasPrice >> gas >> sinit;
					trim_all(sinit);
					int size = sinit.length();
					bytes init;
					cnote << "Init:";
					cnote << sinit;
					cnote << "Code size:" << size;
					if (size < 1)
						cwarn << "No code submitted";
					else
					{
						cnote << "Assembled:";
						stringstream ssc;
						try
						{
							init = fromHex(sinit, WhenError::Throw);
						}
						catch (BadHexCharacter& _e)
						{
							cwarn << "invalid hex character, code rejected";
							cwarn << boost::diagnostic_information(_e);
							init = bytes();
						}
						catch (...)
						{
							cwarn << "code rejected";
							init = bytes();
						}
						ssc.str(string());
						ssc << disassemble(init);
						cnote << "Init:";
						cnote << ssc.str();
					}
					u256 minGas = (u256)Transaction::gasRequired(init, 0);
					if (!init.size())
						cwarn << "Contract creation aborted, no init code.";
					else if (endowment < 0)
						cwarn << "Invalid endowment";
					else if (gas < minGas)
						cwarn << "Minimum gas amount is" << minGas;
					else
						c->submitTransaction(sigKey.secret(), endowment, init, gas, gasPrice);
				}
				else
					cwarn << "Require parameters: contract ENDOWMENT GASPRICE GAS CODEHEX";
			}
			else if (c && cmd == "dumpreceipt")
			{
				unsigned block;
				unsigned index;
				iss >> block >> index;
				dev::eth::TransactionReceipt r = c->blockChain().receipts(c->blockChain().numberHash(block)).receipts[index];
				auto rb = r.rlp();
				cout << "RLP: " << RLP(rb) << endl;
				cout << "Hex: " << toHex(rb) << endl;
				cout << r << endl;
			}
			else if (c && cmd == "dumptrace")
			{
				unsigned block;
				unsigned index;
				string filename;
				string format;
				iss >> block >> index >> filename >> format;
				ofstream f;
				f.open(filename);

				dev::eth::State state =c->state(index + 1,c->blockChain().numberHash(block));
				if (index < state.pending().size())
				{
					Executive e(state, c->blockChain(), 0);
					Transaction t = state.pending()[index];
					state = state.fromPending(index);
					try
					{
						OnOpFunc oof;
						if (format == "pretty")
							oof = [&](uint64_t steps, Instruction instr, bigint newMemSize, bigint gasCost, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
							{
								dev::eth::VM* vm = vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								f << endl << "    STACK" << endl;
								for (auto i: vm->stack())
									f << (h256)i << endl;
								f << "    MEMORY" << endl << dev::memDump(vm->memory());
								f << "    STORAGE" << endl;
								for (auto const& i: ext->state().storage(ext->myAddress))
									f << showbase << hex << i.first << ": " << i.second << endl;
								f << dec << ext->depth << " | " << ext->myAddress << " | #" << steps << " | " << hex << setw(4) << setfill('0') << vm->curPC() << " : " << dev::eth::instructionInfo(instr).name << " | " << dec << vm->gas() << " | -" << dec << gasCost << " | " << newMemSize << "x32";
							};
						else if (format == "standard")
							oof = [&](uint64_t, Instruction instr, bigint, bigint, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
							{
								dev::eth::VM* vm = vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								f << ext->myAddress << " " << hex << toHex(dev::toCompactBigEndian(vm->curPC(), 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)instr, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)vm->gas(), 1)) << endl;
							};
						else if (format == "standard+")
							oof = [&](uint64_t, Instruction instr, bigint, bigint, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
							{
								dev::eth::VM* vm = vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								if (instr == Instruction::STOP || instr == Instruction::RETURN || instr == Instruction::SUICIDE)
									for (auto const& i: ext->state().storage(ext->myAddress))
										f << toHex(dev::toCompactBigEndian(i.first, 1)) << " " << toHex(dev::toCompactBigEndian(i.second, 1)) << endl;
								f << ext->myAddress << " " << hex << toHex(dev::toCompactBigEndian(vm->curPC(), 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)instr, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)vm->gas(), 1)) << endl;
							};
						e.initialize(t);
						if (!e.execute())
							e.go(oof);
						e.finalize();
					}
					catch(Exception const& _e)
					{
						// TODO: a bit more information here. this is probably quite worrying as the transaction is already in the blockchain.
						cwarn << diagnostic_information(_e);
					}
				}
			}
			else if (c && cmd == "inspect")
			{
				string rechex;
				iss >> rechex;

				if (rechex.length() != 40)
					cwarn << "Invalid address length";
				else
				{
					auto h = h160(fromHex(rechex));
					stringstream s;

					try
					{
						auto storage =c->storageAt(h, PendingBlock);
						for (auto const& i: storage)
							s << "@" << showbase << hex << i.first << "    " << showbase << hex << i.second << endl;
						s << endl << disassemble( c->codeAt(h, PendingBlock)) << endl;

						string outFile = getDataDir() + "/" + rechex + ".evm";
						ofstream ofs;
						ofs.open(outFile, ofstream::binary);
						ofs.write(s.str().c_str(), s.str().length());
						ofs.close();

						cnote << "Saved" << rechex << "to" << outFile;
					}
					catch (dev::InvalidTrie)
					{
						cwarn << "Corrupted trie.";
					}
				}
			}
			else if (cmd == "setsecret")
			{
				if (iss.peek() != -1)
				{
					string hexSec;
					iss >> hexSec;
					sigKey = KeyPair(h256(fromHex(hexSec)));
				}
				else
					cwarn << "Require parameter: setSecret HEXSECRETKEY";
			}
			else if (cmd == "setaddress")
			{
				if (iss.peek() != -1)
				{
					string hexAddr;
					iss >> hexAddr;
					if (hexAddr.length() != 40)
						cwarn << "Invalid address length: " << hexAddr.length();
					else
					{
						try
						{
							coinbase = h160(fromHex(hexAddr, WhenError::Throw));
						}
						catch (BadHexCharacter& _e)
						{
							cwarn << "invalid hex character, coinbase rejected";
							cwarn << boost::diagnostic_information(_e);
						}
						catch (...)
						{
							cwarn << "coinbase rejected";
						}
					}
				}
				else
					cwarn << "Require parameter: setAddress HEXADDRESS";
			}
			else if (cmd == "exportconfig")
			{
				if (iss.peek() != -1)
				{
					string path;
					iss >> path;
					RLPStream config(2);
					config << sigKey.secret() << coinbase;
					writeFile(path, config.out());
				}
				else
					cwarn << "Require parameter: exportConfig PATH";
			}
			else if (cmd == "importconfig")
			{
				if (iss.peek() != -1)
				{
					string path;
					iss >> path;
					bytes b = contents(path);
					if (b.size())
					{
						RLP config(b);
						sigKey = KeyPair(config[0].toHash<Secret>());
						coinbase = config[1].toHash<Address>();
					}
					else
						cwarn << path << "has no content!";
				}
				else
					cwarn << "Require parameter: importConfig PATH";
			}
			else if (cmd == "help")
				interactiveHelp();
			else if (cmd == "exit")
				break;
			else
				cout << "Unrecognised command. Type 'help' for help in interactive mode." << endl;
		}
#if ETH_JSONRPC
		if (jsonrpcServer.get())
			jsonrpcServer->StopListening();
#endif
	}
	else if (c)
	{
		unsigned n =c->blockChain().details().number;
		if (mining)
			c->startMining();
		while (!g_exit)
		{
			if ( c->isMining() &&c->blockChain().details().number - n == mining)
				c->stopMining();
			this_thread::sleep_for(chrono::milliseconds(100));
		}
	}
	else
		while (!g_exit)
			this_thread::sleep_for(chrono::milliseconds(1000));

	StructuredLogger::stopping(clientImplString, dev::Version);
	auto netData = web3.saveNetwork();
	if (!netData.empty())
		writeFile((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp", netData);
	return 0;
}

