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

#include <libdevcore/FileSystem.h>
#include <libevmcore/Instruction.h>
#include <libdevcore/StructuredLogger.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/EthashAux.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <libethereum/All.h>
#include <libethcore/KeyManager.h>

#include <libwebthree/WebThree.h>
#if ETH_JSCONSOLE || !ETH_TRUE
#include <libjsconsole/JSConsole.h>
#endif
#if ETH_READLINE || !ETH_TRUE
#include <readline/readline.h>
#include <readline/history.h>
#endif
#if ETH_JSONRPC || !ETH_TRUE
#include <libweb3jsonrpc/AccountHolder.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#endif
#include "BuildInfo.h"
#if ETH_JSONRPC || !ETH_TRUE
#include "PhoneHome.h"
#include "Farm.h"
#endif
#include <ethminer/MinerAux.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
using namespace boost::algorithm;
using dev::eth::Instruction;

static bool g_silence = false;

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
		<< "    block  Gives the current block height." << endl
		<< "    accounts  Gives information on all owned accounts (balances, mining beneficiary and default signer)." << endl
		<< "    newaccount <name>  Creates a new account with the given name." << endl
		<< "    transact  Execute a given transaction." << endl
		<< "    send  Execute a given transaction with current secret." << endl
		<< "    contract  Create a new contract with current secret." << endl
		<< "    peers  List the peers that are connected" << endl
#if ETH_FATDB || !ETH_TRUE
		<< "    listaccounts  List the accounts on the network." << endl
		<< "    listcontracts  List the contracts on the network." << endl
#endif
		<< "    setsigningkey <addr>  Set the address with which to sign transactions." << endl
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
		<< "    --genesis-nonce <nonce>  Set the Genesis Nonce to the given hex nonce." << endl
		<< "    -s,--import-secret <secret>  Import a secret key into the key store and use as the default." << endl
		<< "    -S,--import-session-secret <secret>  Import a secret key into the key store and use as the default for this session only." << endl
		<< "    --sign-key <address>  Sign all transactions with the key of the given address." << endl
		<< "    --session-sign-key <address>  Sign all transactions with the key of the given address for this session only." << endl
		<< "    --master <password>  Give the master password for the key store." << endl
		<< "    --password <password>  Give a password for a private key." << endl
		<< endl
		<< "Client transacting:" << endl
		/*<< "    -B,--block-fees <n>  Set the block fee profit in the reference unit e.g. ¢ (default: 15)." << endl
		<< "    -e,--ether-price <n>  Set the ether price in the reference unit e.g. ¢ (default: 30.679)." << endl
		<< "    -P,--priority <0 - 100>  Default % priority of a transaction (default: 50)." << endl*/
		<< "    --ask <wei>  Set the minimum ask gas price under which no transactions will be mined (default 500000000000)." << endl
		<< "    --bid <wei>  Set the bid gas price for to pay for transactions (default 500000000000)." << endl
		<< endl
		<< "Client mining:" << endl
		<< "    -a,--address <addr>  Set the coinbase (mining payout) address to addr (default: auto)." << endl
		<< "    -m,--mining <on/off/number>  Enable mining, optionally for a specified number of blocks (default: off)" << endl
		<< "    -f,--force-mining  Mine even when there are no transactions to mine (default: off)" << endl
		<< "    -C,--cpu  When mining, use the CPU." << endl
		<< "    -G,--opencl  When mining use the GPU via OpenCL." << endl
		<< "    --opencl-platform <n>  When mining using -G/--opencl use OpenCL platform n (default: 0)." << endl
		<< "    --opencl-device <n>  When mining using -G/--opencl use OpenCL device n (default: 0)." << endl
		<< "    -t, --mining-threads <n> Limit number of CPU/GPU miners to n (default: use everything available on selected platform)" << endl
		<< endl
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
		<< endl;
	MinerCLI::streamHelp(cout);
	cout
		<< "Client structured logging:" << endl
		<< "    --structured-logging  Enable structured logging (default output to stdout)." << endl
		<< "    --structured-logging-format <format>  Set the structured logging time format." << endl
		<< "    --structured-logging-url <URL>  Set the structured logging destination (currently only file:// supported)." << endl
		<< "Import/export modes:" << endl
		<< "    -I,--import <file>  Import file as a concatenated series of blocks and exit." << endl
		<< "    -E,--export <file>  Export file as a concatenated series of blocks and exit." << endl
		<< "    --from <n>  Export only from block n; n may be a decimal, a '0x' prefixed hash, or 'latest'." << endl
		<< "    --to <n>  Export only to block n (inclusive); n may be a decimal, a '0x' prefixed hash, or 'latest'." << endl
		<< "    --only <n>  Equivalent to --export-from n --export-to n." << endl
		<< endl
		<< "General Options:" << endl
		<< "    -d,--db-path <path>  Load database from path (default: " << getDataDir() << ")" << endl
#if ETH_EVMJIT || !ETH_TRUE
		<< "    -J,--jit  Enable EVM JIT (default: off)." << endl
#endif
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 8)." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
#if ETH_JSCONSOLE || !ETH_TRUE
		<< "    --console Use interactive javascript console" << endl
#endif
		;
		exit(0);
}

string ethCredits(bool _interactive = false)
{
	std::ostringstream cout;
	if (_interactive)
		cout
			<< "Type 'exit' to quit" << endl << endl;
	return credits() + cout.str();
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

enum class OperationMode
{
	Node,
	Import,
	Export
};

enum class Format
{
	Binary,
	Hex,
	Human
};

void stopMiningAfterXBlocks(eth::Client* _c, unsigned _start, unsigned _mining)
{
	if (_c->isMining() && _c->blockChain().details().number - _start == _mining)
		_c->stopMining();
	this_thread::sleep_for(chrono::milliseconds(100));
}

int main(int argc, char** argv)
{
	// Init defaults
	Defaults::get();

	/// Operating mode.
	OperationMode mode = OperationMode::Node;
	string dbPath;

	/// File name for import/export.
	string filename;

	/// Hashes/numbers for export range.
	string exportFrom = "1";
	string exportTo = "latest";
	Format exportFormat = Format::Binary;

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
	unsigned peers = 11;
	bool bootstrap = false;
	unsigned networkId = 0;

	/// Mining params
	unsigned mining = 0;
	bool forceMining = false;
	Address signingKey;
	Address sessionKey;
	Address beneficiary = signingKey;

	/// Structured logging params
	bool structuredLogging = false;
	string structuredLoggingFormat = "%Y-%m-%dT%H:%M:%S";
	string structuredLoggingURL;

	/// Transaction params
	TransactionPriority priority = TransactionPriority::Medium;
//	double etherPrice = 30.679;
//	double blockFees = 15.0;
	u256 askPrice("500000000000");
	u256 bidPrice("500000000000");

	// javascript console
	bool useConsole = false;

	/// Wallet password stuff
	string masterPassword;

	string configFile = getDataDir() + "/config.rlp";
	bytes b = contents(configFile);

	strings passwordsToNote;
	Secrets toImport;
	if (b.size())
	{
		RLP config(b);
		if (config[0].size() == 32)	// secret key - import and forget.
		{
			Secret s = config[0].toHash<Secret>();
			toImport.push_back(s);
		}
		else							// new format - just use it as an address.
			signingKey = config[0].toHash<Address>();
		beneficiary = config[1].toHash<Address>();
	}

	MinerCLI m(MinerCLI::OperationMode::None);

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (m.interpretOption(i, argc, argv)) {}
		else if (arg == "--listen-ip" && i + 1 < argc)
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
		else if (arg == "--password" && i + 1 < argc)
			passwordsToNote.push_back(argv[++i]);
		else if (arg == "--master" && i + 1 < argc)
			masterPassword = argv[++i];
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
				beneficiary = h160(fromHex(argv[++i], WhenError::Throw));
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
		else if ((arg == "-s" || arg == "--import-secret") && i + 1 < argc)
		{
			Secret s(fromHex(argv[++i]));
			toImport.push_back(s);
			signingKey = toAddress(s);
		}
		else if ((arg == "-S" || arg == "--import-session-secret") && i + 1 < argc)
		{
			Secret s(fromHex(argv[++i]));
			toImport.push_back(s);
			sessionKey = toAddress(s);
		}
		else if ((arg == "--sign-key") && i + 1 < argc)
			sessionKey = Address(fromHex(argv[++i]));
		else if ((arg == "--session-sign-key") && i + 1 < argc)
			sessionKey = Address(fromHex(argv[++i]));
		else if (arg == "--structured-logging-format" && i + 1 < argc)
			structuredLoggingFormat = string(argv[++i]);
		else if (arg == "--structured-logging")
			structuredLogging = true;
		else if (arg == "--structured-logging-url" && i + 1 < argc)
		{
			structuredLogging = true;
			structuredLoggingURL = argv[++i];
		}
		else if ((arg == "-d" || arg == "--path" || arg == "--db-path") && i + 1 < argc)
			dbPath = argv[++i];
		else if (arg == "--genesis-nonce" && i + 1 < argc)
		{
			try
			{
				CanonBlockChain::setGenesisNonce(Nonce(argv[++i]));
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
/*		else if ((arg == "-B" || arg == "--block-fees") && i + 1 < argc)
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
		}*/
		else if (arg == "--ask" && i + 1 < argc)
		{
			try
			{
				askPrice = u256(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
		else if (arg == "--bid" && i + 1 < argc)
		{
			try
			{
				bidPrice = u256(argv[++i]);
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
#if ETH_JSCONSOLE
		else if (arg == "--console")
			useConsole = true;
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

	m.execute();

	KeyManager keyManager;
	for (auto const& s: passwordsToNote)
		keyManager.notePassword(s);

	{
		RLPStream config(2);
		config << signingKey << beneficiary;
		writeFile(configFile, config.out());
	}

	if (sessionKey)
		signingKey = sessionKey;

	if (!clientName.empty())
		clientName += "/";

	string logbuf;
	std::string additional;
	g_logPost = [&](std::string const& a, char const*){
		if (g_silence)
			logbuf += a + "\n";
		else
			cout << "\r           \r" << a << endl << additional << flush;
	};

	auto getPassword = [&](string const& prompt){
		auto s = g_silence;
		g_silence = true;
		cout << endl;
		string ret = dev::getPassword(prompt);
		g_silence = s;
		return ret;
	};
	auto getAccountPassword = [&](Address const& a){
		return getPassword("Enter password for address " + keyManager.accountDetails()[a].first + " (" + a.abridged() + "; hint:" + keyManager.accountDetails()[a].second + "): ");
	};

	StructuredLogger::get().initialize(structuredLogging, structuredLoggingFormat, structuredLoggingURL);
	VMFactory::setKind(jit ? VMKind::JIT : VMKind::Interpreter);
	auto netPrefs = publicIP.empty() ? NetworkPreferences(listenIP ,listenPort, upnp) : NetworkPreferences(publicIP, listenIP ,listenPort, upnp);
	auto nodesState = contents((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp");
	std::string clientImplString = "++eth/" + clientName + "v" + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) + (jit ? "/JIT" : "");
	dev::WebThreeDirect web3(
		clientImplString,
		dbPath,
		killChain,
		nodeMode == NodeMode::Full ? set<string>{"eth"/*, "shh"*/} : set<string>(),
		netPrefs,
		&nodesState);

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

	if (keyManager.exists())
	{
		if (masterPassword.empty() || !keyManager.load(masterPassword))
			while (true)
			{
				masterPassword = getPassword("Please enter your MASTER password: ");
				if (keyManager.load(masterPassword))
					break;
				cout << "Password invalid. Try again." << endl;
			}
	}
	else
	{
		while (masterPassword.empty())
		{
			masterPassword = getPassword("Please enter a MASTER password to protect your key store (make it strong!): ");
			string confirm = getPassword("Please confirm the password by entering it again: ");
			if (masterPassword != confirm)
			{
				cout << "Passwords were different. Try again." << endl;
				masterPassword.clear();
			}
		}
		keyManager.create(masterPassword);
	}

	for (auto const& s: toImport)
	{
		keyManager.import(s, "Imported key (UNSAFE)");
		if (!signingKey)
			signingKey = toAddress(s);
	}

	if (keyManager.accounts().empty())
		keyManager.import(Secret::random(), "Default key");

	cout << ethCredits();
	web3.setIdealPeerCount(peers);
//	std::shared_ptr<eth::BasicGasPricer> gasPricer = make_shared<eth::BasicGasPricer>(u256(double(ether / 1000) / etherPrice), u256(blockFees * 1000));
	std::shared_ptr<eth::TrivialGasPricer> gasPricer = make_shared<eth::TrivialGasPricer>(askPrice, bidPrice);
	eth::Client* c = nodeMode == NodeMode::Full ? web3.ethereum() : nullptr;
	StructuredLogger::starting(clientImplString, dev::Version);
	if (c)
	{
		c->setGasPricer(gasPricer);
		c->setForceMining(forceMining);
		c->setTurboMining(m.minerType() == MinerCLI::MinerType::GPU);
		c->setAddress(beneficiary);
		c->setNetworkId(networkId);
	}

	cout << "Transaction Signer: " << signingKey << endl;
	cout << "Mining Benefactor: " << beneficiary << endl;
	web3.startNetwork();
	cout << "Node ID: " << web3.enode() << endl;

	if (bootstrap)
		for (auto const& i: Host::pocHosts())
			web3.requirePeer(i.first, i.second);
	if (remoteHost.size())
		web3.addNode(p2p::NodeId(), remoteHost + ":" + toString(remotePort));

#if ETH_JSONRPC || !ETH_TRUE
	shared_ptr<WebThreeStubServer> jsonrpcServer;
	unique_ptr<jsonrpc::AbstractServerConnector> jsonrpcConnector;
	if (jsonrpc > -1)
	{
		jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonrpc, "", "", SensibleHttpThreads));
		jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, make_shared<SimpleAccountHolder>([&](){return web3.ethereum();}, getAccountPassword, keyManager), vector<KeyPair>()));
		jsonrpcServer->StartListening();
	}
#endif

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	if (interactive)
	{
		additional = "Press Enter";
		string l;
		while (!g_exit)
		{
			g_silence = false;
			cout << logbuf << "Press Enter" << flush;
			std::getline(cin, l);
			logbuf.clear();
			g_silence = true;

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
/*			else if (c && cmd == "setblockfees")
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
			}*/
			else if (cmd == "verbosity")
			{
				if (iss.peek() != -1)
					iss >> g_logVerbosity;
				cout << "Verbosity: " << g_logVerbosity << endl;
			}
#if ETH_JSONRPC || !ETH_TRUE
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
				jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, make_shared<SimpleAccountHolder>([&](){return web3.ethereum();}, getAccountPassword, keyManager), vector<KeyPair>()));
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
				cout << "Current mining beneficiary:" << endl << beneficiary << endl;
				cout << "Current signing account:" << endl << signingKey << endl;
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
			else if (cmd == "newaccount")
			{
				string name;
				std::getline(iss, name);
				auto s = Secret::random();
				string password;
				while (password.empty())
				{
					password = getPassword("Please enter a password to protect this key (press enter for protection only be the MASTER password/keystore): ");
					string confirm = getPassword("Please confirm the password by entering it again: ");
					if (password != confirm)
					{
						cout << "Passwords were different. Try again." << endl;
						password.clear();
					}
				}
				if (!password.empty())
				{
					cout << "Enter a hint for this password: " << flush;
					string hint;
					std::getline(cin, hint);
					keyManager.import(s, name, password, hint);
				}
				else
					keyManager.import(s, name);
				cout << "New account created: " << toAddress(s);
			}
			else if (c && cmd == "accounts")
			{
				cout << "Accounts:" << endl;
				u256 total = 0;
				for (auto const& i: keyManager.accountDetails())
				{
					auto b = c->balanceAt(i.first);
					cout << ((i.first == signingKey) ? "SIGNING " : "        ") << ((i.first == beneficiary) ? "COINBASE " : "         ") << i.second.first << " (" << i.first << "): " << formatBalance(b) << " = " << b << " wei" << endl;
					total += b;
				}
				cout << "Total: " << formatBalance(total) << " = " << total << " wei" << endl;
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
							c->submitTransaction(keyManager.secret(signingKey, [&](){ return getAccountPassword(signingKey); }), amount, dest, bytes(), minGas);
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
						c->submitTransaction(keyManager.secret(signingKey, [&](){ return getAccountPassword(signingKey); }), endowment, init, gas, gasPrice);
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
							oof = [&](uint64_t steps, Instruction instr, bigint newMemSize, bigint gasCost, bigint gas, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
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
								f << dec << ext->depth << " | " << ext->myAddress << " | #" << steps << " | " << hex << setw(4) << setfill('0') << vm->curPC() << " : " << dev::eth::instructionInfo(instr).name << " | " << dec << gas << " | -" << dec << gasCost << " | " << newMemSize << "x32";
							};
						else if (format == "standard")
							oof = [&](uint64_t, Instruction instr, bigint, bigint, bigint gas, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
							{
								dev::eth::VM* vm = vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								f << ext->myAddress << " " << hex << toHex(dev::toCompactBigEndian(vm->curPC(), 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)instr, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)gas, 1)) << endl;
							};
						else if (format == "standard+")
							oof = [&](uint64_t, Instruction instr, bigint, bigint, bigint gas, dev::eth::VM* vvm, dev::eth::ExtVMFace const* vextVM)
							{
								dev::eth::VM* vm = vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								if (instr == Instruction::STOP || instr == Instruction::RETURN || instr == Instruction::SUICIDE)
									for (auto const& i: ext->state().storage(ext->myAddress))
										f << toHex(dev::toCompactBigEndian(i.first, 1)) << " " << toHex(dev::toCompactBigEndian(i.second, 1)) << endl;
								f << ext->myAddress << " " << hex << toHex(dev::toCompactBigEndian(vm->curPC(), 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)instr, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)gas, 1)) << endl;
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
			else if (cmd == "setsigningkey")
			{
				if (iss.peek() != -1)
				{
					string hexSec;
					iss >> hexSec;
					signingKey = Address(fromHex(hexSec));
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
							beneficiary = h160(fromHex(hexAddr, WhenError::Throw));
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
					config << signingKey << beneficiary;
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
						signingKey = config[0].toHash<Address>();
						beneficiary = config[1].toHash<Address>();
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
		if (useConsole)
		{
#if ETH_JSCONSOLE
			JSConsole console(web3, make_shared<SimpleAccountHolder>([&](){return web3.ethereum();}, getAccountPassword, keyManager));
			while (!g_exit)
			{
				console.repl();
				stopMiningAfterXBlocks(c, n, mining);
			}
#endif
		}
		else
			while (!g_exit)
				stopMiningAfterXBlocks(c, n, mining);
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

