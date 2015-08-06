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
#include <libethcore/EthashAux.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <libethereum/All.h>
#include <libethereum/BlockChainSync.h>
#include <libethcore/KeyManager.h>

#include <libwebthree/WebThree.h>
#if ETH_JSCONSOLE || !ETH_TRUE
#include <libjsconsole/JSLocalConsole.h>
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

static std::atomic<bool> g_silence = {false};

void interactiveHelp()
{
	cout
		<< "Commands:" << endl
		<< "    netstart <port>  Starts the network subsystem on a specific port." << endl
		<< "    netstop  Stops the network subsystem." << endl
		<< "    connect <addr> <port>  Connects to a specific peer." << endl
		<< "    verbosity (<level>)  Gets or sets verbosity level." << endl
		<< "    minestart  Starts mining." << endl
		<< "    minestop  Stops mining." << endl
		<< "    mineforce <enable>  Forces mining, even when there are no transactions." << endl
		<< "    block  Gives the current block height." << endl
		<< "    blockhashfromnumber <number>  Gives the block hash with the givne number." << endl
		<< "    numberfromblockhash <hash>  Gives the block number with the given hash." << endl
		<< "    blockqueue  Gives the current block queue status." << endl
		<< "    findblock <hash>  Searches for the block in the blockchain and blockqueue." << endl
		<< "    firstunknown  Gives the first unknown block from the blockqueue." << endl
		<< "    retryunknown  retries to import all unknown blocks from the blockqueue." << endl
		<< "    accounts  Gives information on all owned accounts (balances, mining beneficiary and default signer)." << endl
		<< "    newaccount <name>  Creates a new account with the given name." << endl
		<< "    transact  Execute a given transaction." << endl
		<< "    transactnonce  Execute a given transaction with a specified nonce." << endl
		<< "    txcreate  Execute a given contract creation transaction." << endl
		<< "    send  Execute a given transaction with current secret." << endl
		<< "    contract  Create a new contract with current secret." << endl
		<< "    peers  List the peers that are connected" << endl
#if ETH_FATDB || !ETH_TRUE
		<< "    listaccounts  List the accounts on the network." << endl
		<< "    listcontracts  List the contracts on the network." << endl
		<< "    balanceat <address>  Gives the balance of the given account." << endl
		<< "    balanceatblock <address> <blocknumber>  Gives the balance of the given account." << endl
		<< "    storageat <address>  Gives the storage of the given account." << endl
		<< "    storageatblock <address> <blocknumber>  Gives the storahe of the given account at a given blocknumber." << endl
		<< "    codeat <address>  Gives the code of the given account." << endl
#endif
		<< "    setsigningkey <addr>  Set the address with which to sign transactions." << endl
		<< "    setaddress <addr>  Set the coinbase (mining payout) address." << endl
		<< "    exportconfig <path>  Export the config (.RLP) to the path provided." << endl
		<< "    importconfig <path>  Import the config (.RLP) from the path provided." << endl
		<< "    inspect <contract>  Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    reprocess <block>  Reprocess a given block." << endl
		<< "    dumptrace <block> <index> <filename> <format>  Dumps a transaction trace" << endl << "to <filename>. <format> should be one of pretty, standard, standard+." << endl
		<< "    dumpreceipt <block> <index>  Dumps a transation receipt." << endl
		<< "    hashrate  Print the current hashrate in hashes per second if the client is mining." << endl
		<< "    exit  Exits the application." << endl;
}

void help()
{
	cout
		<< "Usage eth [OPTIONS]" << endl
		<< "Options:" << endl << endl
		<< "Client mode (default):" << endl
		<< "    --olympic  Use the Olympic (0.9) protocol." << endl
		<< "    --frontier  Use the Frontier (1.0) protocol." << endl
		<< "    --private <name>  Use a private chain." << endl
		<< "    --genesis-json <file>  Import the genesis block information from the given json file." << endl
		<< endl
		<< "    -o,--mode <full/peer>  Start a full node or a peer node (default: full)." << endl
#if ETH_JSCONSOLE || !ETH_TRUE
		<< "    -i,--interactive  Enter interactive mode (default: non-interactive)." << endl
#endif
		<< endl
#if ETH_JSONRPC || !ETH_TRUE
		<< "    -j,--json-rpc  Enable JSON-RPC server (default: off)." << endl
		<< "    --json-rpc-port <n>  Specify JSON-RPC server port (implies '-j', default: " << SensibleHttpPort << ")." << endl
		<< "    --admin <password>  Specify admin session key for JSON-RPC (default: auto-generated and printed at startup)." << endl
#endif
		<< "    -K,--kill  First kill the blockchain." << endl
		<< "    -R,--rebuild  Rebuild the blockchain from the existing database." << endl
		<< "    --rescue  Attempt to rescue a corrupt database." << endl
		<< endl
		<< "    --import-presale <file>  Import a presale key; you'll need to type the password to this." << endl
		<< "    -s,--import-secret <secret>  Import a secret key into the key store and use as the default." << endl
		<< "    -S,--import-session-secret <secret>  Import a secret key into the key store and use as the default for this session only." << endl
		<< "    --sign-key <address>  Sign all transactions with the key of the given address." << endl
		<< "    --session-sign-key <address>  Sign all transactions with the key of the given address for this session only." << endl
		<< "    --master <password>  Give the master password for the key store." << endl
		<< "    --password <password>  Give a password for a private key." << endl
		<< "    --sentinel <server>  Set the sentinel for reporting bad blocks or chain issues." << endl
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
		<< "    --mine-on-wrong-chain  Mine even when we know it's the wrong chain (default: off)" << endl
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
		<< "    --network-id <n> Only connect to other hosts with this network id." << endl
		<< "    --upnp <on/off>  Use UPnP for NAT (default: on)." << endl
		<< "    --no-discovery  Disable Node discovery." << endl
		<< "    --pin  Only connect to required (trusted) peers." << endl
		<< "    --hermit  Equivalent to --no-discovery --pin." << endl
		<< "    --sociable  Forces discovery and no pinning." << endl
//		<< "    --require-peers <peers.json>  List of required (trusted) peers. (experimental)" << endl
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
		<< "    --dont-check  Avoids checking some of the aspects of blocks. Faster importing, but only do if you know the data is valid." << endl
		<< endl
		<< "General Options:" << endl
		<< "    -d,--db-path <path>  Load database from path (default: " << getDataDir() << ")" << endl
#if ETH_EVMJIT || !ETH_TRUE
		<< "    --vm <vm-kind>  Select VM. Options are: interpreter, jit, smart. (default: interpreter)" << endl
#endif
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 8)." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
		<< endl
		<< "Experimental / Proof of Concept:" << endl
		<< "    --shh  Enable Whisper" << endl
		<< endl
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

void importPresale(KeyManager& _km, string const& _file, function<string()> _pass)
{
	KeyPair k = _km.presaleSecret(contentsString(_file), [&](bool){ return _pass(); });
	_km.import(k.secret(), "Presale wallet" + _file + " (insecure)");
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

inline bool isPrime(unsigned _number)
{
	if (((!(_number & 1)) && _number != 2 ) || (_number < 2) || (_number % 3 == 0 && _number != 3))
		return false;
	for(unsigned k = 1; 36 * k * k - 12 * k < _number; ++k)
		if ((_number % (6 * k + 1) == 0) || (_number % (6 * k - 1) == 0))
			return false;
	return true;
}

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

void stopMiningAfterXBlocks(eth::Client* _c, unsigned _start, unsigned& io_mining)
{
	if (io_mining != ~(unsigned)0 && io_mining && _c->isMining() && _c->blockChain().details().number - _start == io_mining)
	{
		_c->stopMining();
		io_mining = ~(unsigned)0;
	}
	this_thread::sleep_for(chrono::milliseconds(100));
}

void interactiveMode(eth::Client*, std::shared_ptr<eth::TrivialGasPricer>, WebThreeDirect&, KeyManager&, string&, string&, function<string(string const&)>, function<string(Address const&)>, NetworkPreferences, Address, Address, TransactionPriority) {}

int main(int argc, char** argv)
{
	// Init defaults
	Defaults::get();

	/// Operating mode.
	OperationMode mode = OperationMode::Node;
	string dbPath;
//	unsigned prime = 0;
//	bool yesIReallyKnowWhatImDoing = false;

	/// File name for import/export.
	string filename;
	bool safeImport = false;

	/// Hashes/numbers for export range.
	string exportFrom = "1";
	string exportTo = "latest";
	Format exportFormat = Format::Binary;

	/// General params for Node operation
	NodeMode nodeMode = NodeMode::Full;
	bool interactive = false;
#if ETH_JSONRPC || !ETH_TRUE
	int jsonRPCURL = -1;
#endif
	string jsonAdmin;
	string genesisJSON;
	dev::eth::Network releaseNetwork = c_network;
	u256 gasFloor = UndefinedU256;
	string privateChain;

	bool upnp = true;
	WithExisting withExisting = WithExisting::Trust;
	string sentinel;

	/// Networking params.
	string clientName;
	string listenIP;
	unsigned short listenPort = 30303;
	string publicIP;
	string remoteHost;
	unsigned short remotePort = 30303;
	unsigned peers = 11;
	bool bootstrap = false;
	bool disableDiscovery = false;
	bool pinning = false;
	bool enableDiscovery = false;
	bool noPinning = false;
	unsigned networkId = (unsigned)-1;

	/// Mining params
	unsigned mining = 0;
	bool forceMining = false;
	bool mineOnWrongChain = false;
	Address signingKey;
	Address sessionKey;
	Address beneficiary = signingKey;
	strings presaleImports;

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
	
	/// Whisper
	bool useWhisper = false;

	string configFile = getDataDir() + "/config.rlp";
	bytes b = contents(configFile);

	strings passwordsToNote;
	Secrets toImport;
	if (b.size())
	{
		try
		{
			RLP config(b);
			signingKey = config[0].toHash<Address>();
			beneficiary = config[1].toHash<Address>();
		}
		catch (...) {}
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
		else if (arg == "--dont-check")
			safeImport = true;
		else if ((arg == "-E" || arg == "--export") && i + 1 < argc)
		{
			mode = OperationMode::Export;
			filename = argv[++i];
		}
/*		else if (arg == "--prime" && i + 1 < argc)
			try
			{
				prime = stoi(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "--yes-i-really-know-what-im-doing")
			yesIReallyKnowWhatImDoing = true;
*/		else if (arg == "--sentinel" && i + 1 < argc)
			sentinel = argv[++i];
		else if (arg == "--mine-on-wrong-chain")
			mineOnWrongChain = true;
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
		else if (arg == "--private" && i + 1 < argc)
			try {
				privateChain = argv[++i];
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		else if (arg == "-K" || arg == "--kill-blockchain" || arg == "--kill")
			withExisting = WithExisting::Kill;
		else if (arg == "-R" || arg == "--rebuild")
			withExisting = WithExisting::Verify;
		else if (arg == "-R" || arg == "--rescue")
			withExisting = WithExisting::Rescue;
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
			toImport.emplace_back(s);
			signingKey = toAddress(s);
		}
		else if ((arg == "-S" || arg == "--import-session-secret") && i + 1 < argc)
		{
			Secret s(fromHex(argv[++i]));
			toImport.emplace_back(s);
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
		else if ((arg == "--genesis-json" || arg == "--genesis") && i + 1 < argc)
		{
			try
			{
				genesisJSON = contentsString(argv[++i]);
			}
			catch (...)
			{
				cerr << "Bad " << arg << " option: " << argv[i] << endl;
				return -1;
			}
		}
		else if (arg == "--frontier")
			releaseNetwork = eth::Network::Frontier;
		else if (arg == "--gas-floor" && i + 1 < argc)
			gasFloor = u256(argv[++i]);
		else if (arg == "--olympic")
			releaseNetwork = eth::Network::Olympic;
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
		else if (arg == "--no-discovery")
			disableDiscovery = true;
		else if (arg == "--pin")
			pinning = true;
		else if (arg == "--hermit")
			pinning = disableDiscovery = true;
		else if (arg == "--sociable")
			noPinning = enableDiscovery = true;
		else if (arg == "--import-presale" && i + 1 < argc)
			presaleImports.push_back(argv[++i]);
		else if (arg == "-f" || arg == "--force-mining")
			forceMining = true;
		else if (arg == "--old-interactive")
			interactive = true;
#if ETH_JSONRPC || !ETH_TRUE
		else if ((arg == "-j" || arg == "--json-rpc"))
			jsonRPCURL = jsonRPCURL == -1 ? SensibleHttpPort : jsonRPCURL;
		else if (arg == "--json-rpc-port" && i + 1 < argc)
			jsonRPCURL = atoi(argv[++i]);
		else if (arg == "--json-admin" && i + 1 < argc)
			jsonAdmin = argv[++i];
#endif
#if ETH_JSCONSOLE || !ETH_TRUE
		else if (arg == "-i" || arg == "--interactive" || arg == "--console")
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
		else if (arg == "--vm" && i + 1 < argc)
		{
			string vmKind = argv[++i];
			if (vmKind == "interpreter")
				VMFactory::setKind(VMKind::Interpreter);
			else if (vmKind == "jit")
				VMFactory::setKind(VMKind::JIT);
			else if (vmKind == "smart")
				VMFactory::setKind(VMKind::Smart);
			else
			{
				cerr << "Unknown VM kind: " << vmKind << endl;
				return -1;
			}
		}
#endif
		else if (arg == "--shh")
			useWhisper = true;
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

	// Set up all the chain config stuff.
	resetNetwork(releaseNetwork);
	if (!privateChain.empty())
		CanonBlockChain<Ethash>::forceGenesisExtraData(sha3(privateChain).asBytes());
	if (!genesisJSON.empty())
		CanonBlockChain<Ethash>::setGenesis(genesisJSON);
	if (gasFloor != UndefinedU256)
		c_gasFloorTarget = gasFloor;
	if (networkId == (unsigned)-1)
		networkId =  (unsigned)c_network;

	if (g_logVerbosity > 0)
	{
		cout << EthGrayBold "(++)Ethereum" EthReset << endl;
		if (c_network == eth::Network::Olympic)
			cout << "Welcome to Olympic!" << endl;
		else if (c_network == eth::Network::Frontier)
			cout << "Welcome to the " EthMaroonBold "Frontier" EthReset "!" << endl;
	}

	m.execute();

	KeyManager keyManager;
	for (auto const& s: passwordsToNote)
		keyManager.notePassword(s);

	writeFile(configFile, rlpList(signingKey, beneficiary));

	if (sessionKey)
		signingKey = sessionKey;

	if (!clientName.empty())
		clientName += "/";

	string logbuf;
	std::string additional;
	if (interactive)
		g_logPost = [&](std::string const& a, char const*){
			static SpinLock s_lock;
			SpinGuard l(s_lock);

			if (g_silence)
				logbuf += a + "\n";
			else
				cout << "\r           \r" << a << endl << additional << flush;

			// helpful to use OutputDebugString on windows
	#ifdef _WIN32
			{
				OutputDebugStringA(a.data());
				OutputDebugStringA("\n");
			}
	#endif
		};

	auto getPassword = [&](string const& prompt){
		bool s = g_silence;
		g_silence = true;
		cout << endl;
		string ret = dev::getPassword(prompt);
		g_silence = s;
		return ret;
	};
	auto getAccountPassword = [&](Address const& a){
		return getPassword("Enter password for address " + keyManager.accountName(a) + " (" + a.abridged() + "; hint:" + keyManager.passwordHint(a) + "): ");
	};

	StructuredLogger::get().initialize(structuredLogging, structuredLoggingFormat, structuredLoggingURL);
	auto netPrefs = publicIP.empty() ? NetworkPreferences(listenIP, listenPort, upnp) : NetworkPreferences(publicIP, listenIP ,listenPort, upnp);
	netPrefs.discovery = (privateChain.empty() && !disableDiscovery) || enableDiscovery;
	netPrefs.pin = (pinning || !privateChain.empty()) && !noPinning;

	auto nodesState = contents((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp");
	auto caps = useWhisper ? set<string>{"eth", "shh"} : set<string>{"eth"};
	dev::WebThreeDirect web3(
		WebThreeDirect::composeClientVersion("++eth", clientName),
		dbPath,
		withExisting,
		nodeMode == NodeMode::Full ? caps : set<string>(),
		netPrefs,
		&nodesState);
	web3.ethereum()->setMineOnBadChain(mineOnWrongChain);
	web3.ethereum()->setSentinel(sentinel);

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
		chrono::steady_clock::time_point t = chrono::steady_clock::now();
		double last = 0;
		unsigned lastImported = 0;
		unsigned imported = 0;
		while (in.peek() != -1)
		{
			bytes block(8);
			in.read((char*)block.data(), 8);
			block.resize(RLP(block, RLP::LaissezFaire).actualSize());
			in.read((char*)block.data() + 8, block.size() - 8);

			switch (web3.ethereum()->queueBlock(block, safeImport))
			{
			case ImportResult::Success: good++; break;
			case ImportResult::AlreadyKnown: alreadyHave++; break;
			case ImportResult::UnknownParent: unknownParent++; break;
			case ImportResult::FutureTimeUnknown: unknownParent++; futureTime++; break;
			case ImportResult::FutureTimeKnown: futureTime++; break;
			default: bad++; break;
			}

			// sync chain with queue
			tuple<ImportRoute, bool, unsigned> r = web3.ethereum()->syncQueue(10);
			imported += get<2>(r);

			double e = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t).count() / 1000.0;
			if ((unsigned)e >= last + 10)
			{
				auto i = imported - lastImported;
				auto d = e - last;
				cout << i << " more imported at " << (round(i * 10 / d) / 10) << " blocks/s. " << imported << " imported in " << e << " seconds at " << (round(imported * 10 / e) / 10) << " blocks/s (#" << web3.ethereum()->number() << ")" << endl;
				last = (unsigned)e;
				lastImported = imported;
//				cout << web3.ethereum()->blockQueueStatus() << endl;
			}
		}

		while (web3.ethereum()->blockQueue().items().first + web3.ethereum()->blockQueue().items().second > 0)
		{
			this_thread::sleep_for(chrono::seconds(1));
			web3.ethereum()->syncQueue(100000);
		}
		double e = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t).count() / 1000.0;
		cout << imported << " imported in " << e << " seconds at " << (round(imported * 10 / e) / 10) << " blocks/s (#" << web3.ethereum()->number() << ")" << endl;
		return 0;
	}
/*
	if (c_network == eth::Network::Frontier && !yesIReallyKnowWhatImDoing)
	{
		auto pd = contents(getDataDir() + "primes");
		unordered_set<unsigned> primes = RLP(pd).toUnorderedSet<unsigned>();
		while (true)
		{
			if (!prime)
				try
				{
					prime = stoi(getPassword("To enter the Frontier, enter a 6 digit prime that you have not entered before: "));
				}
				catch (...) {}
			if (isPrime(prime) && !primes.count(prime))
				break;
			prime = 0;
		}
		primes.insert(prime);
		writeFile(getDataDir() + "primes", rlp(primes));
	}
*/
	if (keyManager.exists())
	{
		if (masterPassword.empty() || !keyManager.load(masterPassword))
			while (true)
			{
				masterPassword = getPassword("Please enter your MASTER password: ");
				if (keyManager.load(masterPassword))
					break;
				cout << "The password you entered is incorrect. If you have forgotten your password, and you wish to start afresh, manually remove the file: " + getDataDir("ethereum") + "/keys.info" << endl;
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

	for (auto const& presale: presaleImports)
		importPresale(keyManager, presale, [&](){ return getPassword("Enter your wallet password for " + presale + ": "); });

	for (auto const& s: toImport)
	{
		keyManager.import(s, "Imported key (UNSAFE)");
		if (!signingKey)
			signingKey = toAddress(s);
	}

	if (keyManager.accounts().empty())
	{
		h128 uuid = keyManager.import(Secret::random(), "Default key");
		if (!beneficiary)
			beneficiary = keyManager.address(uuid);
		if (!signingKey)
			signingKey = keyManager.address(uuid);
		writeFile(configFile, rlpList(signingKey, beneficiary));
	}

	cout << ethCredits();
	web3.setIdealPeerCount(peers);
//	std::shared_ptr<eth::BasicGasPricer> gasPricer = make_shared<eth::BasicGasPricer>(u256(double(ether / 1000) / etherPrice), u256(blockFees * 1000));
	std::shared_ptr<eth::TrivialGasPricer> gasPricer = make_shared<eth::TrivialGasPricer>(askPrice, bidPrice);
	eth::Client* c = nodeMode == NodeMode::Full ? web3.ethereum() : nullptr;
	StructuredLogger::starting(WebThreeDirect::composeClientVersion("++eth", clientName), dev::Version);
	if (c)
	{
		c->setGasPricer(gasPricer);
		c->setForceMining(forceMining);
		// TODO: expose sealant interface.
		c->setShouldPrecomputeDAG(m.shouldPrecompute());
		c->setTurboMining(m.minerType() == MinerCLI::MinerType::GPU);
		c->setBeneficiary(beneficiary);
		c->setNetworkId(networkId);
	}

	cout << "Transaction Signer: " << signingKey << endl;
	cout << "Mining Benefactor: " << beneficiary << endl;

	if (bootstrap || !remoteHost.empty() || disableDiscovery)
	{
		web3.startNetwork();
		cout << "Node ID: " << web3.enode() << endl;
	}
	else
		cout << "Networking disabled. To start, use netstart or pass -b or a remote host." << endl;

	if (useConsole && jsonRPCURL == -1)
		jsonRPCURL = SensibleHttpPort;

#if ETH_JSONRPC || !ETH_TRUE
	shared_ptr<dev::WebThreeStubServer> jsonrpcServer;
	unique_ptr<jsonrpc::AbstractServerConnector> jsonrpcConnector;
	if (jsonRPCURL > -1)
	{
		jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonRPCURL, "", "", SensibleHttpThreads));
		jsonrpcServer = make_shared<dev::WebThreeStubServer>(*jsonrpcConnector.get(), web3, make_shared<SimpleAccountHolder>([&](){ return web3.ethereum(); }, getAccountPassword, keyManager), vector<KeyPair>(), keyManager, *gasPricer);
		jsonrpcServer->StartListening();
		if (jsonAdmin.empty())
			jsonAdmin = jsonrpcServer->newSession(SessionPermissions{{Privilege::Admin}});
		else
			jsonrpcServer->addSession(jsonAdmin, SessionPermissions{{Privilege::Admin}});
		cout << "JSONRPC Admin Session Key: " << jsonAdmin << endl;
		writeFile(getDataDir("web3") + "/session.key", jsonAdmin);
		writeFile(getDataDir("web3") + "/session.url", "http://localhost:" + toString(jsonRPCURL));
	}
#endif

	if (bootstrap)
		for (auto const& i: Host::pocHosts())
			web3.requirePeer(i.first, i.second);
	if (!remoteHost.empty())
		web3.addNode(p2p::NodeId(), remoteHost + ":" + toString(remotePort));

	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	if (interactive)
		interactiveMode(c, gasPricer, web3, keyManager, logbuf, additional, getPassword, getAccountPassword, netPrefs, beneficiary, signingKey, priority);
	else if (c)
	{
		unsigned n = c->blockChain().details().number;
		if (mining)
			c->startMining();
		if (useConsole)
		{
#if ETH_JSCONSOLE || !ETH_TRUE
			JSLocalConsole console;
			shared_ptr<dev::WebThreeStubServer> rpcServer = make_shared<dev::WebThreeStubServer>(*console.connector(), web3, make_shared<SimpleAccountHolder>([&](){ return web3.ethereum(); }, getAccountPassword, keyManager), vector<KeyPair>(), keyManager, *gasPricer);
			string sessionKey = rpcServer->newSession(SessionPermissions{{Privilege::Admin}});
			console.eval("web3.admin.setSessionKey('" + sessionKey + "')");
			while (!g_exit)
			{
				console.readExpression();
				stopMiningAfterXBlocks(c, n, mining);
			}
			rpcServer->StopListening();
#endif
		}
		else
			while (!g_exit)
				stopMiningAfterXBlocks(c, n, mining);
	}
	else
		while (!g_exit)
			this_thread::sleep_for(chrono::milliseconds(1000));

#if ETH_JSONRPC
	if (jsonrpcServer.get())
		jsonrpcServer->StopListening();
#endif

	StructuredLogger::stopping(WebThreeDirect::composeClientVersion("++eth", clientName), dev::Version);
	auto netData = web3.saveNetwork();
	if (!netData.empty())
		writeFile((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp", netData);
	return 0;
}
