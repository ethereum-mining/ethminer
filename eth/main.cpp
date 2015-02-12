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
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <libethereum/All.h>
#include <libwebthree/WebThree.h>
#if ETH_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif
#if ETH_JSONRPC
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#endif
#include "BuildInfo.h"
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
		<< "    listAccounts  List the accounts on the network." << endl
		<< "    listContracts  List the contracts on the network." << endl
		<< "    setSecret <secret>  Set the secret to the hex secret key." <<endl
		<< "    setAddress <addr>  Set the coinbase (mining payout) address." <<endl
		<< "    exportConfig <path>  Export the config (.RLP) to the path provided." <<endl
		<< "    importConfig <path>  Import the config (.RLP) from the path provided." <<endl
		<< "    inspect <contract>  Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    dumptrace <block> <index> <filename> <format>  Dumps a transaction trace" << endl << "to <filename>. <format> should be one of pretty, standard, standard+." << endl
		<< "    dumpreceipt <block> <index>  Dumps a transation receipt." << endl
		<< "    exit  Exits the application." << endl;
}

void help()
{
	cout
        << "Usage eth [OPTIONS] <remote-host>" << endl
        << "Options:" << endl
        << "    -a,--address <addr>  Set the coinbase (mining payout) address to addr (default: auto)." << endl
		<< "    -b,--bootstrap  Connect to the default Ethereum peerserver." << endl
        << "    -c,--client-name <name>  Add a name to your client's version string (default: blank)." << endl
        << "    -d,--db-path <path>  Load database from path (default:  ~/.ethereum " << endl
        << "                         <APPDATA>/Etherum or Library/Application Support/Ethereum)." << endl
		<< "    -f,--force-mining  Mine even when there are no transaction to mine (Default: off)" << endl
		<< "    -h,--help  Show this help message and exit." << endl
        << "    -i,--interactive  Enter interactive mode (default: non-interactive)." << endl
#if ETH_JSONRPC
		<< "    -j,--json-rpc  Enable JSON-RPC server (default: off)." << endl
		<< "    --json-rpc-port  Specify JSON-RPC server port (implies '-j', default: 8080)." << endl
#endif
        << "    -l,--listen <port>  Listen on the given port for incoming connected (default: 30303)." << endl
		<< "    -m,--mining <on/off/number>  Enable mining, optionally for a specified number of blocks (Default: off)" << endl
		<< "    -n,--upnp <on/off>  Use upnp for NAT (default: on)." << endl
		<< "    -L,--local-networking Use peers whose addresses are local." << endl
		<< "    -o,--mode <full/peer>  Start a full node or a peer node (Default: full)." << endl
        << "    -p,--port <port>  Connect to remote port (default: 30303)." << endl
        << "    -r,--remote <host>  Connect to remote host (default: none)." << endl
        << "    -s,--secret <secretkeyhex>  Set the secret key for use with send command (default: auto)." << endl
		<< "    -t,--miners <number>  Number of mining threads to start (Default: " << thread::hardware_concurrency() << ")" << endl
        << "    -u,--public-ip <ip>  Force public ip to given (default; auto)." << endl
        << "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (Default: 8)." << endl
        << "    -x,--peers <number>  Attempt to connect to given number of peers (Default: 5)." << endl
        << "    -V,--version  Show the version and exit." << endl
#if ETH_EVMJIT
		<< "    --jit  Use EVM JIT (default: off)." << endl
#endif
		;
        exit(0);
}

string credits(bool _interactive = false)
{
	std::ostringstream cout;
	cout
		<< "Ethereum (++) " << dev::Version << endl
		<< "  Code by Gav Wood, (c) 2013, 2014." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;

	if (_interactive)
	{
		cout << "Type 'netstart 30303' to start networking" << endl;
		cout << "Type 'connect " << Host::pocHost() << " 30303' to connect" << endl;
		cout << "Type 'exit' to quit" << endl << endl;
	}
	return cout.str();
}

void version()
{
	cout << "eth version " << dev::Version << endl;
	cout << "Network protocol version: " << dev::eth::c_protocolVersion << endl;
	cout << "Client database version: " << dev::eth::c_databaseVersion << endl;
	cout << "Build: " << DEV_QUOTED(ETH_BUILD_PLATFORM) << "/" << DEV_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
}

Address c_config = Address("ccdeac59d35627b7de09332e819d5159e7bb7250");
string pretty(h160 _a, dev::eth::State _st)
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

int main(int argc, char** argv)
{
	unsigned short listenPort = 30303;
	string remoteHost;
	unsigned short remotePort = 30303;
	string dbPath;
	unsigned mining = ~(unsigned)0;
	NodeMode mode = NodeMode::Full;
	unsigned peers = 5;
	int miners = -1;
	bool interactive = false;
#if ETH_JSONRPC
	int jsonrpc = -1;
#endif
	string publicIP;
	bool bootstrap = false;
	bool upnp = true;
	bool useLocal = false;
	bool forceMining = false;
	bool jit = false;
	string clientName;

	// Init defaults
	Defaults::get();

	// Our address.
	KeyPair us = KeyPair::create();
	Address coinbase = us.address();

	string configFile = getDataDir() + "/config.rlp";
	bytes b = contents(configFile);
	if (b.size())
	{
		RLP config(b);
		us = KeyPair(config[0].toHash<Secret>());
		coinbase = config[1].toHash<Address>();
	}
	else
	{
		RLPStream config(2);
		config << us.secret() << coinbase;
		writeFile(configFile, config.out());
	}

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if ((arg == "-l" || arg == "--listen" || arg == "--listen-port") && i + 1 < argc)
			listenPort = (short)atoi(argv[++i]);
		else if ((arg == "-u" || arg == "--public-ip" || arg == "--public") && i + 1 < argc)
			publicIP = argv[++i];
		else if ((arg == "-r" || arg == "--remote") && i + 1 < argc)
			remoteHost = argv[++i];
		else if ((arg == "-p" || arg == "--port") && i + 1 < argc)
			remotePort = (short)atoi(argv[++i]);
		else if ((arg == "-n" || arg == "--upnp") && i + 1 < argc)
		{
			string m = argv[++i];
			if (isTrue(m))
				upnp = true;
			else if (isFalse(m))
				upnp = false;
			else
			{
				cerr << "Invalid -n/--upnp option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "-L" || arg == "--local-networking")
			useLocal = true;
		else if ((arg == "-c" || arg == "--client-name") && i + 1 < argc)
			clientName = argv[++i];
		else if ((arg == "-a" || arg == "--address" || arg == "--coinbase-address") && i + 1 < argc)
		{
			try
			{
				coinbase = h160(fromHex(argv[++i], ThrowType::Throw));
			}
			catch (BadHexCharacter& _e)
			{
				cwarn << "invalid hex character, coinbase rejected";
				cwarn << boost::diagnostic_information(_e);
				break;
			}
			catch (...)
			{
				cwarn << "coinbase rejected";
				break;
			}
		}
		else if ((arg == "-s" || arg == "--secret") && i + 1 < argc)
			us = KeyPair(h256(fromHex(argv[++i])));
		else if ((arg == "-d" || arg == "--path" || arg == "--db-path") && i + 1 < argc)
			dbPath = argv[++i];
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
					cerr << "Unknown -m/--mining option: " << m << endl;
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
			jsonrpc = jsonrpc == -1 ? 8080 : jsonrpc;
		else if (arg == "--json-rpc-port" && i + 1 < argc)
			jsonrpc = atoi(argv[++i]);
#endif
		else if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc)
			g_logVerbosity = atoi(argv[++i]);
		else if ((arg == "-x" || arg == "--peers") && i + 1 < argc)
			peers = atoi(argv[++i]);
		else if ((arg == "-t" || arg == "--miners") && i + 1 < argc)
			miners = atoi(argv[++i]);
		else if ((arg == "-o" || arg == "--mode") && i + 1 < argc)
		{
			string m = argv[++i];
			if (m == "full")
				mode = NodeMode::Full;
			else if (m == "peer")
				mode = NodeMode::PeerServer;
			else
			{
				cerr << "Unknown mode: " << m << endl;
				return -1;
			}
		}
		else if (arg == "--jit")
		{
#if ETH_EVMJIT
			jit = true;
#else
			cerr << "EVM JIT not enabled" << endl;
			return -1;
#endif
		}
		else if (arg == "-h" || arg == "--help")
			help();
		else if (arg == "-V" || arg == "--version")
			version();
		else
			remoteHost = argv[i];
	}

	if (!clientName.empty())
		clientName += "/";

	cout << credits();

	VMFactory::setKind(jit ? VMKind::JIT : VMKind::Interpreter);
	NetworkPreferences netPrefs(listenPort, publicIP, upnp, useLocal);
	auto nodesState = contents((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp");
	dev::WebThreeDirect web3(
		"Ethereum(++)/" + clientName + "v" + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) + (jit ? "/JIT" : ""),
		dbPath,
		false,
		mode == NodeMode::Full ? set<string>{"eth", "shh"} : set<string>(),
		netPrefs,
		&nodesState,
		miners
		);
	web3.setIdealPeerCount(peers);
	eth::Client* c = mode == NodeMode::Full ? web3.ethereum() : nullptr;

	if (c)
	{
		c->setForceMining(forceMining);
		c->setAddress(coinbase);
	}

	cout << "Address: " << endl << toHex(us.address().asArray()) << endl;
	web3.startNetwork();

	if (bootstrap)
		web3.connect(Host::pocHost());
	if (remoteHost.size())
		web3.connect(remoteHost, remotePort);

#if ETH_JSONRPC
	shared_ptr<WebThreeStubServer> jsonrpcServer;
	unique_ptr<jsonrpc::AbstractServerConnector> jsonrpcConnector;
	if (jsonrpc > -1)
	{
		jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonrpc));
		jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, vector<KeyPair>({us})));
		jsonrpcServer->setIdentities({us});
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
			if (cmd == "netstart")
			{
				iss >> netPrefs.listenPort;
				web3.setNetworkPreferences(netPrefs);
				web3.startNetwork();
			}
			else if (cmd == "connect")
			{
				string addr;
				unsigned port;
				iss >> addr >> port;
				web3.connect(addr, (short)port);
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
					jsonrpc = 8080;
				jsonrpcConnector = unique_ptr<jsonrpc::AbstractServerConnector>(new jsonrpc::HttpServer(jsonrpc));
				jsonrpcServer = shared_ptr<WebThreeStubServer>(new WebThreeStubServer(*jsonrpcConnector.get(), web3, vector<KeyPair>({us})));
				jsonrpcServer->setIdentities({us});
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
				cout << "Current address:" << endl
					 << toHex(us.address().asArray()) << endl;
			}
			else if (cmd == "secret")
			{
				cout << "Secret Key: " << toHex(us.secret().asArray()) << endl;
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
				cout << "Current balance: " << formatBalance( c->balanceAt(us.address())) << " = " <<c->balanceAt(us.address()) << " wei" << endl;
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
					u256 minGas = (u256)Client::txGas(data, 0);
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
							c->transact(secret, amount, dest, data, gas, gasPrice);
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
					cwarn << "Require parameters: transact ADDRESS AMOUNT GASPRICE GAS SECRET DATA";
			}
			else if (c && cmd == "listContracts")
			{
				auto acs =c->addresses();
				string ss;
				for (auto const& i: acs)
					if ( c->codeAt(i, 0).size())
					{
						ss = toString(i) + " : " + toString( c->balanceAt(i)) + " [" + toString((unsigned) c->countAt(i)) + "]";
						cout << ss << endl;
					}
			}
			else if (c && cmd == "listAccounts")
			{
				auto acs =c->addresses();
				string ss;
				for (auto const& i: acs)
					if ( c->codeAt(i, 0).empty())
					{
						ss = toString(i) + " : " + toString( c->balanceAt(i)) + " [" + toString((unsigned) c->countAt(i)) + "]";
						cout << ss << endl;
					}
			}
			else if (c && cmd == "send")
			{
				if (iss.peek() != -1)
				{
					string hexAddr;
					u256 amount;
					int size = hexAddr.length();

					iss >> hexAddr >> amount;
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
						u256 minGas = (u256)Client::txGas(bytes(), 0);
						try
						{
							Address dest = h160(fromHex(hexAddr, ThrowType::Throw));
							c->transact(us.secret(), amount, dest, bytes(), minGas);
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
							init = fromHex(sinit, ThrowType::Throw);
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
					u256 minGas = (u256)Client::txGas(init, 0);
					if (!init.size())
						cwarn << "Contract creation aborted, no init code.";
					else if (endowment < 0)
						cwarn << "Invalid endowment";
					else if (gas < minGas)
						cwarn << "Minimum gas amount is" << minGas;
					else
						c->transact(us.secret(), endowment, init, gas, gasPrice);
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
					bytes r = t.rlp();
					try
					{
						e.setup(&r);

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
								dev::eth::VM* vm = (VM*)vvm;
								dev::eth::ExtVM const* ext = static_cast<ExtVM const*>(vextVM);
								if (instr == Instruction::STOP || instr == Instruction::RETURN || instr == Instruction::SUICIDE)
									for (auto const& i: ext->state().storage(ext->myAddress))
										f << toHex(dev::toCompactBigEndian(i.first, 1)) << " " << toHex(dev::toCompactBigEndian(i.second, 1)) << endl;
								f << ext->myAddress << " " << hex << toHex(dev::toCompactBigEndian(vm->curPC(), 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)instr, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)vm->gas(), 1)) << endl;
							};
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
						auto storage =c->storageAt(h, 0);
						for (auto const& i: storage)
							s << "@" << showbase << hex << i.first << "    " << showbase << hex << i.second << endl;
						s << endl << disassemble( c->codeAt(h, 0)) << endl;

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
			else if (cmd == "setSecret")
			{
				if (iss.peek() != -1)
				{
					string hexSec;
					iss >> hexSec;
					us = KeyPair(h256(fromHex(hexSec)));
				} 
				else
					cwarn << "Require parameter: setSecret HEXSECRETKEY";
			}
			else if (cmd == "setAddress")
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
							coinbase = h160(fromHex(hexAddr, ThrowType::Throw));
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
			else if (cmd == "exportConfig")
			{
				if (iss.peek() != -1)
				{
					string path;
					iss >> path;
					RLPStream config(2);
					config << us.secret() << coinbase;
					writeFile(path, config.out());
				} 
				else
					cwarn << "Require parameter: exportConfig PATH";
			}
			else if (cmd == "importConfig")
			{
				if (iss.peek() != -1)
				{
					string path;
					iss >> path;
					bytes b = contents(path);
					if (b.size())
					{
						RLP config(b);
						us = KeyPair(config[0].toHash<Secret>());
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

	writeFile((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp", web3.saveNetwork());
	return 0;
}

