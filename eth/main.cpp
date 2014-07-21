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
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#if ETH_JSONRPC
#include <jsonrpc/connectors/httpserver.h>
#endif
#include <libethcore/FileSystem.h>
#include <libevmface/Instruction.h>
#include <libethereum/Defaults.h>
#include <libethereum/Client.h>
#include <libethereum/PeerNetwork.h>
#include <libethereum/BlockChain.h>
#include <libethereum/State.h>
#include <libethcore/CommonEth.h>
#if ETH_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif
#if ETH_JSONRPC
#include "EthStubServer.h"
#endif
#include "BuildInfo.h"
using namespace std;
using namespace eth;
using namespace boost::algorithm;
using eth::Instruction;
using eth::c_instructionInfo;

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
		<< "    address  Gives the current address." << endl
		<< "    secret  Gives the current secret" << endl
		<< "    block  Gives the current block height." << endl
		<< "    balance  Gives the current balance." << endl
		<< "    transact  Execute a given transaction." << endl
		<< "    send  Execute a given transaction with current secret." << endl
		<< "    contract  Create a new contract with current secret." << endl
		<< "    peers  List the peers that are connected" << endl
		<< "    listAccounts List the accounts on the network." << endl
		<< "    listContracts List the contracts on the network." << endl
		<< "    setSecret <secret> Set the secret to the hex secret key." <<endl
		<< "    setAddress <addr> Set the coinbase (mining payout) address." <<endl
		<< "    exportConfig <path> Export the config (.RLP) to the path provided." <<endl
		<< "    importConfig <path> Import the config (.RLP) from the path provided." <<endl
		<< "    inspect <contract> Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    exit  Exits the application." << endl;
}

void help()
{
	cout
        << "Usage eth [OPTIONS] <remote-host>" << endl
        << "Options:" << endl
        << "    -a,--address <addr>  Set the coinbase (mining payout) address to addr (default: auto)." << endl
        << "    -c,--client-name <name>  Add a name to your client's version string (default: blank)." << endl
        << "    -d,--db-path <path>  Load database from path (default:  ~/.ethereum " << endl
        << "                         <APPDATA>/Etherum or Library/Application Support/Ethereum)." << endl
        << "    -h,--help  Show this help message and exit." << endl
        << "    -i,--interactive  Enter interactive mode (default: non-interactive)." << endl
#if ETH_JSONRPC
		<< "    -j,--json-rpc  Enable JSON-RPC server (default: off)." << endl
		<< "    --json-rpc-port  Specify JSON-RPC server port (implies '-j', default: 8080)." << endl
#endif
        << "    -l,--listen <port>  Listen on the given port for incoming connected (default: 30303)." << endl
		<< "    -m,--mining <on/off/number>  Enable mining, optionally for a specified number of blocks (Default: off)" << endl
        << "    -n,--upnp <on/off>  Use upnp for NAT (default: on)." << endl
        << "    -o,--mode <full/peer>  Start a full node or a peer node (Default: full)." << endl
        << "    -p,--port <port>  Connect to remote port (default: 30303)." << endl
        << "    -r,--remote <host>  Connect to remote host (default: none)." << endl
        << "    -s,--secret <secretkeyhex>  Set the secret key for use with send command (default: auto)." << endl
        << "    -u,--public-ip <ip>  Force public ip to given (default; auto)." << endl
        << "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (Default: 8)." << endl
        << "    -x,--peers <number>  Attempt to connect to given number of peers (Default: 5)." << endl
        << "    -V,--version  Show the version and exit." << endl;
        exit(0);
}

string credits(bool _interactive = false)
{
	std::ostringstream cout;
	cout
        << "Ethereum (++) " << eth::EthVersion << endl
		<< "  Code by Gav Wood, (c) 2013, 2014." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;

	if (_interactive)
	{
        string vs = toString(eth::EthVersion);
		vs = vs.substr(vs.find_first_of('.') + 1)[0];
		int pocnumber = stoi(vs);
		string m_servers;
		if (pocnumber == 4)
			m_servers = "54.72.31.55";
		else
			m_servers = "54.72.69.180";

		cout << "Type 'netstart 30303' to start networking" << endl;
		cout << "Type 'connect " << m_servers << " 30303' to connect" << endl;
		cout << "Type 'exit' to quit" << endl << endl;
	}
	return cout.str();
}

void version()
{
    cout << "eth version " << eth::EthVersion << endl;
	cout << "Build: " << ETH_QUOTED(ETH_BUILD_PLATFORM) << "/" << ETH_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
}

Address c_config = Address("ccdeac59d35627b7de09332e819d5159e7bb7250");
string pretty(h160 _a, eth::State _st)
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
bytes parse_data(string _args);
int main(int argc, char** argv)
{
	unsigned short listenPort = 30303;
	string remoteHost;
	unsigned short remotePort = 30303;
	string dbPath;
	eth::uint mining = ~(eth::uint)0;
	NodeMode mode = NodeMode::Full;
	unsigned peers = 5;
	bool interactive = false;
#if ETH_JSONRPC
	int jsonrpc = 8080;
#endif
	string publicIP;
	bool upnp = true;
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
				cerr << "Invalid UPnP option: " << m << endl;
				return -1;
			}
		}
		else if ((arg == "-c" || arg == "--client-name") && i + 1 < argc)
			clientName = argv[++i];
		else if ((arg == "-a" || arg == "--address" || arg == "--coinbase-address") && i + 1 < argc)
			coinbase = h160(fromHex(argv[++i]));
		else if ((arg == "-s" || arg == "--secret") && i + 1 < argc)
			us = KeyPair(h256(fromHex(argv[++i])));
		else if ((arg == "-d" || arg == "--path" || arg == "--db-path") && i + 1 < argc)
			dbPath = argv[++i];
		else if ((arg == "-m" || arg == "--mining") && i + 1 < argc)
		{
			string m = argv[++i];
			if (isTrue(m))
				mining = ~(eth::uint)0;
			else if (isFalse(m))
				mining = 0;
			else if (int i = stoi(m))
				mining = i;
			else
			{
				cerr << "Unknown mining option: " << m << endl;
				return -1;
			}
		}
		else if (arg == "-i" || arg == "--interactive")
			interactive = true;
#if ETH_JSONRPC
		else if ((arg == "-j" || arg == "--json-rpc"))
			jsonrpc = jsonrpc ? jsonrpc : 8080;
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
				mode = NodeMode::Full;
			else if (m == "peer")
				mode = NodeMode::PeerServer;
			else
			{
				cerr << "Unknown mode: " << m << endl;
				return -1;
			}
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
    Client c("Ethereum(++)/" + clientName + "v" + eth::EthVersion + "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM), coinbase, dbPath);
	c.start();
	cout << credits();

	cout << "Address: " << endl << toHex(us.address().asArray()) << endl;
	c.startNetwork(listenPort, remoteHost, remotePort, mode, peers, publicIP, upnp);

#if ETH_JSONRPC
	auto_ptr<EthStubServer> jsonrpcServer;
	if (jsonrpc > -1)
	{
		jsonrpcServer = auto_ptr<EthStubServer>(new EthStubServer(new jsonrpc::HttpServer(jsonrpc), c));
		jsonrpcServer->setKeys({us});
		jsonrpcServer->StartListening();
	}
#endif

	if (interactive)
	{
		string l;
		while (true)
		{
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
				eth::uint port;
				iss >> port;
				ClientGuard g(&c);
				c.startNetwork((short)port);
			}
			else if (cmd == "connect")
			{
				string addr;
				eth::uint port;
				iss >> addr >> port;
				ClientGuard g(&c);
				c.connect(addr, (short)port);
			}
			else if (cmd == "netstop")
			{
				ClientGuard g(&c);
				c.stopNetwork();
			}
			else if (cmd == "minestart")
			{
				c.startMining();
			}
			else if (cmd == "minestop")
			{
				c.stopMining();
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
				jsonrpcServer = auto_ptr<EthStubServer>(new EthStubServer(new jsonrpc::HttpServer(jsonrpc), c));
				jsonrpcServer->setKeys({us});
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
				cout << "Current address:" << endl;
				const char* addchr = toHex(us.address().asArray()).c_str();
				cout << addchr << endl;
			}
			else if (cmd == "secret")
			{
				cout << "Secret Key: " << toHex(us.secret().asArray()) << endl;
			}
			else if (cmd == "block")
			{
				ClientGuard g(&c);
				cout << "Current block: " << c.blockChain().details().number;
			}
			else if (cmd == "peers")
			{
				ClientGuard g(&c);
				for (auto it: c.peers())
					cout << it.host << ":" << it.port << ", " << it.clientVersion << ", "
						<< std::chrono::duration_cast<std::chrono::milliseconds>(it.lastPing).count() << "ms"
						<< endl;
			}
			else if (cmd == "balance")
			{
				ClientGuard g(&c);
				cout << "Current balance: " << formatBalance(c.postState().balance(us.address())) << " = " << c.postState().balance(us.address()) << " wei" << endl;
			}
			else if (cmd == "transact")
			{
				ClientGuard g(&c);
				auto const& bc = c.blockChain();
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
					bytes data = parse_data(sdata);
					cnote << "Bytes:";
					string sbd = asString(data);
					bytes bbd = asBytes(sbd);
					stringstream ssbd;
					ssbd << bbd;
					cnote << ssbd.str();
					int ssize = sechex.length();
					int size = hexAddr.length();
					u256 minGas = (u256)c.state().callGas(data.size(), 0);
					if (size < 40)
					{
						if (size > 0)
							cwarn << "Invalid address length: " << size;
					}
					else if (gasPrice < info.minGasPrice)
						cwarn << "Minimum gas price is " << info.minGasPrice;
					else if (gas < minGas)
						cwarn << "Minimum gas amount is " << minGas;
					else if (ssize < 40)
					{
						if (ssize > 0)
							cwarn << "Invalid secret length:" << ssize;
					}
					else
					{
						Secret secret = h256(fromHex(sechex));
						Address dest = h160(fromHex(hexAddr));
						c.transact(secret, amount, dest, data, gas, gasPrice);
					}
				}
				else
					cwarn << "Require parameters: transact ADDRESS AMOUNT GASPRICE GAS SECRET DATA";
			}
			else if (cmd == "listContracts")
			{
				ClientGuard g(&c);
				auto const& st = c.state();
				auto acs = st.addresses();
				string ss;
				for (auto const& i: acs)
				{
					auto r = i.first;
					if (st.addressHasCode(r))
					{
						ss = toString(r) + " : " + toString(formatBalance(i.second)) + " [" + toString((unsigned)st.transactionsFrom(i.first)) + "]";
						cout << ss << endl;
					}
				}
			}
			else if (cmd == "listAccounts")
			{
				ClientGuard g(&c);
				auto const& st = c.state();
				auto acs = st.addresses();
				string ss;
				for (auto const& i: acs)
				{
					auto r = i.first;
					if (!st.addressHasCode(r))
					{
						ss = toString(r) + pretty(r, st) + " : " + toString(formatBalance(i.second)) + " [" + toString((unsigned)st.transactionsFrom(i.first)) + "]";
						cout << ss << endl;
					}
					
				}
			}
			else if (cmd == "send")
			{
				ClientGuard g(&c);
				if (iss.peek() != -1)
				{
					string hexAddr;
					u256 amount;
					int size = hexAddr.length();

					iss >> hexAddr >> amount;
					if (size < 40)
					{
						if (size > 0)
							cwarn << "Invalid address length: " << size;
					}
					else 
					{
						auto const& bc = c.blockChain();
						auto h = bc.currentHash();
						auto blockData = bc.block(h);
						BlockInfo info(blockData);
						u256 minGas = (u256)c.state().callGas(0, 0);
						Address dest = h160(fromHex(hexAddr));
						c.transact(us.secret(), amount, dest, bytes(), minGas, info.minGasPrice);
					}
					
				} 
				else
					cwarn << "Require parameters: send ADDRESS AMOUNT";
			}
			else if (cmd == "contract")
			{
				ClientGuard g(&c);
				auto const& bc = c.blockChain();
				auto h = bc.currentHash();
				auto blockData = bc.block(h);
				BlockInfo info(blockData);
				if(iss.peek() != -1) {
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
					cnote << "Code size: " << size;
					if (size < 1)
						cwarn << "No code submitted";
					else
					{
						cnote << "Assembled:";
						stringstream ssc;
						init = fromHex(sinit);
						ssc.str(string());
						ssc << disassemble(init);
						cnote << "Init:";
						cnote << ssc.str();
					}
					u256 minGas = (u256)c.state().createGas(init.size(), 0);
					if (endowment < 0)
						cwarn << "Invalid endowment";
					else if (gasPrice < info.minGasPrice)
						cwarn << "Minimum gas price is " << info.minGasPrice;
					else if (gas < minGas)
						cwarn << "Minimum gas amount is " << minGas;
					else
						c.transact(us.secret(), endowment, init, gas, gasPrice);
				} 
				else
					cwarn << "Require parameters: contract ENDOWMENT GASPRICE GAS CODEHEX";
			}
			else if (cmd == "inspect")
			{
				string rechex;
				iss >> rechex;

				if (rechex.length() != 40)
					cwarn << "Invalid address length";
				else
				{
					ClientGuard g(&c);
					auto h = h160(fromHex(rechex));
					stringstream s;
					auto mem = c.state().storage(h);

					for (auto const& i: mem)
						s << "@" << showbase << hex << i.first << "    " << showbase << hex << i.second << endl;
					s << endl << disassemble(c.state().code(h));

					string outFile = getDataDir() + "/" + rechex + ".evm";
					ofstream ofs;
					ofs.open(outFile, ofstream::binary);
					ofs.write(s.str().c_str(), s.str().length());
					ofs.close();
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
						coinbase = h160(fromHex(hexAddr));
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
	else
	{
		eth::uint n = c.blockChain().details().number;
		if (mining)
			c.startMining();
		while (true)
		{
			if (c.blockChain().details().number - n == mining)
				c.stopMining();
			this_thread::sleep_for(chrono::milliseconds(100));
		}
	}


	return 0;
}

bytes parse_data(string _args)
{
	bytes m_data;
	stringstream args(_args);
	string arg;
	int cc = 0;
	while (args >> arg)
	{
		int al = arg.length();
		if (boost::starts_with(arg, "0x"))
		{
			bytes bs = fromHex(arg);
			m_data += bs;
		}
		else if (arg[0] == '@')
		{
			arg = arg.substr(1, arg.length());
			if (boost::starts_with(arg, "0x"))
			{
				cnote << "hex: " << arg;
				bytes bs = fromHex(arg);
				int size = bs.size();
				if (size < 32)
					for (auto i = 0; i < 32 - size; ++i)
						m_data.push_back(0);
				m_data += bs;
			}
			else if (boost::starts_with(arg, "\"") && boost::ends_with(arg, "\""))
			{
				arg = arg.substr(1, arg.length() - 2);
				cnote << "string: " << arg;
				if (al < 32)
					for (int i = 0; i < 32 - al; ++i)
						m_data.push_back(0);
				for (int i = 0; i < al; ++i)
					m_data.push_back(arg[i]);
			}
			else
			{
				cnote << "value: " << arg;
				bytes bs = toBigEndian(u256(arg));
				int size = bs.size();
				if (size < 32)
					for (auto i = 0; i < 32 - size; ++i)
						m_data.push_back(0);
				m_data += bs;
			}
		}
		else
			for (int i = 0; i < al; ++i)
				m_data.push_back(arg[i]);
		cc++;
	}
	return m_data;
}
