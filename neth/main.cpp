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
#include <libethereum/All.h>
#if ETH_JSONRPC
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#endif
#include <libwebthree/WebThree.h>
#include "BuildInfo.h"

#undef KEY_EVENT // from windows.h
#include <ncurses.h>
#undef OK
#include <form.h>
#undef OK

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;
using namespace boost::algorithm;
using dev::eth::Instruction;

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
		<< "Usage neth [OPTIONS] <remote-host>" << endl
		<< "Options:" << endl
		<< "    -a,--address <addr>  Set the coinbase (mining payout) address to addr (default: auto)." << endl
		<< "    -c,--client-name <name>  Add a name to your client's version string (default: blank)." << endl
		<< "    -d,--db-path <path>  Load database from path (default:  ~/.ethereum " << endl
		<< "                         <APPDATA>/Etherum or Library/Application Support/Ethereum)." << endl
		<< "    -h,--help  Show this help message and exit." << endl
#if ETH_JSONRPC
		<< "    -j,--json-rpc  Enable JSON-RPC server (default: off)." << endl
		<< "    --json-rpc-port  Specify JSON-RPC server port (implies '-j', default: 8080)." << endl
#endif
		<< "    -l,--listen <port>  Listen on the given port for incoming connected (default: 30303)." << endl
		<< "    -m,--mining <on/off>  Enable mining (default: off)" << endl
		<< "    -n,--upnp <on/off>  Use upnp for NAT (default: on)." << endl
		<< "    -o,--mode <full/peer>  Start a full node or a peer node (Default: full)." << endl
		<< "    -p,--port <port>  Connect to remote port (default: 30303)." << endl
		<< "    -r,--remote <host>  Connect to remote host (default: none)." << endl
		<< "    -s,--secret <secretkeyhex>  Set the secret key for use with send command (default: auto)." << endl
		<< "    -t,--miners <number>  Number of mining threads to start (Default: " << thread::hardware_concurrency() << ")" << endl
		<< "    -u,--public-ip <ip>  Force public ip to given (default; auto)." << endl
		<< "    -v,--verbosity <0..9>  Set the log verbosity from 0 to 9 (tmp forced to 1)." << endl
		<< "    -x,--peers <number>  Attempt to connect to given number of peers (default: 5)." << endl
		<< "    -V,--version  Show the version and exit." << endl;
		exit(0);
}

void interactiveHelp()
{
	cout
		<< "Commands:" << endl
		<< "    netstart <port>  Starts the network sybsystem on a specific port." << endl
		<< "    netstop  Stops the network subsystem." << endl
#if ETH_JSONRPC
		<< "    jsonstart <port>  Starts the JSON-RPC server." << endl
		<< "    jsonstop  Stops the JSON-RPC server." << endl
#endif
		<< "    connect <addr> <port>  Connects to a specific peer." << endl
		<< "    minestart  Starts mining." << endl
		<< "    minestop  Stops mining." << endl
		<< "    address  Gives the current address." << endl
		<< "    secret  Gives the current secret" << endl
		<< "    block  Gives the current block height." << endl
		<< "    balance  Gives the current balance." << endl
		<< "    peers  List the peers that are connected" << endl
		<< "    transact  Execute a given transaction." << endl
		<< "    send  Execute a given transaction with current secret." << endl
		<< "    contract  Create a new contract with current secret." << endl
		<< "    inspect <contract> Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    reset  Resets ncurses windows" << endl
		<< "    exit  Exits the application." << endl;
}

string credits()
{
	std::ostringstream ccout;
	ccout
		<< "NEthereum (++) " << dev::Version << endl
		<< "  Code by Gav Wood & , (c) 2013, 2014." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;

	ccout << "Type 'netstart 30303' to start networking" << endl;
	ccout << "Type 'connect " << Host::pocHost() << " 30303' to connect" << endl;
	ccout << "Type 'exit' to quit" << endl << endl;
	return ccout.str();
}

void version()
{
	cout << "neth version " << dev::Version << endl;
	cout << "Network protocol version: " << dev::eth::c_protocolVersion << endl;
	cout << "Client database version: " << dev::eth::c_databaseVersion << endl;
	cout << "Build: " << DEV_QUOTED(ETH_BUILD_PLATFORM) << "/" << DEV_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
}

Address c_config = Address("661005d2720d855f1d9976f88bb10c1a3398c77f");
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

namespace nc
{

class nc_window_streambuf: public std::streambuf
{
public:
	nc_window_streambuf(WINDOW* p, std::ostream& os, unsigned long cursesAttr = 0);
	nc_window_streambuf(WINDOW* p, unsigned long _cursesAttr = 0);
	nc_window_streambuf(nc_window_streambuf const& _rhs);
	virtual ~nc_window_streambuf();

	nc_window_streambuf& operator=(nc_window_streambuf const& _rhs);

	virtual int overflow(int c);
	virtual int sync();

private:
	void copy(nc_window_streambuf const& _rhs);

	WINDOW* m_pnl;
	unsigned long m_flags;
	std::ostream* m_os;
	std::streambuf* m_old;
};

nc_window_streambuf::nc_window_streambuf(WINDOW * p, unsigned long _cursesAttr):
	m_pnl(p),
	m_flags(_cursesAttr),
	m_os(0),
	m_old(0)
{
	// Tell parent class that we want to call overflow() for each
	// input char:
	setp(0, 0);
	setg(0, 0, 0);
	scrollok(p, true);
	mvwinch(p, 0, 0);
}

nc_window_streambuf::nc_window_streambuf(WINDOW* _p, std::ostream& _os, unsigned long _cursesAttr):
	m_pnl(_p),
	m_flags(_cursesAttr),
	m_os(&_os),
	m_old(_os.rdbuf())
{
	setp(0, 0);
	setg(0, 0, 0);
	_os.rdbuf(this);
	scrollok(_p, true);
	mvwinch(_p, 0, 0);
}

void nc_window_streambuf::copy(nc_window_streambuf const& _rhs)
{
	if (this != &_rhs)
	{
		m_pnl = _rhs.m_pnl;
		m_flags = _rhs.m_flags;
		m_os = _rhs.m_os;
		m_old = _rhs.m_old;
	}
}

nc_window_streambuf::nc_window_streambuf(nc_window_streambuf const& _rhs):
	std::streambuf()
{
	copy(_rhs);
}

nc_window_streambuf& nc_window_streambuf::operator=(nc_window_streambuf const& _rhs)
{
	copy(_rhs);
	return *this;
}

nc_window_streambuf::~nc_window_streambuf()
{
	if (m_os)
		m_os->rdbuf(m_old);
}

int nc_window_streambuf::overflow(int c)
{
	int ret = c;
	if (c != EOF)
	{
		int x = 0;
		int y = 0;
		int mx = 0;
		int my = 0;
		getyx(m_pnl, y, x);
		getmaxyx(m_pnl, my, mx);
		if (y < 1)
			y = 1;
		if (x < 2)
			x = 2;
		if (x > mx - 4)
		{
			if (y + 1 >= my)
				scroll(m_pnl);
			else
				y++;
			x = 2;
		}
		if (m_flags)
		{
			wattron(m_pnl, m_flags);
			if (mvwaddch(m_pnl, y, x++, (chtype)c) == ERR)
				ret = EOF;
			wattroff(m_pnl, m_flags);
		}
		else if (mvwaddch(m_pnl, y, x++, (chtype)c) == ERR)
			ret = EOF;
	}
	if (c == EOF) // || std::isspace(c)
		if (sync() == EOF)
			ret = EOF;
	return ret;
}

int nc_window_streambuf::sync()
{
	if (stdscr && m_pnl)
		return (wrefresh(m_pnl) == ERR) ? EOF : 0;
	return EOF;
}

}

vector<string> form_dialog(vector<string> _sfields, vector<string> _lfields, vector<string> _bfields, int _cols, int _rows, string _post_form);

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
#if ETH_JSONRPC
	int jsonrpc = 8080;
#endif
	string publicIP;
	bool bootstrap = false;
	bool upnp = true;
	bool useLocal = false;
	bool forceMining = false;
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
		else if ((arg == "-t" || arg == "--miners") && i + 1 < argc)
			miners = atoi(argv[++i]);
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

	NetworkPreferences netPrefs(listenPort, publicIP, upnp, useLocal);
	auto nodesState = contents((dbPath.size() ? dbPath : getDataDir()) + "/network.rlp");
	dev::WebThreeDirect web3(
		"NEthereum(++)/" + clientName + "v" + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM),
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
	if (mining)
		c->startMining();

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

	std::ostringstream ccout;

	// Initialize ncurses
	char* str = new char[255];
	int width;
	int height;
	int y = 0;
	int x = 2;
	string cmd;
	WINDOW * mainwin, * consolewin, * logwin, * blockswin, * pendingwin, *addswin, * contractswin, * peerswin;

	if (!(mainwin = initscr()))
	{
		cerr << "Error initialising ncurses.";
		delete [] str;
		return -1;
	}

	getmaxyx(mainwin, height, width);
	int qheight = height * 3 / 5;
	int qwidth = width / 4 - 4;

	nonl();
	cbreak();
	timeout(30000);
	echo();
	keypad(mainwin, true);

	// Initialize color pairs
	start_color();
	init_pair(1, COLOR_WHITE, COLOR_BLACK);
	init_pair(2, COLOR_RED, COLOR_BLACK);
	init_pair(3, 7, COLOR_BLACK);
	use_default_colors();

	logwin = newwin(height * 2 / 5 - 2, width * 2 / 3, qheight, 0);
	nc::nc_window_streambuf outbuf(logwin, std::cout);

	consolewin   = newwin(qheight, width / 4, 0, 0);
	nc::nc_window_streambuf coutbuf(consolewin, ccout);
	blockswin    = newwin(qheight, width / 4, 0, width / 4);
	pendingwin   = newwin(height * 1 / 5, width / 4, 0, width * 2 / 4);
	peerswin     = newwin(height * 2 / 5, width / 4, height * 1 / 5, width * 2 / 4);
	addswin      = newwin(height * 2 / 5 - 2, width / 3, qheight, width * 2 / 3);
	contractswin = newwin(qheight, width / 4, 0, width * 3 / 4);

	int vl = qheight - 4;
	wsetscrreg(consolewin, 1, vl);
	wsetscrreg(blockswin, 1, vl);
	wsetscrreg(pendingwin, 1, vl);
	wsetscrreg(peerswin, 1, vl);
	wsetscrreg(addswin, 1, vl);
	wsetscrreg(contractswin, 1, vl);

	mvwprintw(mainwin, 1, 1, " > ");
	wresize(mainwin, 3, width);
	mvwin(mainwin, height - 3, 0);

	wmove(mainwin, 1, 4);

	while (true)
	{
		wclrtobot(consolewin);
		wclrtobot(pendingwin);
		wclrtobot(peerswin);
		wclrtobot(addswin);
		wclrtobot(contractswin);

		ccout << credits();

		// Prompt
		wmove(mainwin, 1, 4);
		getstr(str);

		string s(str);
		istringstream iss(s);
		iss >> cmd;

		// Address
		ccout << "Address:" << endl;
		ccout << toHex(us.address().asArray()) << endl << endl;

		mvwprintw(mainwin, 1, 1, " > ");
		clrtoeol();

		if (s.length() > 1)
		{
			ccout << "> ";
			ccout << str << endl;
		}

		if (cmd == "netstart")
		{
			unsigned port;
			iss >> port;
			web3.setNetworkPreferences(NetworkPreferences((short)port, publicIP, upnp));
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
			ccout << "Current address:" << endl;
			ccout << toHex(us.address().asArray()) << endl;
		}
		else if (cmd == "secret")
		{
			ccout << "Current secret:" << endl;
			ccout << toHex(us.secret().asArray()) << endl;
		}
		else if (cmd == "block")
		{
			unsigned n = c->blockChain().details().number;
			ccout << "Current block # ";
			ccout << toString(n) << endl;
		}
		else if (cmd == "peers")
		{
			for (auto it: web3.peers())
				cout << it.host << ":" << it.port << ", " << it.clientVersion << ", "
					<< std::chrono::duration_cast<std::chrono::milliseconds>(it.lastPing).count() << "ms"
					<< endl;
		}
		else if (cmd == "balance")
		{
			u256 balance = c->balanceAt(us.address());
			ccout << "Current balance:" << endl;
			ccout << toString(balance) << endl;
		}
		else if (cmd == "transact")
		{
			auto const& bc = c->blockChain();
			auto h = bc.currentHash();
			auto blockData = bc.block(h);
			BlockInfo info(blockData);
			vector<string> s;
			s.push_back("Address");
			vector<string> l;
			l.push_back("Amount");
			stringstream label;
			label << "Gas price";
			l.push_back(label.str());
			l.push_back("Gas");
			vector<string> b;
			b.push_back("Secret");
			b.push_back("Data");
			vector<string> fields = form_dialog(s, l, b, height, width, cmd);
			int fs = fields.size();
			if (fs < 6)
			{
				if (fs > 0)
					cwarn << "Missing parameter";
			}
			else
			{
				fields[0].erase(std::remove(fields[0].begin(), fields[0].end(), ' '), fields[0].end());
				fields[4].erase(std::remove(fields[4].begin(), fields[4].end(), ' '), fields[4].end());
				fields[5].erase(std::find_if(fields[5].rbegin(), fields[5].rend(), std::bind1st(std::not_equal_to<char>(), ' ')).base(), fields[5].end());
				int size = fields[0].length();
				u256 amount;
				u256 gasPrice;
				u256 gas;
				stringstream ssa;
				ssa << fields[1];
				ssa >> amount;
				stringstream ssg;
				ssg << fields[3];
				ssg >> gas;
				stringstream ssp;
				ssp << fields[2];
				ssp >> gasPrice;
				string sechex = fields[4];
				string sdata = fields[5];
				cnote << "Data:";
				cnote << sdata;
				bytes data = dev::eth::parseData(sdata);
				cnote << "Bytes:";
				string sbd = asString(data);
				bytes bbd = asBytes(sbd);
				stringstream ssbd;
				ssbd << bbd;
				cnote << ssbd.str();
				int ssize = fields[4].length();
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
						Secret secret = h256(fromHex(sechex, ThrowType::Throw));
						Address dest = h160(fromHex(fields[0], ThrowType::Throw));
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
		}
		else if (cmd == "send")
		{
			vector<string> s;
			s.push_back("Address");
			vector<string> l;
			l.push_back("Amount");
			vector<string> b;
			vector<string> fields = form_dialog(s, l, b, height, width, cmd);
			int fs = fields.size();
			if (fs < 2)
			{
				if (fs > 0)
					cwarn << "Missing parameter";
			}
			else
			{
				fields[0].erase(std::remove(fields[0].begin(), fields[0].end(), ' '), fields[0].end());
				int size = fields[0].length();
				u256 amount;
				stringstream sss;
				sss << fields[1];
				sss >> amount;
				if (size < 40)
				{
					if (size > 0)
						cwarn << "Invalid address length:" << size;
				}
				else
				{
					auto const& bc = c->blockChain();
					auto h = bc.currentHash();
					auto blockData = bc.block(h);
					BlockInfo info(blockData);
					u256 minGas = (u256)Client::txGas(bytes(), 0);
					try
					{
						Address dest = h160(fromHex(fields[0], ThrowType::Throw));
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
		}
		else if (cmd == "contract")
		{
			auto const& bc = c->blockChain();
			auto h = bc.currentHash();
			auto blockData = bc.block(h);
			BlockInfo info(blockData);
			vector<string> s;
			vector<string> l;
			l.push_back("Endowment");
			stringstream label;
			label << "Gas price";
			l.push_back(label.str());
			l.push_back("Gas");
			vector<string> b;
			b.push_back("Code (hex)");
			vector<string> fields = form_dialog(s, l, b, height, width, cmd);
			int fs = fields.size();
			if (fs < 4)
			{
				if (fs > 0)
					cwarn << "Missing parameter";
			}
			else
			{
				u256 endowment;
				u256 gas;
				u256 gasPrice;
				stringstream sse;
				sse << fields[0];
				sse >> endowment;
				stringstream ssg;
				ssg << fields[2];
				ssg >> gas;
				stringstream ssp;
				ssp << fields[1];
				ssp >> gasPrice;
				string sinit = fields[3];
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
				{
					c->transact(us.secret(), endowment, init, gas);
				}
			}
		}
		else if (cmd == "inspect")
		{
			string rechex;
			iss >> rechex;

			if (rechex.length() != 40)
				cwarn << "Invalid address length";
			else
			{
				auto address = h160(fromHex(rechex));
				stringstream s;

				try
				{
					auto storage = c->storageAt(address);
					for (auto const& i: storage)
						s << "@" << showbase << hex << i.first << "    " << showbase << hex << i.second << endl;
					s << endl << disassemble(c->codeAt(address)) << endl;

					string outFile = getDataDir() + "/" + rechex + ".evm";
					ofstream ofs;
					ofs.open(outFile, ofstream::binary);
					ofs.write(s.str().c_str(), s.str().length());
					ofs.close();

					cnote << "Saved" << rechex << "to" << outFile;
				}
				catch (dev::InvalidTrie const& _e)
				{
					cwarn << "Corrupted trie.\n" << diagnostic_information(_e);
				}
			}
		}
		else if (cmd == "reset")
		{
			vector<WINDOW*> ws { consolewin, blockswin, pendingwin, peerswin, contractswin, addswin, mainwin };
			for (auto &w: ws)
			{
				wclear(w);
				wrefresh(w);
			}
		}
		else if (cmd == "help")
			interactiveHelp();
		else if (cmd == "exit")
			break;

		// Clear cmd at each pass
		cmd = "";


		// Lock to prevent corrupt block-chain errors
		auto const& bc = c->blockChain();
		ccout << "Genesis hash: " << bc.genesisHash() << endl;

		// Blocks
		y = 1;
		for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
		{
			auto d = bc.details(h);
			string s = "# " + std::to_string(d.number) + ' ' +  toString(h); // .abridged();
			mvwaddnstr(blockswin, y++, x, s.c_str(), qwidth);

			auto b = bc.block(h);
			for (auto const& i: RLP(b)[1])
			{
				Transaction t(i.data(), CheckSignature::Sender);
				auto s = t.receiveAddress() ?
					boost::format("  %1% %2%> %3%: %4% [%5%]") %
						toString(t.safeSender()) %
						(c->codeAt(t.receiveAddress(), 0).size() ? '*' : '-') %
						toString(t.receiveAddress()) %
						toString(formatBalance(t.value())) %
						toString((unsigned)t.nonce()) :
					boost::format("  %1% +> %2%: %3% [%4%]") %
						toString(t.safeSender()) %
						toString(right160(sha3(rlpList(t.safeSender(), t.nonce())))) %
						toString(formatBalance(t.value())) %
						toString((unsigned)t.nonce());
				mvwaddnstr(blockswin, y++, x, s.str().c_str(), qwidth - 2);
				if (y > qheight - 2)
					break;
			}
			if (y > qheight - 2)
				break;
		}


		// Pending
		y = 1;
		for (Transaction const& t: c->pending())
		{
			auto s = t.receiveAddress() ?
				boost::format("%1% %2%> %3%: %4% [%5%]") %
					toString(t.safeSender()) %
					(c->codeAt(t.receiveAddress(), 0).size() ? '*' : '-') %
					toString(t.receiveAddress()) %
					toString(formatBalance(t.value())) %
					toString((unsigned)t.nonce()) :
				boost::format("%1% +> %2%: %3% [%4%]") %
					toString(t.safeSender()) %
					toString(right160(sha3(rlpList(t.safeSender(), t.nonce())))) %
					toString(formatBalance(t.value())) %
					toString((unsigned)t.nonce());
			mvwaddnstr(pendingwin, y++, x, s.str().c_str(), qwidth);
			if (y > height * 1 / 5 - 2)
				break;
		}


		// Contracts and addresses
		y = 1;
		int cc = 1;
		auto acs = c->addresses();
		for (auto const& i: acs)
			if (c->codeAt(i, 0).size())
			{
				auto s = boost::format("%1%%2% : %3% [%4%]") %
					toString(i) %
					pretty(i, c->postState()) %
					toString(formatBalance(c->balanceAt(i))) %
					toString((unsigned)c->countAt(i, 0));
				mvwaddnstr(contractswin, cc++, x, s.str().c_str(), qwidth);
				if (cc > qheight - 2)
					break;
			}
		for (auto const& i: acs)
			if (c->codeAt(i, 0).empty())
			{
				auto s = boost::format("%1%%2% : %3% [%4%]") %
					toString(i) %
					pretty(i, c->postState()) %
					toString(formatBalance(c->balanceAt(i))) %
					toString((unsigned)c->countAt(i, 0));
				mvwaddnstr(addswin, y++, x, s.str().c_str(), width / 2 - 4);
				if (y > height * 3 / 5 - 4)
					break;
			}

		// Peers
		y = 1;
		for (PeerSessionInfo const& i: web3.peers())
		{
			auto s = boost::format("%1% ms - %2%:%3% - %4%") %
				toString(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()) %
				i.host %
				toString(i.port) %
				i.clientVersion;
			mvwaddnstr(peerswin, y++, x, s.str().c_str(), qwidth);
			if (y > height * 2 / 5 - 4)
				break;
		}

		box(consolewin, 0, 0);
		box(blockswin, 0, 0);
		box(pendingwin, 0, 0);
		box(peerswin, 0, 0);
		box(addswin, 0, 0);
		box(contractswin, 0, 0);
		box(mainwin, 0, 0);

		// Balance
		stringstream ssb;
		u256 balance = c->balanceAt(us.address());
		ssb << "Balance: " << formatBalance(balance);
		mvwprintw(consolewin, 0, x, ssb.str().c_str());

		// Block
		mvwprintw(blockswin, 0, x, "Block # ");
		unsigned n = c->blockChain().details().number;
		mvwprintw(blockswin, 0, 10, toString(n).c_str());

		// Pending
		string pc;
		pc = "Pending: " + toString(c->pending().size());
		mvwprintw(pendingwin, 0, x, pc.c_str());

		// Contracts
		string sc = "Contracts: ";
		sc += toString(cc - 1);
		mvwprintw(contractswin, 0, x, sc.c_str());

		// Peers
		mvwprintw(peerswin, 0, x, "Peers: ");
		mvwprintw(peerswin, 0, 9, toString(web3.peers().size()).c_str());

		// Mining flag
		if (c->isMining())
		{
			mvwprintw(consolewin, qheight - 1, width / 4 - 11, "Mining ON");
			dev::eth::MineProgress p = c->miningProgress();
			auto speed = boost::format("%2% kH/s @ %1%s") % (p.ms / 1000) % (p.ms ? p.hashes / p.ms : 0);
			mvwprintw(consolewin, qheight - 2, width / 4 - speed.str().length() - 2, speed.str().c_str());
		}
		else
			mvwprintw(consolewin, qheight - 1, width / 4 - 12, "Mining OFF");

		wmove(consolewin, 1, x);

		// Addresses
		string ac;
		ac = "Addresses: " + toString(acs.size());
		mvwprintw(addswin, 0, x, ac.c_str());


		wrefresh(consolewin);
		wrefresh(blockswin);
		wrefresh(pendingwin);
		wrefresh(peerswin);
		wrefresh(addswin);
		wrefresh(contractswin);
		wrefresh(mainwin);
	}

	delwin(addswin);
	delwin(contractswin);
	delwin(peerswin);
	delwin(pendingwin);
	delwin(blockswin);
	delwin(consolewin);
	delwin(logwin);
	delwin(mainwin);
	endwin();
	refresh();

#if ETH_JSONRPC
	if (jsonrpcServer.get())
		jsonrpcServer->StopListening();
#endif

	return 0;
}

void print_in_middle(WINDOW *win, int starty, int startx, int width, string str, chtype color)
{
	int length;
	int x = 0;
	int y = 0;
	float temp;

	if (startx != 0)
		x = startx;
	if (starty != 0)
		y = starty;
	if (width == 0)
		width = 80;

	length = str.length();
	temp = (width - length) / 2;
	x = x + (int)temp;
	wattron(win, color);
	mvwprintw(win, y, x, "%s", str.c_str());
	wattroff(win, color);
	refresh();
}

vector<string> form_dialog(vector<string> _sv, vector<string> _lv, vector<string> _bv, int _cols, int _rows, string _post_form)
{
	vector<string> vs;
	WINDOW *form_win;
	int _sfields = _sv.size();
	int _lfields = _lv.size();
	int _bfields = _bv.size();
	int maxfields = _sfields + _lfields + _bfields;
	vector<FIELD*> field(maxfields + 1);

	int ch;
	int starty = 6;
	int height = _cols;
	int width = _rows;

	// Initialize the fields
	int si;
	int li;
	int bi = 0;
	vector<int> labels;
	for (si = 0; si < _sfields; ++si)
	{
		starty++; // Leave room for our labels, no window yet so that or fake fields...
		field[si] = new_field(1, 40, starty++, 1, 0, 0);
		labels.push_back(starty);
		set_field_back(field[si], A_UNDERLINE);
		set_field_type(field[si], TYPE_ALNUM, 40);
	}
	for (li = _sfields; li < _sfields + _lfields; ++li)
	{
		starty++;
		field[li] = new_field(1, 64, starty++, 1, 3, 0);
		labels.push_back(starty);
		set_field_back(field[li], A_UNDERLINE);
	}
	for (bi = _sfields + _lfields; bi < maxfields; ++bi)
	{
		starty++;
		field[bi] = new_field(5, 72, starty++, 1, 0, 0);
		labels.push_back(starty);
		field_opts_off(field[bi], O_STATIC);
		set_field_back(field[bi], A_UNDERLINE);
		starty += 4;
	}

	// The FORM expects a NULL terminated list of fields
	field[maxfields] = NULL;

	// Create the form and post it
	FORM *form = new_form(field.data());

	// Calculate the area required for the form
	scale_form(form, &_rows, &_cols);

	// Create the window to be associated with the form
	form_win = newwin(_rows + 4, _cols + 8, (height / 2 - _rows / 2 - 2), (width / 2 - _cols / 2 - 2));

	// Set main window and sub window
	set_form_win(form, form_win);
	set_form_sub(form, derwin(form_win, _rows, _cols, 2, 2));

	nodelay(form_win, true);
	keypad(form_win, true);
	noecho();
	timeout(0);

	box(form_win, 0, 0);
	print_in_middle(form_win, 1, 0, _cols, _post_form, COLOR_PAIR(2));

	post_form(form);

	// Set labels
	int ca = 0;
	int cf;
	for (cf = 0; cf < _sfields; ++cf)
	{
		wattron(form_win, COLOR_PAIR(3));
		mvwprintw(form_win, labels[ca], 3, _sv[cf].c_str());
		wattroff(form_win, COLOR_PAIR(3));
		ca++;
	}
	for (cf = 0; cf < _lfields; ++cf)
	{
		wattron(form_win, COLOR_PAIR(3));
		mvwprintw(form_win, labels[ca], 3, _lv[cf].c_str());
		mvwprintw(form_win, labels[ca] + 1, _cols - 1, "wei");
		wattroff(form_win, COLOR_PAIR(3));
		ca++;
	}
	for (cf = 0; cf < _bfields; ++cf)
	{
		wattron(form_win, COLOR_PAIR(3));
		mvwprintw(form_win, labels[ca], 3, _bv[cf].c_str());
		wattroff(form_win, COLOR_PAIR(3));
		ca++;
	}

	wrefresh(form_win);

	print_in_middle(form_win, 3, 0, _cols, string("Use the TAB key to switch between fields."), COLOR_PAIR(1));
	print_in_middle(form_win, 4, 0, _cols, string("Use UP, DOWN arrow keys to switch between lines."), COLOR_PAIR(1));
	print_in_middle(form_win, 6, 0, _cols, string("Press ENTER to submit the form and ESC to cancel."), COLOR_PAIR(1));
	refresh();

	while ((ch = wgetch(form_win)) != 27 && ch != 13) // KEY_F(1))
	{
		switch (ch)
		{
			case 9: // Tab
				form_driver(form, REQ_NEXT_FIELD);
				form_driver(form, REQ_END_LINE);
				break;
			case KEY_DOWN:
				form_driver(form, REQ_NEXT_LINE);
				break;
			case KEY_UP:
				form_driver(form, REQ_PREV_LINE);
				break;
			case KEY_LEFT:
				form_driver(form, REQ_LEFT_CHAR);
				break;
			case KEY_RIGHT:
				form_driver(form, REQ_RIGHT_CHAR);
				break;
			case KEY_BACKSPACE: // Backspace
			case KEY_DC:
			case KEY_DL:
			case 127:
				form_driver(form, REQ_DEL_PREV);
				wrefresh(form_win);
				break;
			case KEY_ENTER: // Enter
			case 13:
			case 27: // Esc
				break;
			default:
				form_driver(form, ch);
				break;
		}
	}

	if (form_driver(form, REQ_VALIDATION) != E_OK)
		cwarn << "Validation error";

	int fi;
	for (fi = 0; fi < maxfields; ++fi)
		free_field(field[fi]);
	free_form(form);
	unpost_form(form);
	echo();
	timeout(30000);
	delwin(form_win);

	if (ch == 13)
		for (int fi = 0; fi < maxfields; ++fi)
			vs.push_back(field_buffer(field[fi], 0));

	return vs;
}
