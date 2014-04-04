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

#include <ncurses.h>
#undef OK
#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include "Defaults.h"
#include "Client.h"
#include "PeerNetwork.h"
#include "BlockChain.h"
#include "State.h"
#include "FileSystem.h"
#include "Instruction.h"
#include "BuildInfo.h"
using namespace std;
using namespace eth;
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

void interactiveHelp()
{
	cout
        << "Commands:" << endl
        << "    netstart <port>  Starts the network sybsystem on a specific port." << endl
        << "    netstop  Stops the network subsystem." << endl
        << "    connect <addr> <port>  Connects to a specific peer." << endl
        << "    minestart  Starts mining." << endl
        << "    minestop  Stops mining." << endl
        << "    address  Gives the current address." << endl
        << "    secret  Gives the current secret" << endl
        << "    block  Gives the current block height." << endl
        << "    balance  Gives the current balance." << endl
        << "    peers  List the peers that are connected" << endl
        << "    transact <secret> <dest> <amount> <gasPrice> <gas> <data>  Executes a given transaction." << endl
        << "    send <dest> <amount> <gasPrice> <gas>  Executes a given transaction with current secret." << endl
		<< "    inspect <contract> Dumps a contract to <APPDATA>/<contract>.evm." << endl
		<< "    exit  Exits the application." << endl;
}

string credits(bool _interactive = false)
{
	std::ostringstream ccout;
	ccout
		<< "Ethereum (++) " << ETH_QUOTED(ETH_VERSION) << endl
		<< "  Code by Gav Wood, (c) 2013, 2014." << endl
		<< "  Based on a design by Vitalik Buterin." << endl << endl;

	if (_interactive)
	{
		string vs = toString(ETH_QUOTED(ETH_VERSION));
		vs = vs.substr(vs.find_first_of('.') + 1)[0];
		int pocnumber = stoi(vs);
		string m_servers;
		if (pocnumber == 3)
			m_servers = "54.201.28.117";
		if (pocnumber == 4)
			m_servers = "54.72.31.55";

		ccout << "Type 'netstart 30303' to start networking" << endl;
		ccout << "Type 'connect " << m_servers << " 30303' to connect" << endl;
		ccout << "Type 'exit' to quit" << endl << endl;
	}
	return ccout.str();
}

void version()
{
	cout << "eth version " << ETH_QUOTED(ETH_VERSION) << endl;
	cout << "Build: " << ETH_QUOTED(ETH_BUILD_PLATFORM) << "/" << ETH_QUOTED(ETH_BUILD_TYPE) << endl;
	exit(0);
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
		(void)my;
		if (y < 1)
			y = 1;
		if (x < 2)
			x = 2;
		if (x > mx - 4)
		{
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


int main(int argc, char** argv)
{
	unsigned short listenPort = 30303;
	string remoteHost;
	unsigned short remotePort = 30303;
	bool interactive = false;
	string dbPath;
	eth::uint mining = ~(eth::uint)0;
	NodeMode mode = NodeMode::Full;
	unsigned peers = 5;
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
		else if (arg == "-i" || arg == "--interactive")
			interactive = true;
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
	Client c("Ethereum(++)/" + clientName + "v" ETH_QUOTED(ETH_VERSION) "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM), coinbase, dbPath);
	cout << credits();

	if (interactive)
	{
		std::ostringstream ccout;

		/*  Initialize ncurses  */
		const char* chr;
		char* str = new char[255];
		int width;
		int height;
		int y = 0;
		int x = 2;
		string cmd;
		WINDOW * mainwin, * consolewin, * logwin, * blockswin, * pendingwin, * contractswin, * peerswin;

		if (!(mainwin = initscr()))
		{
			cerr << "Error initialising ncurses.";
			return -1;
		}

		getmaxyx(mainwin, height, width);
		int qwidth = width / 4 - 4;

		nonl();
		nocbreak();
		timeout(30000);
		echo();
		keypad(mainwin, true);

		logwin = newwin(height * 2 / 5 - 2, width, height * 3 / 5, 0);
		nc::nc_window_streambuf outbuf(logwin, std::cout);
		// nc::nc_window_streambuf errbuf( logwin, std::cerr );
		g_logVerbosity = 1; // Force verbosity level for now

		consolewin   = newwin(height * 3 / 5, width / 4, 0, 0);
		nc::nc_window_streambuf coutbuf(consolewin, ccout);
		blockswin    = newwin(height * 3 / 5, width / 4, 0, width / 4);
		pendingwin   = newwin(height * 1 / 5, width / 4, 0, width * 2 / 4);
		peerswin     = newwin(height * 2 / 5, width / 4, height * 1 / 5, width * 2 / 4);
		contractswin = newwin(height * 3 / 5, width / 4, 0, width * 3 / 4);

		int vl = height * 3 / 5 - 4;
		wsetscrreg(consolewin, 1, vl);
		wsetscrreg(blockswin, 1, vl);
		wsetscrreg(pendingwin, 1, vl);
		wsetscrreg(peerswin, 1, vl);
		wsetscrreg(contractswin, 1, vl);

		mvwprintw(mainwin, 1, x, "> ");
		wresize(mainwin, 3, width);
		mvwin(mainwin, height - 3, 0);

		wmove(mainwin, 1, 4);

		if (!remoteHost.empty())
			c.startNetwork(listenPort, remoteHost, remotePort, mode, peers, publicIP, upnp);

		while (true)
		{
			wclrtobot(consolewin);
			wclrtobot(pendingwin);
			wclrtobot(peerswin);
			wclrtobot(contractswin);

			ccout << credits(true);

			// Prompt
			wmove(mainwin, 1, 4);
			getstr(str);

			string s(str);
			istringstream iss(s);
			iss >> cmd;

			// Address
			ccout << "Address:" << endl;
			chr = toHex(us.address().asArray()).c_str();
			ccout << chr << endl << endl;

			mvwprintw(mainwin, 1, x, "> ");
			clrtoeol();

			if (s.length() > 1)
			{
				ccout << "> ";
				ccout << str << endl;
			}

			if (cmd == "netstart")
			{
				eth::uint port;
				iss >> port;
				c.startNetwork((short)port);
			}
			else if (cmd == "connect")
			{
				string addr;
				eth::uint port;
				iss >> addr >> port;
				c.connect(addr, (short)port);
			}
			else if (cmd == "netstop")
				c.stopNetwork();
			else if (cmd == "minestart")
				c.startMining();
			else if (cmd == "minestop")
				c.stopMining();
			else if (cmd == "address")
			{
				ccout << "Current address:" << endl;
				const char* addchr = toHex(us.address().asArray()).c_str();
				ccout << addchr << endl;
			}
			else if (cmd == "secret")
			{
				ccout << "Current secret:" << endl;
				const char* addchr = toHex(us.secret().asArray()).c_str();
				ccout << addchr << endl;
			}
			else if (cmd == "block")
			{
				eth::uint n = c.blockChain().details().number;
				ccout << "Current block # ";
				const char* addchr = toString(n).c_str();
				ccout << addchr << endl;
			}
			else if (cmd == "peers")
			{
				for (auto it: c.peers())
					cout << it.host << ":" << it.port << ", " << it.clientVersion << ", "
						<< std::chrono::duration_cast<std::chrono::milliseconds>(it.lastPing).count() << "ms"
						<< endl;
			}
			else if (cmd == "balance")
			{
				u256 balance = c.state().balance(us.address());
				ccout << "Current balance:" << endl;
				const char* addchr = toString(balance).c_str();
				ccout << addchr << endl;
			}	
			else if (cmd == "transact")
			{
				string sechex;
				string rechex;
				u256 amount;
				u256 gasPrice;
				u256 gas;
				iss >> sechex >> rechex >> amount >> gasPrice >> gas;
				Secret secret = h256(fromHex(sechex));
				Address dest = h160(fromHex(rechex));
				bytes data;

				c.transact(secret, amount, dest, data, gas, gasPrice);
			}
			else if (cmd == "send")
			{
				string rechex;
				u256 amount;
				u256 gasPrice;
				u256 gas;
				iss >> rechex >> amount >> gasPrice >> gas;
				Address dest = h160(fromHex(rechex));

				c.transact(us.secret(), amount, dest, bytes(), gas, gasPrice);
			}
			else if (cmd == "inspect")
			{
				string rechex;
				iss >> rechex;

				if (rechex.length() != 40)
					cout << "Invalid address length" << endl;
				else
				{
					c.lock();
					auto h = h160(fromHex(rechex));

					stringstream s;
					auto mem = c.state().contractStorage(h);
					u256 next = 0;
					unsigned numerics = 0;
					bool unexpectedNumeric = false;
					for (auto const& i: mem)
					{
						if (next < i.first)
						{
							unsigned j;
							for (j = 0; j <= numerics && next + j < i.first; ++j)
								s << (j < numerics || unexpectedNumeric ? " 0" : " STOP");
							unexpectedNumeric = false;
							numerics -= min(numerics, j);
							if (next + j < i.first)
								s << "\n@" << showbase << hex << i.first << "    ";
						}
						else if (!next)
							s << "@" << showbase << hex << i.first << "    ";
						auto iit = c_instructionInfo.find((Instruction)(unsigned)i.second);
						if (numerics || iit == c_instructionInfo.end() || (u256)(unsigned)iit->first != i.second)	// not an instruction or expecting an argument...
						{
							if (numerics)
								numerics--;
							else
								unexpectedNumeric = true;
							s << " " << showbase << hex << i.second;
						}
						else
						{
							auto const& ii = iit->second;
							s << " " << ii.name;
							numerics = ii.additional;
						}
						next = i.first + 1;
					}

					string outFile = getDataDir() + "/" + rechex + ".evm";
					ofstream ofs;
					ofs.open(outFile, ofstream::binary);
					ofs.write(s.str().c_str(), s.str().length());
					ofs.close();

					c.unlock();
				}
			}
			else if (cmd == "help")
				interactiveHelp();
			else if (cmd == "exit")
				break;

			// Clear cmd at each pass
			cmd = "";


			// Blocks
			auto const& st = c.state();
			auto const& bc = c.blockChain();
			y = 0;
			for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
			{
				auto d = bc.details(h);
				string s = "# " + std::to_string(d.number) + ' ' +  toString(h); // .abridged();
				y += 1;
				mvwaddnstr(blockswin, y, x, s.c_str(), qwidth);

				for (auto const& i: RLP(bc.block(h))[1])
				{
					Transaction t(i.data());
					string ss;
					ss = t.receiveAddress ?
						"  " + toString(toHex(t.safeSender().asArray())) + " " + (st.isContractAddress(t.receiveAddress) ? '*' : '-') + "> " + toString(t.receiveAddress) + ": " + toString(formatBalance(t.value)) + " [" + toString((unsigned)t.nonce) + "]":
						"  " + toString(toHex(t.safeSender().asArray())) + " +> " + toString(right160(t.sha3())) + ": " + toString(formatBalance(t.value)) + " [" + toString((unsigned)t.nonce) + "]";
					y += 1;
					mvwaddnstr(blockswin, y, x, ss.c_str(), qwidth - 2);
					if (y > height * 3 / 5 - 2)
						break;
				}
				if (y > height * 3 / 5 - 2)
					break;
			}


			// Pending
			y = 0;
			for (Transaction const& t: c.pending())
			{
				string ss;
				if (t.receiveAddress)
					ss = toString(toHex(t.safeSender().asArray())) + " " + (st.isContractAddress(t.receiveAddress) ? '*' : '-') + "> " + toString(t.receiveAddress) + ": " + toString(formatBalance(t.value)) + " " + " [" + toString((unsigned)t.nonce) + "]";
				else
					ss = toString(toHex(t.safeSender().asArray())) + " +> " + toString(right160(t.sha3())) + ": " + toString(formatBalance(t.value)) + "[" + toString((unsigned)t.nonce) + "]";
				y += 1;
				mvwaddnstr(pendingwin, y, x, ss.c_str(), qwidth);
				if (y > height * 3 / 5 - 4)
					break;
			}


			// Contracts
			auto acs = st.addresses();
			y = 0;
			for (auto n = 0; n < 2; ++n)
				for (auto const& i: acs)
				{
					auto r = i.first;

					if (st.isContractAddress(r))
					{
						string ss;
						ss = toString(r) + " : " + toString(formatBalance(i.second)) + " [" + toString((unsigned)st.transactionsFrom(i.first)) + "]";
						y += 1;
						mvwaddnstr(contractswin, y, x, ss.c_str(), qwidth);
						if (y > height * 3 / 5 - 2)
							break;
					}
				}

			// Peers
			y = 0;
			string psc;
			string pss;
			auto cp = c.peers();
			psc = toString(cp.size()) + " peer(s)";
			for (PeerInfo const& i: cp)
			{
				pss = toString(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()) + " ms - " + i.host + ":" + toString(i.port) + " - " + i.clientVersion;
				y += 1;
				mvwaddnstr(peerswin, y, x, pss.c_str(), qwidth);
				if (y > height * 2 / 5 - 4)
					break;
			}

			box(consolewin, 0, 0);
			box(blockswin, 0, 0);
			box(pendingwin, 0, 0);
			box(peerswin, 0, 0);
			box(contractswin, 0, 0);
			box(mainwin, 0, 0);

			// Balance
			mvwprintw(consolewin, 0, x, "Balance: ");
			u256 balance = c.state().balance(us.address());
			chr = toString(balance).c_str();
			mvwprintw(consolewin, 0, 11, chr);
			wmove(consolewin, 1, x);

			// Block
			mvwprintw(blockswin, 0, x, "Block # ");
			eth::uint n = c.blockChain().details().number;
			chr = toString(n).c_str();
			mvwprintw(blockswin, 0, 10, chr);

			mvwprintw(pendingwin, 0, x, "Pending");
			mvwprintw(contractswin, 0, x, "Contracts");

			// Peers
			mvwprintw(peerswin, 0, x, "Peers: ");
			chr = toString(c.peers().size()).c_str();
			mvwprintw(peerswin, 0, 9, chr);

			wrefresh(consolewin);
			wrefresh(blockswin);
			wrefresh(pendingwin);
			wrefresh(peerswin);
			wrefresh(contractswin);
			wrefresh(mainwin);
		}

		delwin(contractswin);
		delwin(peerswin);
		delwin(pendingwin);
		delwin(blockswin);
		delwin(consolewin);
		delwin(logwin);
		delwin(mainwin);
		endwin();
		refresh();
	}
	else
	{
		cout << "Address: " << endl << toHex(us.address().asArray()) << endl;
		c.startNetwork(listenPort, remoteHost, remotePort, mode, peers, publicIP, upnp);
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
