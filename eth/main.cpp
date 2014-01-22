/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */

#include "Client.h"
#include "PeerNetwork.h"
#include "BlockChain.h"
#include "State.h"
using namespace std;
using namespace eth;

int main(int argc, char** argv)
{
	short listenPort = 30303;
	string remoteHost;
	short remotePort = 30303;
	bool interactive = false;
	string dbPath;

	// Our address.
	Address us;							// TODO: should be loaded from config file

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-l" && i + 1 < argc)
			listenPort = atoi(argv[++i]);
		else if (arg == "-r" && i + 1 < argc)
			remoteHost = argv[++i];
		else if (arg == "-p" && i + 1 < argc)
			remotePort = atoi(argv[++i]);
		else if (arg == "-a" && i + 1 < argc)
			us = h160(fromUserHex(argv[++i]));
		else if (arg == "-i")
			interactive = true;
		else if (arg == "-d")
			dbPath = arg;
		else
			remoteHost = argv[i];
	}

	Client c("Ethereum(++)/v0.1", us, dbPath);
	if (interactive)
	{
		cout << "Ethereum (++)" << endl;
		cout << "  By Gav Wood, Tim Hughes & team Ethereum, (c) 2013, 2014" << endl << endl;

		while (true)
		{
			cout << "> " << flush;
			std::string cmd;
			cin >> cmd;
			if (cmd == "netstart")
			{
				eth::uint port;
				cin >> port;
				c.startNetwork(port);
			}
			else if (cmd == "connect")
			{
				string addr;
				eth::uint port;
				cin >> addr >> port;
				c.connect(addr, port);
			}
			else if (cmd == "netstop")
			{
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
			else if (cmd == "transact")
			{
				string sechex;
				string rechex;
				u256 amount;
				u256 fee;
				cin >> sechex >> rechex >> amount >> fee;
				Secret secret = h256(fromUserHex(sechex));
				Address dest = h160(fromUserHex(rechex));
				c.transact(secret, dest, amount, fee);
			}
		}
	}
	else
	{
		c.startNetwork(listenPort, remoteHost, remotePort);
		c.startMining();
		while (true)
			sleep(1);
	}


	return 0;
}
