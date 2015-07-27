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
 * @author Marek
 * @date 2014
 */

#include <string>
#include <libdevcore/FileSystem.h>
#include <libjsconsole/JSRemoteConsole.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

int main(int argc, char** argv)
{
	string remote = contentsString(getDataDir("web3") + "/session.url");
	if (remote.empty())
		remote = "http://localhost:8545";
	string sessionKey = contentsString(getDataDir("web3") + "/session.key");

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "--url" && i + 1 < argc)
			remote = argv[++i];
		else if (arg == "--session-key" && i + 1 < argc)
			sessionKey = argv[++i];
		else
		{
			cerr << "Invalid argument: " << arg << endl;
			exit(-1);
		}
	}

	JSRemoteConsole console(remote);

	if (!sessionKey.empty())
		console.eval("web3.admin.setSessionKey('" + sessionKey + "')");

	while (true)
		console.readExpression();

	return 0;
}
