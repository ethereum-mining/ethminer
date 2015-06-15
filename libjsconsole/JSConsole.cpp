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
/** @file JSConsole.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include <iostream>
#include <libdevcore/Log.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include "JSConsole.h"
#include "JSV8Connector.h"

// TODO! make readline optional!
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

JSConsole::JSConsole(WebThreeDirect& _web3, shared_ptr<AccountHolder> const& _accounts):
	m_engine(),
	m_printer(m_engine)
{
	m_jsonrpcConnector.reset(new JSV8Connector(m_engine));
	m_jsonrpcServer.reset(new WebThreeStubServer(*m_jsonrpcConnector.get(), _web3, _accounts, vector<KeyPair>()));
}

JSConsole::~JSConsole() {}

void JSConsole::repl() const
{
	string cmd = "";
	g_logPost = [](std::string const& a, char const*) { cout << "\r           \r" << a << endl << flush; rl_forced_update_display(); };

	bool isEmpty = true;
	int openBrackets = 0;
	do {
		char* buff = readline(promptForIndentionLevel(openBrackets).c_str());
		isEmpty = !(buff && *buff);
		if (!isEmpty)
		{
			cmd += string(buff);
			cmd += " ";
			free(buff);
			int open = count(cmd.begin(), cmd.end(), '{');
			open += count(cmd.begin(), cmd.end(), '(');
			int closed = count(cmd.begin(), cmd.end(), '}');
			closed += count(cmd.begin(), cmd.end(), ')');
			openBrackets = open - closed;
		}
	} while (openBrackets > 0);

	if (!isEmpty)
	{
		add_history(cmd.c_str());
		auto value = m_engine.eval(cmd.c_str());
		string result = m_printer.prettyPrint(value).cstr();
		cout << result << endl;
	}
}

std::string JSConsole::promptForIndentionLevel(int _i) const
{
	if (_i == 0)
		return "> ";

	return string((_i + 1) * 2, ' ');
}
