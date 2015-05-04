//
// Created by Marek Kotewicz on 28/04/15.
//

#include <iostream>
#include <libdevcore/Log.h>
#include "JSConsole.h"
#include "JSV8Connector.h"
#include "libjsconsole/JSConsoleResources.hpp"

// TODO: readline!
#include <readline/readline.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

JSConsole::JSConsole(WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts):
		m_engine(),
		m_printer(m_engine)
{
	m_jsonrpcConnector.reset(new JSV8Connector(m_engine));
	m_jsonrpcServer.reset(new WebThreeStubServer(*m_jsonrpcConnector.get(), _web3, _accounts));
}

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
		string result = m_printer.prettyPrint(value);
		cout << result << endl;
	}
}

std::string JSConsole::promptForIndentionLevel(int _i) const
{
	if (_i == 0)
		return "> ";

	return string((_i + 1) * 2, ' ');
}


