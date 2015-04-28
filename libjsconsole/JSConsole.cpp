//
// Created by Marek Kotewicz on 28/04/15.
//

#include <iostream>
#include <libdevcore/Log.h>
#include "JSConsole.h"

// TODO: readline!
#include <readline/readline.h>
#include <readline/history.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

int JSConsole::repl() const
{
	string cmd = "";
	g_logPost = [](std::string const& a, char const*) { cout << "\r           \r" << a << endl << flush; rl_forced_update_display(); };

	char* c = readline("> ");
	if (c && *c)
	{
		cmd = string(c);
		add_history(c);
		auto value = m_engine.eval(cmd.c_str());
		string result = m_printer.print(value);
		free(c);
		cout << result << endl;
	}
}
