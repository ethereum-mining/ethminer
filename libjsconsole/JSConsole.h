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
/** @file JSConsole.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#pragma once

#include <libdevcore/Log.h>
// TODO! make readline optional!
#include <readline/readline.h>
#include <readline/history.h>

namespace dev
{
namespace eth
{

template<typename Engine, typename Printer>
class JSConsole
{
public:
	JSConsole(): m_engine(Engine()), m_printer(Printer(m_engine)) {}
	~JSConsole() {}

	void readExpression() const
	{
		std::string cmd = "";
		g_logPost = [](std::string const& a, char const*) { std::cout << "\r           \r" << a << std::endl << std::flush; rl_forced_update_display(); };

		bool isEmpty = true;
		int openBrackets = 0;
		do {
			char* buff = readline(promptForIndentionLevel(openBrackets).c_str());
			isEmpty = !(buff && *buff);
			if (!isEmpty)
			{
				cmd += std::string(buff);
				cmd += " ";
				free(buff);
				int open = std::count(cmd.begin(), cmd.end(), '{');
				open += std::count(cmd.begin(), cmd.end(), '(');
				int closed = std::count(cmd.begin(), cmd.end(), '}');
				closed += std::count(cmd.begin(), cmd.end(), ')');
				openBrackets = open - closed;
			}
		} while (openBrackets > 0);

		if (!isEmpty)
		{
			add_history(cmd.c_str());
			auto value = m_engine.eval(cmd.c_str());
			std::string result = m_printer.prettyPrint(value).cstr();
			std::cout << result << std::endl;
		}
	}

protected:
	Engine m_engine;
	Printer m_printer;

	virtual std::string promptForIndentionLevel(int _i) const
	{
		if (_i == 0)
			return "> ";

		return std::string((_i + 1) * 2, ' ');
	}
};

}
}
