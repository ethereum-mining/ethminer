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
#include <fstream>
#include <iostream>
#include "MinerAux.h"
#include <ethminer-buildinfo.h>

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace boost::algorithm;


void help()
{
	cout
		<< "Usage ethminer [OPTIONS]" << endl
		<< "Options:" << endl << endl;
	MinerCLI::streamHelp(cout);
	cout
		<< " General Options:" << endl
		<< "    -v,--verbosity <0 - 9>  Set the log verbosity from 0 to 9 (default: 5). Set to 9 for switch time logging." << endl
		<< "    -V,--version  Show the version and exit." << endl
		<< "    -h,--help  Show this help message and exit." << endl
		<< " Envionment variables:" << endl
		<< "     NO_COLOR - set to any value to disable color output. Unset to re-enable color output." << endl
		;
	exit(0);
}

void version()
{
    auto* bi = ethminer_get_buildinfo();
    cout << "ethminer version " << bi->project_version << "+git." << string(bi->git_commit_hash).substr(0, 8) << endl;
    cout << "Build: " << bi->system_name << "/" << bi->build_type << "/" << bi->compiler_id << endl;
    exit(0);
}

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	if (getenv("NO_COLOR"))
		g_useColor = false;
#if defined(_WIN32)
	if (g_useColor)
	{
		g_useColor = false;
		// Set output mode to handle virtual terminal sequences
		// Only works on Windows 10, but most user should use it anyways
		HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
		if (hOut != INVALID_HANDLE_VALUE)
		{
			DWORD dwMode = 0;
			if (GetConsoleMode(hOut, &dwMode))
			{
				dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
				if (SetConsoleMode(hOut, dwMode))
					g_useColor = true;
			}
		}
	}
#endif

	MinerCLI m(MinerCLI::OperationMode::Farm);

	try
	{
		for (int i = 1; i < argc; ++i)
		{
			// Mining options:
			if (m.interpretOption(i, argc, argv))
				continue;

			// Standard options:
			string arg = argv[i];
			if ((arg == "-v" || arg == "--verbosity") && i + 1 < argc)
				g_logVerbosity = atoi(argv[++i]);
			else if (arg == "-h" || arg == "--help")
				help();
			else if (arg == "-V" || arg == "--version")
				version();
			else
			{
				cerr << "Invalid argument: " << arg << endl;
				exit(-1);
			}
		}
	}
	catch (BadArgument ex)
	{
		std::cerr << "Error: " << ex.what() << "\n";
		exit(-1);
	}

	try
	{
		m.execute();
	}
	catch (std::exception& ex)
	{
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
