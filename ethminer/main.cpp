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

int main(int argc, char** argv)
{
	// Set env vars controlling GPU driver behavior.
	setenv("GPU_MAX_HEAP_SIZE", "100");
	setenv("GPU_MAX_ALLOC_PERCENT", "100");
	setenv("GPU_SINGLE_ALLOC_PERCENT", "100");

	if (getenv("SYSLOG"))
	{
		g_syslog = true;
		g_useColor = false;
	}
	if (getenv("NO_COLOR"))
		g_useColor = false;
#if defined(_WIN32)
	if (g_useColor)
	{
		g_useColor = false;
		// Set output mode to handle virtual terminal sequences
		// Only works on Windows 10, but most users should use it anyway
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

	MinerCLI m;

	m.ParseCommandLine(argc, argv);

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
