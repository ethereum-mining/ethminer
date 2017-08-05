/*
Enable colors in windows terminal which support ANSI escape codes.
*/
#ifdef _WIN32
#include <windows.h>
#include <iostream>
#include "Terminal.h"

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

namespace dev
{
	namespace con {

		bool IsANSISupported()
		{
			HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			DWORD consoleMode;
			if (GetConsoleMode(hConsole, &consoleMode))
			{
				if (consoleMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
					return true;
			}
			return false;
		}

		char* EthEscapeSequence(char* escapeSequence)
		{
			if (IsANSISupported())
				return escapeSequence;
			else
				return "";
		}
	}
}

#endif