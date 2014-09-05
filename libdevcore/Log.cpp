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
/** @file Log.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Log.h"

#include <string>
#include <iostream>
using namespace std;
using namespace dev;

// Logging
int dev::g_logVerbosity = 5;
map<type_info const*, bool> dev::g_logOverride;

ThreadLocalLogName dev::t_logThreadName("main");

// foward declare without all of Windows.h
#ifdef _WIN32
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char* lpOutputString);
#endif

void dev::simpleDebugOut(std::string const& _s, char const*)
{
	cout << _s << endl << flush;

	// helpful to use OutputDebugString on windows
	#ifdef _WIN32
	{
		OutputDebugStringA(_s.data());
		OutputDebugStringA("\n");
	}
	#endif
}

std::function<void(std::string const&, char const*)> dev::g_logPost = simpleDebugOut;

