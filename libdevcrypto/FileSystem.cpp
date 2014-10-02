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
/** @file FileSystem.cpp
 * @authors
 *	 Eric Lombrozo <elombrozo@gmail.com>
 *	 Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "FileSystem.h"
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>

#ifdef _WIN32
#include <shlobj.h>
#endif
#include <boost/filesystem.hpp>
using namespace std;
using namespace dev;

std::string dev::getDataDir()
{
#ifdef _WIN32
	char path[1024] = "";
	if (SHGetSpecialFolderPathA(NULL, path, CSIDL_APPDATA, true))
		return (boost::filesystem::path(path) / "Ethereum").string();
	else
	{
	#ifndef _MSC_VER // todo?
		cwarn << "getDataDir(): SHGetSpecialFolderPathA() failed.";
	#endif
		throw std::runtime_error("getDataDir() - SHGetSpecialFolderPathA() failed.");
	}
#else
	boost::filesystem::path dataDirPath;
	char const* homeDir = getenv("HOME");
	if (!homeDir || strlen(homeDir) == 0)
		dataDirPath = boost::filesystem::path("/");
	else
		dataDirPath = boost::filesystem::path(homeDir);
	
#if defined(__APPLE__) && defined(__MACH__)
	// This eventually needs to be put in proper wrapper (to support sandboxing)
	return (dataDirPath / "Library/Application Support/Ethereum").string();
#else
	return (dataDirPath / ".ethereum").string();
#endif
#endif
}
