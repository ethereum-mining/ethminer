/*
        This file is part of cpp-ethereum.

        cpp-ethereum is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        Foobar is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file FileSystem.cpp
 * @author Eric Lombrozo <elombrozo@gmail.com>
 * @date 2014
 */

#include "FileSystem.h"
#include <boost/filesystem.hpp>

#ifdef _WIN32
#include <shlobj.h>
#endif

std::string getDataDir()
{
#ifdef _WIN32
	char path[1024] = "";
	if (SHGetSpecialFolderPathA(NULL, path, CSIDL_APPDATA, true))
		return (boost::filesystem::path(path) / "Ethereum").string();
	else
        {
		std::cerr << "getDataDir() - SHGetSpecialFolderPathA() failed." << std::endl;
		throw std::runtime_error("getDataDir() - SHGetSpecialFolderPathA() failed.");
	}
#else
	boost::filesystem::path dataDirPath;
	char* homeDir = getenv("HOME");
	if (homeDir == NULL || strlen(homeDir) == 0)
		dataDirPath = boost::filesystem::path("/");
	else
		dataDirPath = boost::filesystem::path(homeDir);
#if defined(__APPLE__) && defined(__MACH__)
	dataDirPath /= "Library/Application Support";
	boost::filesystem::create_directory(dataDirPath);
        return (dataDirPath / "Ethereum").string();
#else
	return (dataDirPath / ".ethereum").string();
#endif
#endif
}
