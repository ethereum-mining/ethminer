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
/** @file CommonIO.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"

#include <fstream>
#include "Exceptions.h"
using namespace std;
using namespace eth;

string eth::memDump(bytes const& _b, unsigned _w, bool _html)
{
	stringstream ret;
	if (_html)
		ret << "<pre style=\"font-family: Monospace, sans-serif; font-size: small\">";
	for (unsigned i = 0; i < _b.size(); i += _w)
	{
		ret << hex << setw(4) << setfill('0') << i << " ";
		for (unsigned j = i; j < i + _w; ++j)
			if (j < _b.size())
				if (_b[j] >= 32 && _b[j] < 128)
					if ((char)_b[j] == '<' && _html)
						ret << "&lt;";
					else if ((char)_b[j] == '&' && _html)
						ret << "&amp;";
					else
						ret << (char)_b[j];
				else
					ret << '?';
			else
				ret << ' ';
		ret << " ";
		for (unsigned j = i; j < i + _w && j < _b.size(); ++j)
			ret << setfill('0') << setw(2) << hex << (unsigned)_b[j] << " ";
		ret << "\n";
	}
	if (_html)
		ret << "</pre>";
	return ret.str();
}

bytes eth::contents(std::string const& _file)
{
	std::ifstream is(_file, std::ifstream::binary);
	if (!is)
		return bytes();
	// get length of file:
	is.seekg (0, is.end);
	streamoff length = is.tellg();
	is.seekg (0, is.beg);
	bytes ret(length);
	is.read((char*)ret.data(), length);
	is.close();
	return ret;
}

void eth::writeFile(std::string const& _file, bytes const& _data)
{
	ofstream(_file, ios::trunc).write((char const*)_data.data(), _data.size());
}

