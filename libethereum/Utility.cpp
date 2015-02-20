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
/** @file Utility.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Utility.h"

#include <boost/regex.hpp>
#include <libethcore/CommonEth.h>
#include <libdevcrypto/SHA3.h>
using namespace std;
using namespace dev;
using namespace dev::eth;

bytes dev::eth::parseData(string const& _args)
{
	bytes m_data;

	boost::smatch what;
	static const boost::regex r("(!|#|@|\\$)?\"([^\"]*)\"(\\s.*)?");
	static const boost::regex d("(@|\\$)?([0-9]+)(\\s*(ether)|(finney)|(szabo))?(\\s.*)?");
	static const boost::regex h("(@|\\$)?(0x)?(([a-fA-F0-9])+)(\\s.*)?");

	string s = _args;
	while (s.size())
		if (boost::regex_match(s, what, d))
		{
			u256 v((string)what[2]);
			if (what[6] == "szabo")
				v *= dev::eth::szabo;
			else if (what[5] == "finney")
				v *= dev::eth::finney;
			else if (what[4] == "ether")
				v *= dev::eth::ether;
			bytes bs = dev::toCompactBigEndian(v);
			if (what[1] != "$")
				for (auto i = bs.size(); i < 32; ++i)
					m_data.push_back(0);
			for (auto b: bs)
				m_data.push_back(b);
			s = what[7];
		}
		else if (boost::regex_match(s, what, h))
		{
			bytes bs = fromHex(((what[3].length() & 1) ? "0" : "") + what[3]);
			if (what[1] != "$")
				for (auto i = bs.size(); i < 32; ++i)
					m_data.push_back(0);
			for (auto b: bs)
				m_data.push_back(b);
			s = what[5];
		}
		else if (boost::regex_match(s, what, r))
		{
			bytes d = asBytes(what[2]);
			if (what[1] == "!")
				m_data += FixedHash<4>(sha3(d)).asBytes();
			else if (what[1] == "#")
				m_data += sha3(d).asBytes();
			else if (what[1] == "$")
				m_data += d + bytes{0};
			else
				m_data += d + bytes(32 - what[2].length() % 32, 0);
			s = what[3];
		}
		else
			s = s.substr(1);

	return m_data;
}
