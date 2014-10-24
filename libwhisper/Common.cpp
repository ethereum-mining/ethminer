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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"

#include <libdevcrypto/SHA3.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

BuildTopic& BuildTopic::shiftBytes(bytes const& _b)
{
	m_parts.push_back(dev::sha3(_b));
	return *this;
}

h256 TopicFilter::sha3() const
{
	RLPStream s;
	fillStream(s);
	return dev::sha3(s.out());
}

TopicMask BuildTopicMask::toTopicMask() const
{
	TopicMask ret;
	if (m_parts.size())
		for (auto i = 0; i < 32; ++i)
		{
			ret.first[i] = m_parts[i * m_parts.size() / 32][i];
			ret.second[i] = m_parts[i * m_parts.size() / 32] ? 255 : 0;
		}
	return ret;
}
/*
web3.shh.watch({}, function(m) { env.note("New message:\n"+JSON.stringify(m)); })
k = web3.shh.newIdentity()
web3.shh.post({from: k, topic: web3.fromAscii("test"), payload: web3.fromAscii("Hello world!")})
*/
