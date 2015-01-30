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
#include "Message.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

CollapsedTopicPart dev::shh::collapse(FullTopicPart const& _p)
{
	return CollapsedTopicPart(sha3(_p));
}

CollapsedTopic dev::shh::collapse(FullTopic const& _fullTopic)
{
	CollapsedTopic ret;
	ret.reserve(_fullTopic.size());
	for (auto const& ft: _fullTopic)
		ret.push_back(collapse(ft));
	return ret;
}

CollapsedTopic BuildTopic::toTopic() const
{
	CollapsedTopic ret;
	ret.reserve(m_parts.size());
	for (auto const& h: m_parts)
		ret.push_back(collapse(h));
	return ret;
}

BuildTopic& BuildTopic::shiftBytes(bytes const& _b)
{
	m_parts.push_back(dev::sha3(_b));
	return *this;
}

h256 TopicFilter::sha3() const
{
	RLPStream s;
	streamRLP(s);
	return dev::sha3(s.out());
}

bool TopicFilter::matches(Envelope const& _e) const
{
	for (TopicMask const& t: m_topicMasks)
	{
		for (unsigned i = 0; i < t.size(); ++i)
		{
			for (auto et: _e.topic())
				if (((t[i].first ^ et) & t[i].second) == 0)
					goto NEXT_TOPICPART;
			// failed to match topicmask against any topics: move on to next mask
			goto NEXT_TOPICMASK;
			NEXT_TOPICPART:;
		}
		// all topicmasks matched.
		return true;
		NEXT_TOPICMASK:;
	}
	return false;
}

TopicMask BuildTopicMask::toTopicMask() const
{
	TopicMask ret;
	ret.reserve(m_parts.size());
	for (auto const& h: m_parts)
		ret.push_back(make_pair(collapse(h), ~CollapsedTopicPart()));
	return ret;
}

/*
web3.shh.watch({}).arrived(function(m) { env.note("New message:\n"+JSON.stringify(m)); })
k = web3.shh.newIdentity()
web3.shh.post({from: k, topic: web3.fromAscii("test"), payload: web3.fromAscii("Hello world!")})
*/

