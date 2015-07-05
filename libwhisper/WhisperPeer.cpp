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
/** @file WhisperPeer.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libdevcore/Log.h>
#include <libp2p/All.h>
#include "WhisperHost.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

WhisperPeer::WhisperPeer(std::shared_ptr<Session> _s, HostCapabilityFace* _h, unsigned _i, CapDesc const&): Capability(_s, _h, _i)
{
	RLPStream s;
	sealAndSend(prep(s, StatusPacket, 1) << version());
	noteAdvertiseTopicsOfInterest();
}

WhisperPeer::~WhisperPeer()
{
}

WhisperHost* WhisperPeer::host() const
{
	return static_cast<WhisperHost*>(Capability::hostCapability());
}

bool WhisperPeer::interpret(unsigned _id, RLP const& _r)
{
	switch (_id)
	{
	case StatusPacket:
	{
		auto protocolVersion = _r[0].toInt<unsigned>();

		clog(NetMessageSummary) << "Status: " << protocolVersion;

		if (protocolVersion != version())
			disable("Invalid protocol version.");

		for (auto const& m: host()->all())
		{
			Guard l(x_unseen);
			m_unseen.insert(make_pair(0, m.first));
		}

		if (session()->id() < host()->host()->id())
			sendMessages();

		noteAdvertiseTopicsOfInterest();
		break;
	}
	case MessagesPacket:
	{
		for (auto i: _r)
			host()->inject(Envelope(i), this);
		break;
	}
	case TopicFilterPacket:
	{
		setBloom((TopicBloomFilterHash)_r[0]);
		break;
	}
	default:
		return false;
	}
	return true;
}

void WhisperPeer::sendMessages()
{
	if (m_advertiseTopicsOfInterest)
		sendTopicsOfInterest(host()->bloom());

	multimap<unsigned, h256> available;
	DEV_GUARDED(x_unseen)
		m_unseen.swap(available);

	RLPStream amalg;

	// send the highest rated messages first
	for (auto i = available.rbegin(); i != available.rend(); ++i)
		host()->streamMessage(i->second, amalg);

	unsigned msgCount = available.size();
	if (msgCount)
	{
		RLPStream s;
		prep(s, MessagesPacket, msgCount).appendRaw(amalg.out(), msgCount);
		sealAndSend(s);
	}
}

void WhisperPeer::noteNewMessage(h256 _h, Envelope const& _m)
{
	unsigned rate = ratingForPeer(_m);
	Guard l(x_unseen);
	m_unseen.insert(make_pair(rate, _h));
}

unsigned WhisperPeer::ratingForPeer(Envelope const& e) const
{
	// we try to estimate, how valuable this nessage will be for the remote peer,
	// according to the following criteria:
	// 1. bloom filter
	// 2. time to live
	// 3. proof of work

	unsigned rating = 0;

	if (e.matchesBloomFilter(bloom()))
		++rating;

	rating *= 256;
	unsigned ttlReward = (256 > e.ttl() ? 256 - e.ttl() : 0);
	rating += ttlReward;

	rating *= 256;
	rating += e.workProved();
	return rating;
}

void WhisperPeer::sendTopicsOfInterest(TopicBloomFilterHash const& _bloom)
{
	DEV_GUARDED(x_advertiseTopicsOfInterest)
		m_advertiseTopicsOfInterest = false;

	RLPStream s;
	prep(s, TopicFilterPacket, 1);
	s << _bloom;
	sealAndSend(s);
}

