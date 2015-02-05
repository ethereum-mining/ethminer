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

#include "WhisperPeer.h"

#include <libdevcore/Log.h>
#include <libp2p/All.h>
#include "WhisperHost.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

#if defined(clogS)
#undef clogS
#endif
#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

WhisperPeer::WhisperPeer(Session* _s, HostCapabilityFace* _h, unsigned _i): Capability(_s, _h, _i)
{
	RLPStream s;
	sealAndSend(prep(s, StatusPacket, 1) << version());
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
		auto protocolVersion = _r[1].toInt<unsigned>();

		clogS(NetMessageSummary) << "Status: " << protocolVersion;

		if (protocolVersion != version())
			disable("Invalid protocol version.");

		for (auto const& m: host()->all())
			m_unseen.insert(make_pair(0, m.first));

		if (session()->id() < host()->host()->id())
			sendMessages();
		break;
	}
	case MessagesPacket:
	{
		unsigned n = 0;
		for (auto i: _r)
			if (n++)
				host()->inject(Envelope(i), this);
		break;
	}
	default:
		return false;
	}
	return true;
}

void WhisperPeer::sendMessages()
{
	RLPStream amalg;
	unsigned msgCount = 0;
	{
		Guard l(x_unseen);
		msgCount = m_unseen.size();
		while (m_unseen.size())
		{
			auto p = *m_unseen.begin();
			m_unseen.erase(m_unseen.begin());
			host()->streamMessage(p.second, amalg);
		}
	}
	
	if (msgCount)
	{
		RLPStream s;
		prep(s, MessagesPacket, msgCount).appendRaw(amalg.out(), msgCount);
		sealAndSend(s);
	}
}

void WhisperPeer::noteNewMessage(h256 _h, Envelope const& _m)
{
	Guard l(x_unseen);
	m_unseen.insert(make_pair(rating(_m), _h));
}
