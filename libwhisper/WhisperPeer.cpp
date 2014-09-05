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
/** @file Whisper.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "WhisperPeer.h"

#include <libethential/Log.h>
#include <libp2p/All.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

WhisperPeer::WhisperPeer(Session* _s, HostCapabilityFace* _h): Capability(_s, _h)
{
	RLPStream s;
	prep(s);
	sealAndSend(s.appendList(2) << StatusPacket << host()->protocolVersion());
}

WhisperPeer::~WhisperPeer()
{
}

WhisperHost* WhisperPeer::host() const
{
	return static_cast<WhisperHost*>(Capability::hostCapability());
}

bool WhisperPeer::interpret(RLP const& _r)
{
	switch (_r[0].toInt<unsigned>())
	{
	case StatusPacket:
	{
		auto protocolVersion = _r[1].toInt<unsigned>();

		clogS(NetMessageSummary) << "Status: " << protocolVersion;

		if (protocolVersion != host()->protocolVersion())
			disable("Invalid protocol version.");

		if (session()->id() < host()->host()->id())
			sendMessages();
		break;
	}
	case MessagesPacket:
	{
		unsigned n = 0;
		for (auto i: _r)
			if (n++)
				host()->inject(Message(i), this);
		sendMessages();
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
	unsigned n = 0;

	Guard l(x_unseen);
	while (m_unseen.size())
	{
		auto p = *m_unseen.begin();
		m_unseen.erase(m_unseen.begin());
		host()->streamMessage(p.second, amalg);
		n++;
	}

	// pause before sending if no messages to send
	if (!n)
		this_thread::sleep_for(chrono::milliseconds(100));

	RLPStream s;
	prep(s);
	s.appendList(n + 1) << MessagesPacket;
	s.appendRaw(amalg.out(), n);
	sealAndSend(s);
}

void WhisperPeer::noteNewMessage(h256 _h, Message const& _m)
{
	Guard l(x_unseen);
	m_unseen[rating(_m)] = _h;
}

WhisperHost::WhisperHost()
{
}

WhisperHost::~WhisperHost()
{
}

void WhisperHost::streamMessage(h256 _m, RLPStream& _s) const
{
	UpgradableGuard l(x_messages);
	if (m_messages.count(_m))
	{
		UpgradeGuard ll(l);
		m_messages.at(_m).streamOut(_s);
	}
}

void WhisperHost::inject(Message const& _m, WhisperPeer* _p)
{
	auto h = _m.sha3();
	{
		UpgradableGuard l(x_messages);
		if (m_messages.count(h))
			return;
		UpgradeGuard ll(l);
		m_messages[h] = _m;
	}

	if (_p)
	{
		Guard l(m_filterLock);
		for (auto const& f: m_filters)
			if (f.second.filter.matches(_m))
				noteChanged(h, f.first);
	}

	for (auto& i: peers())
		if (i->cap<WhisperPeer>().get() == _p)
			i->addRating(1);
		else
			i->cap<WhisperPeer>()->noteNewMessage(h, _m);
}

void WhisperHost::noteChanged(h256 _messageHash, h256 _filter)
{
	for (auto& i: m_watches)
		if (i.second.id == _filter)
		{
			cwatch << "!!!" << i.first << i.second.id;
			i.second.changes.push_back(_messageHash);
		}
}

bool MessageFilter::matches(Message const& _m) const
{
	for (auto const& t: m_topicMasks)
	{
		if (t.first.size() != t.second.size() || _m.topic.size() < t.first.size())
			continue;
		for (unsigned i = 0; i < t.first.size(); ++i)
			if (((t.first[i] ^ _m.topic[i]) & t.second[i]) != 0)
				goto NEXT;
		return true;
		NEXT:;
	}
	return false;
}

unsigned WhisperHost::installWatch(h256 _h)
{
	auto ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
	m_watches[ret] = ClientWatch(_h);
	cwatch << "+++" << ret << _h;
	return ret;
}

unsigned WhisperHost::installWatch(shh::MessageFilter const& _f)
{
	Guard l(m_filterLock);

	h256 h = _f.sha3();

	if (!m_filters.count(h))
		m_filters.insert(make_pair(h, _f));

	return installWatch(h);
}

void WhisperHost::uninstallWatch(unsigned _i)
{
	cwatch << "XXX" << _i;

	Guard l(m_filterLock);

	auto it = m_watches.find(_i);
	if (it == m_watches.end())
		return;
	auto id = it->second.id;
	m_watches.erase(it);

	auto fit = m_filters.find(id);
	if (fit != m_filters.end())
		if (!--fit->second.refCount)
			m_filters.erase(fit);
}
