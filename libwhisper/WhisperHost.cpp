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
/** @file WhisperHost.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "WhisperHost.h"

#include <libdevcore/Log.h>
#include <libp2p/All.h>
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

#if defined(clogS)
#undef clogS
#endif
#define clogS(X) dev::LogOutputStream<X, true>(false) << "| " << std::setw(2) << session()->socketId() << "] "

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
		m_messages.at(_m).streamOut(_s, true);
	}
}

void WhisperHost::inject(Envelope const& _m, WhisperPeer* _p)
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
			cwatshh << "!!!" << i.first << i.second.id;
			i.second.changes.push_back(_messageHash);
		}
}

unsigned WhisperHost::installWatch(h256 _h)
{
	auto ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
	m_watches[ret] = ClientWatch(_h);
	cwatshh << "+++" << ret << _h;
	return ret;
}

unsigned WhisperHost::installWatch(shh::TopicFilter const& _f)
{
	Guard l(m_filterLock);

	h256 h = _f.sha3();

	if (!m_filters.count(h))
		m_filters.insert(make_pair(h, _f));

	return installWatch(h);
}

void WhisperHost::uninstallWatch(unsigned _i)
{
	cwatshh << "XXX" << _i;

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
