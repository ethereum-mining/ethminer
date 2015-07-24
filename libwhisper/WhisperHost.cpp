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
#include <libdevcore/CommonIO.h>
#include <libdevcore/Log.h>
#include <libp2p/All.h>
#include "WhisperDB.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

WhisperHost::WhisperHost(bool _storeMessagesInDB): Worker("shh"), m_storeMessagesInDB(_storeMessagesInDB)
{
	loadMessagesFromBD();
}

WhisperHost::~WhisperHost()
{
	saveMessagesToBD();
}

void WhisperHost::streamMessage(h256 _m, RLPStream& _s) const
{
	UpgradableGuard l(x_messages);
	if (m_messages.count(_m))
	{
		UpgradeGuard ll(l);
		auto const& m = m_messages.at(_m);
		cnote << "streamRLP: " << m.expiry() << m.ttl() << m.topic() << toHex(m.data());
		m.streamRLP(_s);
	}
}

void WhisperHost::inject(Envelope const& _m, WhisperPeer* _p)
{
	// this function processes both outgoing messages originated both by local host (_p == null)
	// and incoming messages from remote peers (_p != null)

	cnote << this << ": inject: " << _m.expiry() << _m.ttl() << _m.topic() << toHex(_m.data());

	if (_m.isExpired())
		return;

	auto h = _m.sha3();
	{
		UpgradableGuard l(x_messages);
		if (m_messages.count(h))
			return;
		UpgradeGuard ll(l);
		m_messages[h] = _m;
		m_expiryQueue.insert(make_pair(_m.expiry(), h));
	}

	// rating of incoming message from remote host is assessed according to the following criteria:
	// 1. installed watch match; 2. bloom filter match; 2. ttl; 3. proof of work

	int rating = 0;

	DEV_GUARDED(m_filterLock)
		if (_m.matchesBloomFilter(m_bloom))
		{
			++rating;
			for (auto const& f: m_filters)
				if (f.second.filter.matches(_m))
					for (auto& i: m_watches)
						if (i.second.id == f.first) // match one of the watches
						{
							i.second.changes.push_back(h);
							rating += 2;
						}
		}

	if (_p) // incoming message from remote peer
	{
		rating *= 256;
		unsigned ttlReward = (256 > _m.ttl() ? 256 - _m.ttl() : 0);
		rating += ttlReward;
		rating *= 256;
		rating += _m.workProved();
	}

	// TODO p2p: capability-based rating
	for (auto i: peerSessions())
	{
		auto w = i.first->cap<WhisperPeer>().get();
		if (w == _p)
			w->addRating(rating);
		else
			w->noteNewMessage(h, _m);
	}
}

unsigned WhisperHost::installWatch(shh::Topics const& _t)
{
	InstalledFilter f(_t);
	h256 h = f.filter.sha3();
	unsigned ret = 0;

	DEV_GUARDED(m_filterLock)
	{
		auto it = m_filters.find(h);
		if (m_filters.end() == it)
			m_filters.insert(make_pair(h, f));
		else
			it->second.refCount++;

		m_bloom.addRaw(f.filter.exportBloom());
		ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
		m_watches[ret] = ClientWatch(h);
		cwatshh << "+++" << ret << h;
	}

	noteAdvertiseTopicsOfInterest();
	return ret;
}

void WhisperHost::uninstallWatch(unsigned _i)
{
	cwatshh << "XXX" << _i;

	DEV_GUARDED(m_filterLock)
	{
		auto it = m_watches.find(_i);
		if (it == m_watches.end())
			return;

		auto id = it->second.id;
		m_watches.erase(it);

		auto fit = m_filters.find(id);
		if (fit != m_filters.end())
		{
			m_bloom.removeRaw(fit->second.filter.exportBloom());
			if (!--fit->second.refCount)
				m_filters.erase(fit);
		}
	}

	noteAdvertiseTopicsOfInterest();
}

h256s WhisperHost::watchMessages(unsigned _watchId)
{
	h256s ret;
	auto wit = m_watches.find(_watchId);
	if (wit == m_watches.end())
		return ret;
	TopicFilter f;
	{
		Guard l(m_filterLock);
		auto fit = m_filters.find(wit->second.id);
		if (fit == m_filters.end())
			return ret;
		f = fit->second.filter;
	}
	ReadGuard l(x_messages);
	for (auto const& m: m_messages)
		if (f.matches(m.second))
			ret.push_back(m.first);
	return ret;
}

h256s WhisperHost::checkWatch(unsigned _watchId)
{
	h256s ret;
	cleanup();

	dev::Guard l(m_filterLock);
	try
	{
		ret = m_watches.at(_watchId).changes;
		m_watches.at(_watchId).changes.clear();
	}
	catch (...)
	{
	}

	return ret;
}

void WhisperHost::doWork()
{
	for (auto i: peerSessions())
		i.first->cap<WhisperPeer>().get()->sendMessages();
	cleanup();
}

void WhisperHost::cleanup()
{
	// remove old messages.
	// should be called every now and again.
	unsigned now = (unsigned)time(0);
	WriteGuard l(x_messages);
	for (auto it = m_expiryQueue.begin(); it != m_expiryQueue.end() && it->first <= now; it = m_expiryQueue.erase(it))
		m_messages.erase(it->second);
}

void WhisperHost::noteAdvertiseTopicsOfInterest()
{
	for (auto i: peerSessions())
		i.first->cap<WhisperPeer>().get()->noteAdvertiseTopicsOfInterest();
}

bool WhisperHost::isWatched(Envelope const& _e) const
{
	DEV_GUARDED(m_filterLock)
		if (_e.matchesBloomFilter(m_bloom))
			for (auto const& f: m_filters)
				if (f.second.filter.matches(_e))
					for (auto const& i: m_watches)
						if (i.second.id == f.first)
							return true;
	return false;
}

void WhisperHost::saveMessagesToBD()
{
	if (!m_storeMessagesInDB)
		return;

	try
	{
		WhisperMessagesDB db;
		ReadGuard g(x_messages);
		unsigned now = (unsigned)time(0);
		for (auto const& m: m_messages)
			if (m.second.expiry() > now)
				if (isWatched(m.second))
					db.saveSingleMessage(m.first, m.second);
	}
	catch(FailedToOpenLevelDB const& ex)
	{
		cwarn << "Exception in WhisperHost::saveMessagesToBD() - failed to open DB:" << ex.what();
	}
	catch(Exception const& ex)
	{
		cwarn << "Exception in WhisperHost::saveMessagesToBD():" << ex.what();
	}
	catch(...)
	{
		cwarn << "Unknown Exception in WhisperHost::saveMessagesToBD()";
	}
}

void WhisperHost::loadMessagesFromBD()
{
	if (!m_storeMessagesInDB)
		return;

	try
	{
		map<h256, Envelope> m;
		WhisperMessagesDB db;
		db.loadAllMessages(m);
		WriteGuard g(x_messages);
		m_messages.swap(m);
		for (auto const& msg: m)
			m_expiryQueue.insert(make_pair(msg.second.expiry(), msg.first));
	}
	catch(Exception const& ex)
	{
		cwarn << "Exception in WhisperHost::loadMessagesFromBD():" << ex.what();
	}
	catch(...)
	{
		cwarn << "Unknown Exception in WhisperHost::loadMessagesFromBD()";
	}
}

void WhisperHost::exportFilters(RLPStream& o_dst) const
{
	DEV_GUARDED(m_filterLock)
	{
		o_dst.appendList(m_filters.size());

		for (auto const& x: m_filters)
		{
			Topics const& topics = x.second.full;
			unsigned const RawDataSize = topics.size() * h256::size;
			unique_ptr<byte> p(new byte[RawDataSize]);
			unsigned i = 0;

			for (auto const& t: topics)
				memcpy(p.get() + h256::size * i++, t.data(), h256::size);
			
			bytesConstRef ref(p.get(), RawDataSize);
			o_dst.append(ref);
		}		
	}
}
