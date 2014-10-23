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
/** @file WhisperHost.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <mutex>
#include <array>
#include <set>
#include <memory>
#include <utility>
#include <libdevcore/RLP.h>
#include <libdevcore/Guards.h>
#include <libdevcrypto/SHA3.h>
#include "Common.h"
#include "Message.h"

namespace dev
{
namespace shh
{

/*struct TopicMask
{
	Topic data;
	Topic mask;
};*/

struct InstalledFilter
{
	InstalledFilter(TopicFilter const& _f): filter(_f) {}

	TopicFilter filter;
	unsigned refCount = 1;
};

struct ClientWatch
{
	ClientWatch() {}
	explicit ClientWatch(h256 _id): id(_id) {}

	h256 id;
	h256s changes;
};

class Interface
{
public:
	virtual ~Interface() {}

	virtual void inject(Envelope const& _m, WhisperPeer* _from = nullptr) = 0;

	unsigned installWatch(TopicMask const& _mask) { return installWatch(TopicFilter(_mask)); }
	virtual unsigned installWatch(TopicFilter const& _filter) = 0;
	virtual unsigned installWatchOnId(h256 _filterId) = 0;
	virtual void uninstallWatch(unsigned _watchId) = 0;
	virtual h256s peekWatch(unsigned _watchId) const = 0;
	virtual h256s checkWatch(unsigned _watchId) = 0;

	virtual Envelope envelope(h256 _m) const = 0;

	void sendRaw(bytes const& _payload, Topic _topic, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_topic, _ttl, _workToProve)); }
	void sendRaw(Public _to, bytes const& _payload, Topic _topic, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_to, _topic, _ttl, _workToProve)); }
	void sendRaw(Secret _from, bytes const& _payload, Topic _topic, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_from, _topic, _ttl, _workToProve)); }
	void sendRaw(Secret _from, Public _to, bytes const& _payload, Topic _topic, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_from, _to, _topic, _ttl, _workToProve)); }
};

struct WatshhChannel: public dev::LogChannel { static const char* name() { return "shh"; } static const int verbosity = 1; };
#define cwatshh dev::LogOutputStream<shh::WatshhChannel, true>()

}
}
/*
namespace std { void swap(shh::Watch& _a, shh::Watch& _b); }

namespace shh
{

class Watch: public boost::noncopyable
{
	friend void std::swap(Watch& _a, Watch& _b);

public:
	Watch() {}
	Watch(Whisper& _c, h256 _f): m_c(&_c), m_id(_c.installWatch(_f)) {}
	Watch(Whisper& _c, TopicFilter const& _tf): m_c(&_c), m_id(_c.installWatch(_tf)) {}
	~Watch() { if (m_c) m_c->uninstallWatch(m_id); }

	bool check() { return m_c ? m_c->checkWatch(m_id) : false; }
	bool peek() { return m_c ? m_c->peekWatch(m_id) : false; }

private:
	Whisper* m_c;
	unsigned m_id;
};

}

namespace shh
{

inline void swap(shh::Watch& _a, shh::Watch& _b)
{
	swap(_a.m_c, _b.m_c);
	swap(_a.m_id, _b.m_id);
}

}
*/
