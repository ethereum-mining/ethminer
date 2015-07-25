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
/** @file Interface.h
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
#include <libdevcore/SHA3.h>
#include "Common.h"
#include "Message.h"

namespace dev
{
namespace shh
{

class Watch;

struct InstalledFilter
{
	InstalledFilter(Topics const& _t): full(_t), filter(_t) {}

	Topics full;
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
	virtual ~Interface();

	virtual void inject(Envelope const& _m, WhisperPeer* _from = nullptr) = 0;

	virtual Topics const& fullTopics(unsigned _id) const = 0;
	virtual unsigned installWatch(Topics const& _filter) = 0;
	virtual void uninstallWatch(unsigned _watchId) = 0;
	virtual h256s peekWatch(unsigned _watchId) const = 0;
	virtual h256s checkWatch(unsigned _watchId) = 0;
	virtual h256s watchMessages(unsigned _watchId) = 0;

	virtual Envelope envelope(h256 _m) const = 0;

	void post(bytes const& _payload, Topics _topics, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_topics, _ttl, _workToProve)); }
	void post(Public _to, bytes const& _payload, Topics _topics, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).sealTo(_to, _topics, _ttl, _workToProve)); }
	void post(Secret const& _from, bytes const& _payload, Topics _topics, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).seal(_from, _topics, _ttl, _workToProve)); }
	void post(Secret const& _from, Public _to, bytes const& _payload, Topics _topics, unsigned _ttl = 50, unsigned _workToProve = 50) { inject(Message(_payload).sealTo(_from, _to, _topics, _ttl, _workToProve)); }
};

struct WatshhChannel: public dev::LogChannel { static const char* name() { return "shh"; } static const int verbosity = 1; };
#define cwatshh dev::LogOutputStream<shh::WatshhChannel, true>()

}
}

namespace std { void swap(dev::shh::Watch& _a, dev::shh::Watch& _b); }

namespace dev
{
namespace shh
{

class Watch: public boost::noncopyable
{
	friend void std::swap(Watch& _a, Watch& _b);

public:
	Watch() {}
	Watch(Interface& _c, Topics const& _t): m_c(&_c), m_id(_c.installWatch(_t)) {}
	~Watch() { if (m_c) m_c->uninstallWatch(m_id); }

	h256s check() { return m_c ? m_c->checkWatch(m_id) : h256s(); }
	h256s peek() { return m_c ? m_c->peekWatch(m_id) : h256s(); }

private:
	Interface* m_c;
	unsigned m_id;
};

}
}

namespace std
{

inline void swap(dev::shh::Watch& _a, dev::shh::Watch& _b)
{
	swap(_a.m_c, _b.m_c);
	swap(_a.m_id, _b.m_id);
}

}

