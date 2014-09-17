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
/** @file Whisper.h
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

namespace dev
{
namespace shh
{

using p2p::Session;
using p2p::HostCapabilityFace;
using p2p::HostCapability;
using p2p::Capability;

struct Message
{
	unsigned expiry = 0;
	unsigned ttl = 0;
	bytes topic;
	bytes payload;

	Message() {}
	Message(unsigned _exp, unsigned _ttl, bytes const& _topic, bytes const& _payload): expiry(_exp), ttl(_ttl), topic(_topic), payload(_payload) {}
	Message(RLP const& _m)
	{
		expiry = _m[0].toInt<unsigned>();
		ttl = _m[1].toInt<unsigned>();
		topic = _m[2].toBytes();
		payload = _m[3].toBytes();
	}

	operator bool () const { return !!expiry; }

	void streamOut(RLPStream& _s) const { _s.appendList(4) << expiry << ttl << topic << payload; }
	h256 sha3() const { RLPStream s; streamOut(s); return dev::eth::sha3(s.out()); }
};

/**
 */
class WhisperPeer: public Capability
{
	friend class WhisperHost;

public:
	WhisperPeer(Session* _s, HostCapabilityFace* _h);
	virtual ~WhisperPeer();

	static std::string name() { return "shh"; }

	WhisperHost* host() const;

private:
	virtual bool interpret(RLP const&);

	void sendMessages();

	unsigned rating(Message const&) const { return 0; }	// TODO
	void noteNewMessage(h256 _h, Message const& _m);

	mutable dev::Mutex x_unseen;
	std::map<unsigned, h256> m_unseen;	///< Rated according to what they want.
};

class MessageFilter
{
public:
	MessageFilter() {}
	MessageFilter(std::vector<std::pair<bytes, bytes> > const& _m): m_topicMasks(_m) {}
	MessageFilter(RLP const& _r): m_topicMasks((std::vector<std::pair<bytes, bytes>>)_r) {}

	void fillStream(RLPStream& _s) const { _s << m_topicMasks; }
	h256 sha3() const { RLPStream s; fillStream(s); return dev::eth::sha3(s.out()); }

	bool matches(Message const& _m) const;

private:
	std::vector<std::pair<bytes, bytes> > m_topicMasks;
};

struct InstalledFilter
{
	InstalledFilter(MessageFilter const& _f): filter(_f) {}

	MessageFilter filter;
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

	virtual void inject(Message const& _m, WhisperPeer* _from = nullptr) = 0;

	virtual unsigned installWatch(MessageFilter const& _filter) = 0;
	virtual unsigned installWatch(h256 _filterId) = 0;
	virtual void uninstallWatch(unsigned _watchId) = 0;
	virtual h256s peekWatch(unsigned _watchId) const = 0;
	virtual h256s checkWatch(unsigned _watchId) = 0;

	virtual Message message(h256 _m) const = 0;

	virtual void sendRaw(bytes const& _payload, bytes const& _topic, unsigned _ttl) = 0;
};

class WhisperHost: public HostCapability<WhisperPeer>, public Interface
{
	friend class WhisperPeer;

public:
	WhisperHost();
	virtual ~WhisperHost();

	unsigned protocolVersion() const { return 0; }

	void inject(Message const& _m, WhisperPeer* _from = nullptr);

	unsigned installWatch(MessageFilter const& _filter);
	unsigned installWatch(h256 _filterId);
	void uninstallWatch(unsigned _watchId);
	h256s peekWatch(unsigned _watchId) const { dev::Guard l(m_filterLock); try { return m_watches.at(_watchId).changes; } catch (...) { return h256s(); } }
	h256s checkWatch(unsigned _watchId) { dev::Guard l(m_filterLock); h256s ret; try { ret = m_watches.at(_watchId).changes; m_watches.at(_watchId).changes.clear(); } catch (...) {} return ret; }

	Message message(h256 _m) const { try { dev::ReadGuard l(x_messages); return m_messages.at(_m); } catch (...) { return Message(); } }

	void sendRaw(bytes const& _payload, bytes const& _topic, unsigned _ttl) { inject(Message(time(0) + _ttl, _ttl, _topic, _payload)); }

private:
	void streamMessage(h256 _m, RLPStream& _s) const;

	void noteChanged(h256 _messageHash, h256 _filter);

	mutable dev::SharedMutex x_messages;
	std::map<h256, Message> m_messages;

	mutable dev::Mutex m_filterLock;
	std::map<h256, InstalledFilter> m_filters;
	std::map<unsigned, ClientWatch> m_watches;
};

struct WatshhChannel: public dev::LogChannel { static const char* name() { return "shh"; } static const int verbosity = 1; };
#define cwatshh dev::LogOutputStream<shh::WatshhChannel, true>()

class Watch;

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
	Watch(Whisper& _c, MessageFilter const& _tf): m_c(&_c), m_id(_c.installWatch(_tf)) {}
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
