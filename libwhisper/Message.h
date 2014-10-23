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
/** @file Message.h
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

class Message;

class Envelope
{
	friend class Message;

public:
	Envelope() {}
	Envelope(RLP const& _m)
	{
		m_expiry = _m[0].toInt<unsigned>();
		m_ttl = _m[1].toInt<unsigned>();
		m_topic = (Topic)_m[2];
		m_data = _m[3].toBytes();
		m_nonce = _m[4].toInt<u256>();
	}

	operator bool() const { return !!m_expiry; }

	void streamOut(RLPStream& _s, bool _withNonce) const { _s.appendList(_withNonce ? 5 : 4) << m_expiry << m_ttl << m_topic << m_data; if (_withNonce) _s << m_nonce; }
	h256 sha3() const { RLPStream s; streamOut(s, true); return dev::sha3(s.out()); }
	h256 sha3NoNonce() const { RLPStream s; streamOut(s, false); return dev::sha3(s.out()); }

	unsigned sent() const { return m_expiry - m_ttl; }
	unsigned expiry() const { return m_expiry; }
	unsigned ttl() const { return m_ttl; }
	Topic const& topic() const { return m_topic; }
	bytes const& data() const { return m_data; }

	Message open(Secret const& _s) const;
	Message open() const;

	unsigned workProved() const;
	void proveWork(unsigned _ms);

private:
	Envelope(unsigned _exp, unsigned _ttl, Topic const& _topic): m_expiry(_exp), m_ttl(_ttl), m_topic(_topic) {}

	unsigned m_expiry = 0;
	unsigned m_ttl = 0;
	u256 m_nonce;

	Topic m_topic;
	bytes m_data;
};

enum /*Message Flags*/
{
	ContainsSignature = 1
};

/// An (unencrypted) message, constructed from the combination of an Envelope, and, potentially,
/// a Secret key to decrypt the Message.
class Message
{
public:
	Message() {}
	Message(Envelope const& _e, Secret const& _s = Secret());
	Message(bytes const& _payload): m_payload(_payload) {}
	Message(bytesConstRef _payload): m_payload(_payload.toBytes()) {}
	Message(bytes&& _payload) { std::swap(_payload, m_payload); }

	Public from() const { return m_from; }
	Public to() const { return m_to; }
	bytes const& payload() const { return m_payload; }

	void setTo(Public _to) { m_to = _to; }

	operator bool() const { return !!m_payload.size() || m_from || m_to; }

	/// Turn this message into a ditributable Envelope.
	Envelope seal(Secret _from, Topic const& _topic, unsigned _workToProve = 50, unsigned _ttl = 50);
	// Overloads for skipping _from or specifying _to.
	Envelope seal(Topic const& _topic, unsigned _ttl = 50, unsigned _workToProve = 50) { return seal(Secret(), _topic, _workToProve, _ttl); }
	Envelope seal(Public _to, Topic const& _topic, unsigned _workToProve = 50, unsigned _ttl = 50) { m_to = _to; return seal(Secret(), _topic, _workToProve, _ttl); }
	Envelope seal(Secret _from, Public _to, Topic const& _topic, unsigned _workToProve = 50, unsigned _ttl = 50) { m_to = _to; return seal(_from, _topic, _workToProve, _ttl); }

private:
	void populate(bytes const& _data);

	Public m_from;
	Public m_to;
	bytes m_payload;
};

}
}
