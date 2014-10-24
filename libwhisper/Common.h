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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <string>
#include <chrono>
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libdevcore/RLP.h>
#include <libp2p/Capability.h>

namespace dev
{
namespace shh
{

/* this makes these symbols ambiguous on VS2013
using h256 = dev::h256;
using h256s = dev::h256s;
using bytes = dev::bytes;
using RLPStream = dev::RLPStream;
using RLP = dev::RLP;
using bytesRef = dev::bytesRef;
using bytesConstRef = dev::bytesConstRef;
using h256Set = dev::h256Set;
*/

class WhisperHost;
class WhisperPeer;
class Whisper;

class Envelope;

enum WhisperPacket
{
	StatusPacket = 0,
	MessagesPacket,
	AddFilterPacket,
	RemoveFilterPacket,
	PacketCount
};

using Topic = h256;

class BuildTopic
{
public:
	BuildTopic() {}
	template <class T> BuildTopic(T const& _t) { shift(_t); }

	template <class T> BuildTopic& shift(T const& _r) { return shiftBytes(RLPStream().append(_r).out()); }
	template <class T> BuildTopic& operator()(T const& _t) { return shift(_t); }

	BuildTopic& shiftRaw(h256 const& _part) { m_parts.push_back(_part); return *this; }

	operator Topic() const { return toTopic(); }
	Topic toTopic() const { Topic ret; for (auto i = 0; i < 32; ++i) ret[i] = m_parts[i * m_parts.size() / 32][i]; return ret; }

protected:
	BuildTopic& shiftBytes(bytes const& _b);

	h256s m_parts;
};

using TopicMask = std::pair<Topic, Topic>;
using TopicMasks = std::vector<TopicMask>;

class TopicFilter
{
public:
	TopicFilter() {}
	TopicFilter(TopicMask const& _m): m_topicMasks(1, _m) {}
	TopicFilter(TopicMasks const& _m): m_topicMasks(_m) {}
	TopicFilter(RLP const& _r): m_topicMasks((TopicMasks)_r) {}

	void fillStream(RLPStream& _s) const { _s << m_topicMasks; }
	h256 sha3() const;

	bool matches(Envelope const& _m) const;

private:
	TopicMasks m_topicMasks;
};

class BuildTopicMask: BuildTopic
{
public:
	enum EmptyType { Empty };

	BuildTopicMask() { shift(); }
	BuildTopicMask(EmptyType) {}
	template <class T> BuildTopicMask(T const& _t) { shift(_t); }

	template <class T> BuildTopicMask& shift(T const& _r) { BuildTopic::shift(_r); return *this; }
	BuildTopicMask& shiftRaw(h256 const& _h) { BuildTopic::shiftRaw(_h); return *this; }
	BuildTopic& shift() { m_parts.push_back(h256()); return *this; }

	BuildTopicMask& operator()() { shift(); return *this; }
	template <class T> BuildTopicMask& operator()(T const& _t) { shift(_t); return *this; }

	operator TopicMask() const { return toTopicMask(); }
	TopicMask toTopicMask() const;
};

}
}
