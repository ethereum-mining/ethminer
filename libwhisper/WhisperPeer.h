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
/** @file WhisperPeer.h
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

using p2p::Session;
using p2p::HostCapabilityFace;
using p2p::HostCapability;
using p2p::Capability;
using p2p::CapDesc;

class WhisperPeer: public Capability
{
	friend class WhisperHost;

public:
	WhisperPeer(std::shared_ptr<Session> _s, HostCapabilityFace* _h, unsigned _i, CapDesc const& _cap);
	virtual ~WhisperPeer();
	WhisperHost* host() const;
	static std::string name() { return "shh"; }
	static u256 version() { return WhisperProtocolVersion; }
	static unsigned messageCount() { return PacketCount; }
	TopicBloomFilterHash bloom() const { dev::Guard g(x_bloom); return m_bloom; }
	void sendTopicsOfInterest(TopicBloomFilterHash const& _bloom); ///< sends our bloom filter to remote peer
	void noteAdvertiseTopicsOfInterest() { dev::Guard g(x_advertiseTopicsOfInterest); m_advertiseTopicsOfInterest = true; }

private:
	virtual bool interpret(unsigned _id, RLP const&) override;
	void sendMessages();
	unsigned ratingForPeer(Envelope const& e) const;
	void noteNewMessage(h256 _h, Envelope const& _m);
	void setBloom(TopicBloomFilterHash const& _b) { dev::Guard g(x_bloom); m_bloom = _b; }

	mutable dev::Mutex x_unseen;
	std::multimap<unsigned, h256> m_unseen;	///< Rated according to what they want.
	std::chrono::system_clock::time_point m_timer = std::chrono::system_clock::now();

	mutable dev::Mutex x_bloom;
	TopicBloomFilterHash m_bloom; ///< Peer's topics of interest

	mutable dev::Mutex x_advertiseTopicsOfInterest;
	bool m_advertiseTopicsOfInterest;
};

}
}
