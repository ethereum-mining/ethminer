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
/** @file RLPXperimental.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */


#pragma once

#include <libdevcore/Guards.h>
#include "RLPXFrameCoder.h"
#include "RLPXPacket.h"
namespace ba = boost::asio;
namespace bi = boost::asio::ip;

namespace dev
{
namespace p2p
{

class RLPXFrameWriter
{
	struct QueueState
	{
		std::deque<RLPXPacket> q;
		RLPXPacket* writing = nullptr;
		size_t remaining = 0;
		bool sequenced = false;
		uint16_t sequence;
		mutable Mutex x;
	};
	
public:
	enum PacketPriority { PriorityLow = 0, PriorityHigh };

	RLPXFrameWriter(uint16_t _protocolType): m_protocolType(_protocolType) {}
	RLPXFrameWriter(RLPXFrameWriter const& _s): m_protocolType(_s.m_protocolType) {}
	
	size_t size() const { size_t l; size_t h; DEV_GUARDED(m_q.first.x) h = m_q.first.q.size(); DEV_GUARDED(m_q.second.x) l = m_q.second.q.size(); return l + h; }
	
	/// Thread-safe.
	void enque(unsigned _packetType, RLPStream& _payload, PacketPriority _priority = PriorityLow);
	
	/// Returns number of packets framed and outputs frames to o_bytes. Not thread-safe.
	size_t drain(RLPXFrameCoder& _coder, unsigned _size, std::vector<bytes>& o_toWrite);
	
private:
	uint16_t const m_protocolType;			// Protocol Type
	std::pair<QueueState, QueueState> m_q;		// High, Low frame queues
	uint16_t m_sequenceId = 0;				// Sequence ID
};

}
}