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

/**
 * @brief Multiplex packets into encrypted RLPX frames.
 * @todo throw when enqueued packet is invalid
 * @todo use RLPXFrameInfo
 */
class RLPXFrameWriter
{
	/**
	 * @brief Queue and state for Writer
	 * Properties are used independently;
	 * only valid packets should be added to q
	 * @todo implement as class
	 */
	struct WriterState
	{
		std::deque<RLPXPacket> q;
		mutable Mutex x;
		
		RLPXPacket* writing = nullptr;
		size_t remaining = 0;
		bool multiFrame = false;
		uint16_t sequence;
	};
	
public:
	enum PacketPriority { PriorityLow = 0, PriorityHigh };
	static const uint16_t EmptyFrameLength;
	static const uint16_t MinFrameDequeLength;

	RLPXFrameWriter(uint16_t _protocolType): m_protocolType(_protocolType) {}
	RLPXFrameWriter(RLPXFrameWriter const& _s): m_protocolType(_s.m_protocolType) {}
	
	/// Returns total number of queued packets. Thread-safe.
	size_t size() const { size_t l; size_t h; DEV_GUARDED(m_q.first.x) h = m_q.first.q.size(); DEV_GUARDED(m_q.second.x) l = m_q.second.q.size(); return l + h; }
	
	/// Moves @_payload output to queue, to be muxed into frames by mux() when network buffer is ready for writing. Thread-safe.
	void enque(uint8_t _packetType, RLPStream& _payload, PacketPriority _priority = PriorityLow);

	/// Returns number of packets framed and outputs frames to o_bytes. Not thread-safe.
	size_t mux(RLPXFrameCoder& _coder, unsigned _size, std::vector<bytes>& o_toWrite);
	
protected:
	/// Moves @_p to queue, to be muxed into frames by mux() when network buffer is ready for writing. Thread-safe.
	void enque(RLPXPacket&& _p, PacketPriority _priority = PriorityLow);
	
private:
	uint16_t const m_protocolType;			// Protocol Type
	std::pair<WriterState, WriterState> m_q;		// High, Low frame queues
	uint16_t m_sequenceId = 0;				// Sequence ID
};

}
}