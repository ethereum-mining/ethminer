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

struct RLPXFrameDecrytFailed: virtual dev::Exception {};
	
/**
 * RLPFrameReader
 * Reads and assembles RLPX frame byte buffers into RLPX packets. Additionally
 * buffers incomplete packets which are pieced into multiple frames (has sequence).
 * @todo drop frame and reset incomplete if
 * @todo percolate sequenceid to p2p protocol
 * @todo informative exception
 */
class RLPXFrameReader
{
public:
	RLPXFrameReader(uint16_t _protocolType): m_protocolType(_protocolType) {}
	
	/// Processes a single frame returning complete packets.
	std::vector<RLPXPacket> demux(RLPXFrameCoder& _coder, bytes& _frame, bool _sequence = false, uint16_t _seq = 0, uint32_t _totalSize = 0)
	{
		if (!_coder.authAndDecryptFrame(&_frame))
			BOOST_THROW_EXCEPTION(RLPXFrameDecrytFailed());
		
		std::vector<RLPXPacket> ret;
		if (!_frame.size() || _frame.size() > _totalSize)
			return ret;

		if (_sequence && m_incomplete.count(_seq))
		{
			uint32_t& remaining = m_incomplete.at(_seq).second;
			if (!_totalSize && _frame.size() <= remaining)
			{
				RLPXPacket& p = m_incomplete.at(_seq).first;
				if (_frame.size() > remaining)
					return ret;
				else if(p.streamIn(&_frame))
				{
					ret.push_back(std::move(p));
					m_incomplete.erase(_seq);
				}
				else
					remaining -= _frame.size();
				return ret;
			}
			else
				m_incomplete.erase(_seq);
		}

		bytesConstRef buffer(&_frame);
		while (!buffer.empty())
		{
			auto type = RLPXPacket::nextRLP(buffer);
			if (type.empty())
				break;
			buffer = buffer.cropped(type.size());
			// consume entire buffer if packet has sequence
			auto packet = _sequence ? buffer : RLPXPacket::nextRLP(buffer);
			buffer = buffer.cropped(packet.size());
			RLPXPacket p(m_protocolType, type);
			if (!packet.empty())
				p.streamIn(packet);
			
			if (p.isValid())
				ret.push_back(std::move(p));
			else if (_sequence)
				m_incomplete.insert(std::make_pair(_seq, std::make_pair(std::move(p), _totalSize - _frame.size())));
		}
		return ret;
	}
	
protected:
	uint16_t m_protocolType;
	std::map<uint16_t, std::pair<RLPXPacket, uint32_t>> m_incomplete;	///< Incomplete packets and bytes remaining.
};

/**
 * RLPXFrameWriter
 * Multiplex and encrypted packets into RLPX frames.
 * @todo flag to disable multiple packets per frame
 */
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
	static const uint16_t EmptyFrameLength = h128::size * 3; // header + headerMAC + frameMAC
	static const uint16_t MinFrameDequeLength = h128::size * 4; // header + headerMAC + padded-block + frameMAC

	RLPXFrameWriter(uint16_t _protocolType): m_protocolType(_protocolType) {}
	RLPXFrameWriter(RLPXFrameWriter const& _s): m_protocolType(_s.m_protocolType) {}
	
	size_t size() const { size_t l; size_t h; DEV_GUARDED(m_q.first.x) h = m_q.first.q.size(); DEV_GUARDED(m_q.second.x) l = m_q.second.q.size(); return l + h; }
	
	/// Contents of _payload will be moved. Adds packet to queue, to be muxed into frames by mux when network buffer is ready for writing. Thread-safe.
	void enque(unsigned _packetType, RLPStream& _payload, PacketPriority _priority = PriorityLow);
	
	/// Returns number of packets framed and outputs frames to o_bytes. Not thread-safe.
	size_t mux(RLPXFrameCoder& _coder, unsigned _size, std::vector<bytes>& o_toWrite);
	
private:
	uint16_t const m_protocolType;			// Protocol Type
	std::pair<QueueState, QueueState> m_q;		// High, Low frame queues
	uint16_t m_sequenceId = 0;				// Sequence ID
};

}
}