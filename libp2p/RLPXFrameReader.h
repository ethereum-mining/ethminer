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
/** @file RLPXFrameReader.h
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
 * RLPFrameReader
 * Reads and assembles RLPX frame byte buffers into RLPX packets. Additionally
 * buffers incomplete packets which are pieced into multiple frames (has sequence).
 */
class RLPXFrameReader
{
public:
	RLPXFrameReader(uint16_t _protocolType): m_protocolType(_protocolType) {}
	
	/// Processes a single frame returning complete packets.
	std::vector<RLPXPacket> demux(RLPXFrameCoder& _coder, bytesRef _frame, bool _sequence = false, uint16_t _seq = 0, uint32_t _totalSize = 0)
	{
		if (!_coder.authAndDecryptFrame(_frame))
			BOOST_THROW_EXCEPTION(RLPXFrameDecrytFailed());
		
		std::vector<RLPXPacket> ret;
		if (!_sequence && (!_frame.size() || _frame.size() > _totalSize))
			return ret;
		
		// trim mac
		bytesConstRef buffer = _frame.cropped(0, _frame.size() - h128::size);
		// continue populating incomplete packets
		if (_sequence && m_incomplete.count(_seq))
		{
			uint32_t& remaining = m_incomplete.at(_seq).second;
			if (!_totalSize && buffer.size() > 0 && buffer.size() <= remaining)
			{
				remaining -= buffer.size();
				
				RLPXPacket& p = m_incomplete.at(_seq).first;
				if(p.append(buffer) && !remaining)
					ret.push_back(std::move(p));
				if (!remaining)
					m_incomplete.erase(_seq);
				
				if (!ret.empty() && remaining)
					BOOST_THROW_EXCEPTION(RLPXInvalidPacket());
				else if (ret.empty() && !remaining)
					BOOST_THROW_EXCEPTION(RLPXInvalidPacket());

				return ret;
			}
			else
				m_incomplete.erase(_seq);
		}
		
		while (!buffer.empty())
		{
			auto type = nextRLP(buffer);
			if (type.empty())
				break;
			buffer = buffer.cropped(type.size());
			// consume entire buffer if packet has sequence
			auto packet = _sequence ? buffer : nextRLP(buffer);
			buffer = buffer.cropped(packet.size());
			RLPXPacket p(m_protocolType, type);
			if (!packet.empty())
				p.append(packet);
			
			uint32_t remaining = _totalSize - type.size() - packet.size();
			if (p.isValid())
				ret.push_back(std::move(p));
			else if (_sequence && remaining)
				m_incomplete.insert(std::make_pair(_seq, std::make_pair(std::move(p), remaining)));
			// else drop invalid packet
		}
		return ret;
	}
	
protected:
	uint16_t m_protocolType;
	std::map<uint16_t, std::pair<RLPXPacket, uint32_t>> m_incomplete;	///< Sequence: Incomplete packet and bytes remaining.
};

}
}