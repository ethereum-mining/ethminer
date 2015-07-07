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
	std::vector<RLPXPacket> demux(RLPXFrameCoder& _coder, RLPXFrameInfo const& _info, bytesRef _frame);
	
protected:
	uint16_t m_protocolType;
	std::map<uint16_t, std::pair<RLPXPacket, uint32_t>> m_incomplete;	///< Sequence: Incomplete packet and bytes remaining.
};

}
}