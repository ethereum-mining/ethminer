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
/** @file RLPXFrameReader.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#include "RLPXFrameReader.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

std::vector<RLPXPacket> RLPXFrameReader::demux(RLPXFrameCoder& _coder, RLPXFrameInfo const& _info, bytesRef _frame)
{
	if (!_coder.authAndDecryptFrame(_frame))
		BOOST_THROW_EXCEPTION(RLPXFrameDecryptFailed());
	
	std::vector<RLPXPacket> ret;
	if (_frame.empty())
		// drop: bad frame (empty)
		return ret;
	if (_info.multiFrame && _info.totalLength && _frame.size() > _info.totalLength)
		// drop: bad frame (too large)
		return ret;
	
	// trim mac
	bytesConstRef buffer = _frame.cropped(0, _frame.size() - (h128::size + _info.padding));
	// continue populating multiframe packets
	if (_info.multiFrame && m_incomplete.count(_info.sequenceId))
	{
		uint32_t& remaining = m_incomplete.at(_info.sequenceId).second;
		if (!_info.totalLength && buffer.size() > 0 && buffer.size() <= remaining)
		{
			remaining -= buffer.size();
			
			RLPXPacket& p = m_incomplete.at(_info.sequenceId).first;
			if(p.append(buffer) && !remaining)
				ret.push_back(std::move(p));
			if (!remaining)
				m_incomplete.erase(_info.sequenceId);
			
			if (!ret.empty() && remaining)
				BOOST_THROW_EXCEPTION(RLPXInvalidPacket());
			else if (ret.empty() && !remaining)
				BOOST_THROW_EXCEPTION(RLPXInvalidPacket());
			
			return ret;
		}
		else
			m_incomplete.erase(_info.sequenceId);
	}
	
	while (!buffer.empty())
	{
		auto type = nextRLP(buffer);
		if (type.empty())
			break;
		buffer = buffer.cropped(type.size());
		// consume entire buffer if packet has sequence
		auto packet = _info.multiFrame ? buffer : nextRLP(buffer);
		buffer = buffer.cropped(packet.size());
		RLPXPacket p(m_protocolType, type);
		if (!packet.empty())
			p.append(packet);
		
		uint32_t remaining = _info.totalLength - type.size() - packet.size();
		if (p.isValid())
			ret.push_back(std::move(p));
		else if (_info.multiFrame && remaining)
			m_incomplete.insert(std::make_pair(_info.sequenceId, std::make_pair(std::move(p), remaining)));
		else
			BOOST_THROW_EXCEPTION(RLPXInvalidPacket());
	}
	return ret;
}
