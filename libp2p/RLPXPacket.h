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
/** @file RLPXPacket.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#pragma once

#include <algorithm>
#include "Common.h"

namespace dev
{
namespace p2p
{

struct RLPXInvalidPacket: virtual dev::Exception {};

static bytesConstRef nextRLP(bytesConstRef _b) { try { RLP r(_b, RLP::AllowNonCanon); return _b.cropped(0, std::min((size_t)r.actualSize(), _b.size())); } catch(...) {} return bytesConstRef(); }

/**
 * RLPX Packet
 */
class RLPXPacket
{
public:
	/// Construct packet. RLPStream data is invalidated.
	RLPXPacket(uint8_t _capId, RLPStream& _type, RLPStream& _data): m_cap(_capId), m_type(std::move(_type.out())), m_data(std::move(_data.out())) {}

	/// Construct packet from single bytestream. RLPStream data is invalidated.
	RLPXPacket(unsigned _capId, bytesConstRef _in): m_cap(_capId), m_type(nextRLP(_in).toBytes()) { if (_in.size() > m_type.size()) { m_data.resize(_in.size() - m_type.size()); _in.cropped(m_type.size()).copyTo(&m_data); } }
	
	RLPXPacket(RLPXPacket const& _p) = delete;
	RLPXPacket(RLPXPacket&& _p): m_cap(_p.m_cap), m_type(std::move(_p.m_type)), m_data(std::move(_p.m_data)) {}

	bytes const& type() const { return m_type; }

	bytes const& data() const { return m_data; }
	
	size_t size() const { try { return RLP(m_type).actualSize() + RLP(m_data, RLP::LaissezFaire).actualSize(); } catch(...) { return 0; } }

	/// Appends byte data and returns if packet is valid.
	bool append(bytesConstRef _in) { auto offset = m_data.size(); m_data.resize(offset + _in.size()); _in.copyTo(bytesRef(&m_data).cropped(offset)); return isValid(); }
	
	virtual bool isValid() const noexcept { try { return !(m_type.empty() && m_data.empty()) && RLP(m_type).actualSize() == m_type.size() && RLP(m_data).actualSize() == m_data.size(); } catch (...) {} return false; }
	
protected:
	uint8_t m_cap;
	bytes m_type;
	bytes m_data;
};

}
}