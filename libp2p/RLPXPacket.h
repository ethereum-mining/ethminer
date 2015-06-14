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

#include "Common.h"

namespace dev
{
namespace p2p
{

struct RLPXPacketNullPacket: virtual dev::Exception {};
	
class RLPXPacket
{
public:
	RLPXPacket(unsigned _capId, unsigned _type, RLPStream& _rlps): m_cap(_capId), m_type(_type), m_data(std::move(_rlps.out())) { if (!_type && !m_data.size()) BOOST_THROW_EXCEPTION(RLPXPacketNullPacket()); }
	
	unsigned type() const { return m_type; }
	bytes const& data() const { return m_data; }
	size_t size() const { return m_data.size(); }
	
protected:
	unsigned m_cap;
	unsigned m_type;
	bytes m_data;
};

}
}