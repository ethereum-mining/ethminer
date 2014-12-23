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
/** @file UDP.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "UDP.h"
using namespace dev;
using namespace dev::p2p;

h256 RLPXDatagram::sign(Secret const& _k)
{
	RLPStream packet;
	streamRLP(packet);
	bytes b(packet.out());
	h256 h(dev::sha3(b));
	Signature sig = dev::sign(_k, h);
	data.resize(b.size() + Signature::size);
	sig.ref().copyTo(&data);
	memcpy(data.data() + sizeof(Signature), b.data(), b.size());
	return std::move(h);
}
