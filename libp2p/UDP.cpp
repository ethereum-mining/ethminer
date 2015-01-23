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

h256 RLPXDatagramFace::sign(Secret const& _k)
{
	assert(packetType());
	
	RLPStream rlpxstream;
//	rlpxstream.appendRaw(toPublic(_k).asBytes()); // for mdc-based signature
	rlpxstream.appendRaw(bytes(1, packetType()));
	streamRLP(rlpxstream);
	bytes rlpBytes(rlpxstream.out());
	
	bytesConstRef rlp(&rlpBytes);
	h256 hash(dev::sha3(rlp));
	Signature sig = dev::sign(_k, hash);
	
	data.resize(h256::size + Signature::size + rlp.size());
	bytesConstRef packetHash(&data[0], h256::size);

	bytesConstRef signedPayload(&data[h256::size], Signature::size + rlp.size());
	bytesConstRef payloadSig(&data[h256::size], Signature::size);
	bytesConstRef payload(&data[h256::size + Signature::size], rlp.size());
	
	sig.ref().copyTo(payloadSig);
//	rlp.cropped(Public::size, rlp.size() - Public::size).copyTo(payload);
	rlp.copyTo(payload);
	
//	hash.ref().copyTo(packetHash); // for mdc-based signature
	dev::sha3(signedPayload).ref().copyTo(packetHash);

	return std::move(hash);
};

Public RLPXDatagramFace::authenticate(bytesConstRef _sig, bytesConstRef _rlp)
{
	Signature const& sig = *(Signature const*)_sig.data();
	return std::move(dev::recover(sig, sha3(_rlp)));
};

