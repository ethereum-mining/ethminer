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
using namespace std;
using namespace dev;
using namespace dev::p2p;

const char* RLPXWarn::name() { return "!X!"; }
const char* RLPXNote::name() { return "-X-"; }

h256 RLPXDatagramFace::sign(Secret const& _k)
{
	assert(packetType());
	
	RLPStream rlpxstream;
//	rlpxstream.appendRaw(toPublic(_k).asBytes()); // for mdc-based signature
	rlpxstream.appendRaw(bytes(1, packetType())); // prefix by 1 byte for type
	streamRLP(rlpxstream);
	bytes rlpxBytes(rlpxstream.out());
	
	bytesConstRef rlpx(&rlpxBytes);
	h256 sighash(dev::sha3(rlpx)); // H(type||data)
	Signature sig = dev::sign(_k, sighash); // S(H(type||data))
	
	data.resize(h256::size + Signature::size + rlpx.size());
	bytesRef rlpxHash(&data[0], h256::size);
	bytesRef rlpxSig(&data[h256::size], Signature::size);
	bytesRef rlpxPayload(&data[h256::size + Signature::size], rlpx.size());
	
	sig.ref().copyTo(rlpxSig);
	rlpx.copyTo(rlpxPayload);
	
	bytesConstRef signedRLPx(&data[h256::size], data.size() - h256::size);
	dev::sha3(signedRLPx).ref().copyTo(rlpxHash);

	return sighash;
}

Public RLPXDatagramFace::authenticate(bytesConstRef _sig, bytesConstRef _rlp)
{
	Signature const& sig = *(Signature const*)_sig.data();
	return dev::recover(sig, sha3(_rlp));
}

