/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file Transaction.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <secp256k1.h>
#include "Exceptions.h"
#include "Transaction.h"
using namespace std;
using namespace eth;

Transaction::Transaction(bytes const& _rlpData)
{
	RLP rlp(_rlpData);
	nonce = rlp[0].toInt<u256>(RLP::StrictlyInt);
	receiveAddress = rlp[1].toInt<u160>(RLP::StrictlyString);
	value = rlp[2].toInt<u256>(RLP::StrictlyInt);
	fee = rlp[3].toInt<u256>(RLP::StrictlyInt);
	data.reserve(rlp[4].itemCountStrict());
	for (auto const& i: rlp[4])
		data.push_back(i.toInt<u256>(RLP::StrictlyInt));
	vrs = Signature{ rlp[5].toInt<byte>(RLP::StrictlyInt), rlp[6].toInt<u256>(RLP::StrictlyInt), rlp[7].toInt<u256>(RLP::StrictlyInt) };
}

Address Transaction::sender() const
{
	secp256k1_start();

	bytes sig = toBigEndian(vrs.r) + toBigEndian(vrs.s);
	assert(sig.size() == 64);
	bytes msg = sha3Bytes(false);

	byte pubkey[65];
	int pubkeylen = 65;
	if (!secp256k1_ecdsa_recover_compact(msg.data(), msg.size(), sig.data(), pubkey, &pubkeylen, 0, (int)vrs.v - 27))
		throw InvalidSignature();
	return low160(eth::sha3(bytesConstRef(&(pubkey[1]), 64)));
}

void Transaction::sign(PrivateKey _priv)
{
	int v = 0;

	u256 msg = sha3(false);
	byte sig[64];
	if (!secp256k1_ecdsa_sign_compact(toBigEndian(msg).data(), 32, sig, toBigEndian(_priv).data(), toBigEndian(kFromMessage(msg, _priv)).data(), &v))
		throw InvalidSignature();

	vrs.v = v + 27;
	vrs.r = fromBigEndian<u256>(bytesConstRef(sig, 32));
	vrs.s = fromBigEndian<u256>(bytesConstRef(&(sig[32]), 32));
}

void Transaction::fillStream(RLPStream& _s, bool _sig) const
{
	_s.appendList(_sig ? 8 : 5);
	_s << nonce << toCompactBigEndianString(receiveAddress) << value << fee << data;
	if (_sig)
		_s << toCompactBigEndianString(vrs.v) << toCompactBigEndianString(vrs.r) << toCompactBigEndianString(vrs.s);
}

u256 Transaction::kFromMessage(u256 _msg, u256 _priv)
{
	// TODO!
	/*
	v = '\x01' * 32
	k = '\x00' * 32
	priv = encode_privkey(priv,'bin')
	msghash = encode(hash_to_int(msghash),256,32)
	k = hmac.new(k, v+'\x00'+priv+msghash, hashlib.sha256).digest()
	v = hmac.new(k, v, hashlib.sha256).digest()
	k = hmac.new(k, v+'\x01'+priv+msghash, hashlib.sha256).digest()
	v = hmac.new(k, v, hashlib.sha256).digest()
	return decode(hmac.new(k, v, hashlib.sha256).digest(),256)
	*/
	return 0;
}

