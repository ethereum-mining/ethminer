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
/** @file CryptoPP.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include <libdevcore/Guards.h>
#include "ECDHE.h"
#include "CryptoPP.h"

using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;

static_assert(dev::Secret::size == 32, "Secret key must be 32 bytes.");
static_assert(dev::Public::size == 64, "Public key must be 64 bytes.");
static_assert(dev::Signature::size == 65, "Signature must be 65 bytes.");

bytes Secp256k1::eciesKDF(Secret _z, bytes _s1, unsigned kdBitLen)
{
	// interop w/go ecies implementation
	
	if (!_s1.size())
	{
		_s1.resize(1);
		asserts(_s1[0] == 0);
	}
	
	// for sha3, hash.blocksize is 1088 bits, but this might really be digest size
	auto reps = ((kdBitLen + 7) * 8) / (32 * 8);
	bytes ctr({0, 0, 0, 1});
	bytes k;
	CryptoPP::SHA256 ctx;
	while (reps--)
	{
		ctx.Update(ctr.data(), ctr.size());
		ctx.Update(_z.data(), Secret::size);
		ctx.Update(_s1.data(), _s1.size());
		// append hash to k
		bytes digest(32);
		ctx.Final(digest.data());
		ctx.Restart();
		
		k.reserve(k.size() + h256::size);
		move(digest.begin(), digest.end(), back_inserter(k));
		
		if (ctr[3]++ && ctr[3] != 0) {
			continue;
		} else if (ctr[2]++ && ctr[2] != 0) {
			continue;
		} else if (ctr[1]++ && ctr[1] != 0) {
			continue;
		} else
			ctr[0]++;
	}
	
	k.resize(kdBitLen / 8);
	return move(k);
}

void Secp256k1::encryptECIES(Public const& _k, bytes& io_cipher)
{
	// interop w/go ecies implementation
	auto r = KeyPair::create();
	h256 z;
	ecdh::agree(r.sec(), _k, z);
	auto key = eciesKDF(z, bytes(), 512);
	bytesConstRef eKey = bytesConstRef(&key).cropped(0, 32);
	bytesRef mKey = bytesRef(&key).cropped(32, 32);
	sha3(mKey, mKey);
	
	bytes cipherText;
	encryptSymNoAuth(*(Secret*)eKey.data(), bytesConstRef(&io_cipher), cipherText, h128());
	if (!cipherText.size())
		return;

	bytes msg(1 + Public::size + h128::size + cipherText.size() + 32);
	msg[0] = 0x04;
	r.pub().ref().copyTo(bytesRef(&msg).cropped(1, Public::size));
	bytesRef msgCipherRef = bytesRef(&msg).cropped(1 + Public::size + h128::size, cipherText.size());
	bytesConstRef(&cipherText).copyTo(msgCipherRef);
	
	// tag message
	CryptoPP::HMAC<SHA256> ctx(mKey.data(), mKey.size());
	bytesConstRef cipherWithIV = bytesRef(&msg).cropped(1 + Public::size, h128::size + cipherText.size());
	ctx.Update(cipherWithIV.data(), cipherWithIV.size());
	ctx.Final(msg.data() + 1 + Public::size + cipherWithIV.size());
	
	io_cipher.resize(msg.size());
	io_cipher.swap(msg);
}

bool Secp256k1::decryptECIES(Secret const& _k, bytes& io_text)
{
	// interop w/go ecies implementation
	
	// io_cipher[0] must be 2, 3, or 4, else invalidpublickey
	if (io_text[0] < 2 || io_text[0] > 4)
		// invalid message: publickey
		return false;
	
	if (io_text.size() < (1 + Public::size + h128::size + 1 + h256::size))
		// invalid message: length
		return false;

	h256 z;
	ecdh::agree(_k, *(Public*)(io_text.data()+1), z);
	auto key = eciesKDF(z, bytes(), 512);
	bytesConstRef eKey = bytesConstRef(&key).cropped(0, 32);
	bytesRef mKey = bytesRef(&key).cropped(32, 32);
	sha3(mKey, mKey);
	
	bytes plain;
	size_t cipherLen = io_text.size() - 1 - Public::size - h128::size - h256::size;
	bytesConstRef cipherWithIV(io_text.data() + 1 + Public::size, h128::size + cipherLen);
	bytesConstRef cipher = cipherWithIV.cropped(h128::size, cipherLen);
	bytesConstRef msgMac(cipher.data() + cipher.size(), h256::size);
	
	// verify tag
	CryptoPP::HMAC<SHA256> ctx(mKey.data(), mKey.size());
	ctx.Update(cipherWithIV.data(), cipherWithIV.size());
	h256 mac;
	ctx.Final(mac.data());
	for (unsigned i = 0; i < h256::size; i++)
		if (mac[i] != msgMac[i])
			return false;
	
	decryptSymNoAuth(*(Secret*)eKey.data(), h128(), cipher, plain);
	io_text.resize(plain.size());
	io_text.swap(plain);
	
	return true;
}

void Secp256k1::encrypt(Public const& _k, bytes& io_cipher)
{
	ECIES<ECP>::Encryptor e;
	initializeDLScheme(_k, e);

	size_t plen = io_cipher.size();
	bytes ciphertext;
	ciphertext.resize(e.CiphertextLength(plen));
	
	{
		Guard l(x_rng);
		e.Encrypt(m_rng, io_cipher.data(), plen, ciphertext.data());
	}
	
	memset(io_cipher.data(), 0, io_cipher.size());
	io_cipher = std::move(ciphertext);
}

void Secp256k1::decrypt(Secret const& _k, bytes& io_text)
{
	CryptoPP::ECIES<CryptoPP::ECP>::Decryptor d;
	initializeDLScheme(_k, d);

	if (!io_text.size())
	{
		io_text.resize(1);
		io_text[0] = 0;
	}
	
	size_t clen = io_text.size();
	bytes plain;
	plain.resize(d.MaxPlaintextLength(io_text.size()));
	
	DecodingResult r;
	{
		Guard l(x_rng);
		r = d.Decrypt(m_rng, io_text.data(), clen, plain.data());
	}
	
	if (!r.isValidCoding)
	{
		io_text.clear();
		return;
	}
	
	io_text.resize(r.messageLength);
	io_text = std::move(plain);
}

Signature Secp256k1::sign(Secret const& _k, bytesConstRef _message)
{
	return sign(_k, sha3(_message));
}

Signature Secp256k1::sign(Secret const& _key, h256 const& _hash)
{
	// assumption made by signing alogrithm
	asserts(m_q == m_qs);
	
	Signature sig;
	
	Integer k(kdf(_key, _hash).data(), 32);
	if (k == 0)
		BOOST_THROW_EXCEPTION(InvalidState());
	k = 1 + (k % (m_qs - 1));
	
	ECP::Point rp;
	Integer r;
	{
		Guard l(x_params);
		rp = m_params.ExponentiateBase(k);
		r = m_params.ConvertElementToInteger(rp);
	}
	sig[64] = 0;
//	sig[64] = (r >= m_q) ? 2 : 0;
	
	Integer kInv = k.InverseMod(m_q);
	Integer z(_hash.asBytes().data(), 32);
	Integer s = (kInv * (Integer(_key.asBytes().data(), 32) * r + z)) % m_q;
	if (r == 0 || s == 0)
		BOOST_THROW_EXCEPTION(InvalidState());
	
//	if (s > m_qs)
//	{
//		s = m_q - s;
//		if (sig[64])
//			sig[64] ^= 1;
//	}
	
	sig[64] |= rp.y.IsOdd() ? 1 : 0;
	r.Encode(sig.data(), 32);
	s.Encode(sig.data() + 32, 32);
	return sig;
}

bool Secp256k1::verify(Signature const& _signature, bytesConstRef _message)
{
	return !!recover(_signature, _message);
}

bool Secp256k1::verify(Public const& _p, Signature const& _sig, bytesConstRef _message, bool _hashed)
{
	// todo: verify w/o recovery (if faster)
	return (bool)_p == _hashed ? (bool)recover(_sig, _message) : (bool)recover(_sig, sha3(_message).ref());
}

Public Secp256k1::recover(Signature _signature, bytesConstRef _message)
{
	Public recovered;
	
	Integer r(_signature.data(), 32);
	Integer s(_signature.data()+32, 32);
	// cryptopp encodes sign of y as 0x02/0x03 instead of 0/1 or 27/28
	byte encodedpoint[33];
	encodedpoint[0] = _signature[64] | 2;
	memcpy(&encodedpoint[1], _signature.data(), 32);
	
	ECP::Element x;
	{
		Guard l(x_curve);
		m_curve.DecodePoint(x, encodedpoint, 33);
		if (!m_curve.VerifyPoint(x))
			return recovered;
	}
	
//	if (_signature[64] & 2)
//	{
//		r += m_q;
//		Guard l(x_params);
//		if (r >= m_params.GetMaxExponent())
//			return recovered;
//	}
	
	Integer z(_message.data(), 32);
	Integer rn = r.InverseMod(m_q);
	Integer u1 = m_q - (rn.Times(z)).Modulo(m_q);
	Integer u2 = (rn.Times(s)).Modulo(m_q);
	
	ECP::Point p;
	byte recoveredbytes[65];
	{
		Guard l(x_curve);
		// todo: make generator member
		p = m_curve.CascadeMultiply(u2, x, u1, m_params.GetSubgroupGenerator());
		m_curve.EncodePoint(recoveredbytes, p, false);
	}
	memcpy(recovered.data(), &recoveredbytes[1], 64);
	return recovered;
}

bool Secp256k1::verifySecret(Secret const& _s, Public& _p)
{
	DL_PrivateKey_EC<ECP> k;
	k.Initialize(m_params, secretToExponent(_s));
	if (!k.Validate(m_rng, 3))
		return false;
	
	DL_PublicKey_EC<CryptoPP::ECP> pub;
	k.MakePublicKey(pub);
	if (!k.Validate(m_rng, 3))
		return false;

	exportPublicKey(pub, _p);
	return true;
}

void Secp256k1::agree(Secret const& _s, Public const& _r, h256& o_s)
{
	(void)o_s;
	(void)_s;
	ECDH<ECP>::Domain d(ASN1::secp256k1());
	assert(d.AgreedValueLength() == sizeof(o_s));
	byte remote[65] = {0x04};
	memcpy(&remote[1], _r.data(), 64);
	assert(d.Agree(o_s.data(), _s.data(), remote));
}

void Secp256k1::exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& o_p)
{
	bytes prefixedKey(_k.GetGroupParameters().GetEncodedElementSize(true));
	
	{
		Guard l(x_params);
		m_params.GetCurve().EncodePoint(prefixedKey.data(), _k.GetPublicElement(), false);
		assert(Public::size + 1 == _k.GetGroupParameters().GetEncodedElementSize(true));
	}

	memcpy(o_p.data(), &prefixedKey[1], Public::size);
}

void Secp256k1::exponentToPublic(Integer const& _e, Public& o_p)
{
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pk;
	
	{
		Guard l(x_params);
		pk.Initialize(m_params, m_params.ExponentiateBase(_e));
	}
	
	exportPublicKey(pk, o_p);
}

