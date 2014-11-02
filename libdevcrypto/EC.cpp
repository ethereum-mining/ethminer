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
/** @file EC.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * ECDSA, ECIES
 */

#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#include <files.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include <secp256k1/secp256k1.h>
#include "CryptoPP.h"
#include "SHA3.h"
#include "SHA3MAC.h"
#include "EC.h"

static_assert(dev::Secret::size == 32, "Secret key must be 32 bytes.");
static_assert(dev::Public::size == 64, "Public key must be 64 bytes.");
static_assert(dev::Signature::size == 65, "Signature must be 65 bytes.");

using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;
using namespace pp;

void crypto::toPublic(Secret const& _s, Public& o_public)
{
	exponentToPublic(Integer(_s.data(),sizeof(_s)), o_public);
}

h256 crypto::kdf(Secret const& _priv, h256 const& _hash)
{
	h256 s;
	sha3mac(Nonce::get().ref(), _priv.ref(), s.ref());
	assert(s);
	return sha3((_hash ^ s).asBytes());
}

void crypto::encrypt(Public const& _k, bytes& io_cipher)
{
	ECIES<ECP>::Encryptor e;
	initializeDLScheme(_k, e);
	size_t plen = io_cipher.size();
	bytes c;
	c.resize(e.CiphertextLength(plen));
	// todo: use StringSource with io_cipher as input and output.
	e.Encrypt(PRNG, io_cipher.data(), plen, c.data());
	memset(io_cipher.data(), 0, io_cipher.size());
	io_cipher = std::move(c);
}

void crypto::decrypt(Secret const& _k, bytes& io_text)
{
	CryptoPP::ECIES<CryptoPP::ECP>::Decryptor d;
	initializeDLScheme(_k, d);
	size_t clen = io_text.size();
	bytes p;
	p.resize(d.MaxPlaintextLength(io_text.size()));
	// todo: use StringSource with io_text as input and output.
	DecodingResult r = d.Decrypt(PRNG, io_text.data(), clen, p.data());
	if (!r.isValidCoding)
	{
		io_text.clear();
		return;
	}
	io_text.resize(r.messageLength);
	io_text = std::move(p);
}

Signature crypto::sign(Secret const& _k, bytesConstRef _message)
{
	return crypto::sign(_k, sha3(_message));
}

Signature crypto::sign(Secret const& _key, h256 const& _hash)
{
	ECDSA<ECP,SHA3_256>::Signer signer;
	initializeDLScheme(_key, signer);

	Integer const& q = secp256k1Params.GetGroupOrder();
	Integer e(_hash.asBytes().data(), 32);

	Integer k(kdf(_key, _hash).data(), 32);
	k %= secp256k1Params.GetSubgroupOrder()-1;
	
	ECP::Point rp = secp256k1Params.ExponentiateBase(k);
	Integer r = secp256k1Params.ConvertElementToInteger(rp);
	int recid = ((r >= q) ? 2 : 0) | (rp.y.IsOdd() ? 1 : 0);
	
	Integer kInv = k.InverseMod(q);
	Integer s = (kInv * (Integer(_key.asBytes().data(), 32)*r + e)) % q;
	assert(!!r && !!s);
	
	if (s > secp256k1Params.GetSubgroupOrder())
	{
		s = q - s;
		if (recid)
			recid ^= 1;
	}
	
	Signature sig;
	r.Encode(sig.data(), 32);
	s.Encode(sig.data()+32, 32);
	sig[64] = recid;
	return sig;
}

bool crypto::verify(Signature const& _signature, bytesConstRef _message)
{
	return crypto::verify(crypto::recover(_signature, _message), _signature, _message);
}

bool crypto::verify(Public const& _p, Signature const& _sig, bytesConstRef _message, bool _hashed)
{
	static size_t derMaxEncodingLength = 72;
	if (_hashed)
	{
		assert(_message.size() == 32);
		byte encpub[65] = {0x04};
		memcpy(&encpub[1], _p.data(), 64);
		byte dersig[derMaxEncodingLength];
		size_t cssz = DSAConvertSignatureFormat(dersig, derMaxEncodingLength, DSA_DER, _sig.data(), 64, DSA_P1363);
		assert(cssz <= derMaxEncodingLength);
		return (1 == secp256k1_ecdsa_verify(_message.data(), _message.size(), dersig, cssz, encpub, 65));
	}
	
	ECDSA<ECP, SHA3_256>::Verifier verifier;
	initializeDLScheme(_p, verifier);
	return verifier.VerifyMessage(_message.data(), _message.size(), _sig.data(), sizeof(Signature) - 1);
}

Public crypto::recover(Signature _signature, bytesConstRef _message)
{
	secp256k1_start();
	
	byte pubkey[65];
	int pubkeylen = 65;
	if (!secp256k1_ecdsa_recover_compact(_message.data(), 32, _signature.data(), pubkey, &pubkeylen, 0, (int)_signature[64]))
		return Public();
	
#if ETH_CRYPTO_TRACE
	h256* sig = (h256 const*)_signature.data();
	cout << "---- RECOVER -------------------------------" << endl;
	cout << "MSG: " << _message << endl;
	cout << "R S V: " << sig[0] << " " << sig[1] << " " << (int)(_signature[64] - 27) << "+27" << endl;
	cout << "PUB: " << toHex(bytesConstRef(&(pubkey[1]), 64)) << endl;
#endif
	
	Public ret;
	memcpy(&ret, &(pubkey[1]), sizeof(Public));
	return ret;
}

bool crypto::verifySecret(Secret const& _s, Public const& _p)
{
	secp256k1_start();
	int ok = secp256k1_ecdsa_seckey_verify(_s.data());
	if (!ok)
		return false;
	
	byte pubkey[65];
	int pubkeylen = 65;
	ok = secp256k1_ecdsa_pubkey_create(pubkey, &pubkeylen, _s.data(), 0);
	if (!ok || pubkeylen != 65)
		return false;
	
	ok = secp256k1_ecdsa_pubkey_verify(pubkey, 65);
	if (!ok)
		return false;
	
	for (int i = 0; i < 32; i++)
		if (pubkey[i+1]!=_p[i])
			return false;

	return true;
}

