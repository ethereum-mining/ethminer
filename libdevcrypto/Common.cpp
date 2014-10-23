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
/** @file CommonEth.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include <random>
#include <secp256k1/secp256k1.h>
#include "SHA3.h"
using namespace std;
using namespace dev;

//#define ETH_ADDRESS_DEBUG 1

Address dev::toAddress(Secret _private)
{
	secp256k1_start();

	byte pubkey[65];
	int pubkeylen = 65;
	int ok = secp256k1_ecdsa_seckey_verify(_private.data());
	if (!ok)
		return Address();
	ok = secp256k1_ecdsa_pubkey_create(pubkey, &pubkeylen, _private.data(), 0);
	if (asserts(pubkeylen == 65))
		return Address();
	if (!ok)
		return Address();
	ok = secp256k1_ecdsa_pubkey_verify(pubkey, 65);
	if (!ok)
		return Address();
	auto ret = right160(dev::sha3(bytesConstRef(&(pubkey[1]), 64)));
#if ETH_ADDRESS_DEBUG
	cout << "---- ADDRESS -------------------------------" << endl;
	cout << "SEC: " << _private << endl;
	cout << "PUB: " << toHex(bytesConstRef(&(pubkey[1]), 64)) << endl;
	cout << "ADR: " << ret << endl;
#endif
	return ret;
}

KeyPair KeyPair::create()
{
	secp256k1_start();
	static std::mt19937_64 s_eng(time(0));
	std::uniform_int_distribution<uint16_t> d(0, 255);

	for (int i = 0; i < 100; ++i)
	{
		h256 sec;
		for (unsigned i = 0; i < 32; ++i)
			sec[i] = (byte)d(s_eng);

		KeyPair ret(sec);
		if (ret.address())
			return ret;
	}
	return KeyPair();
}

KeyPair::KeyPair(h256 _sec):
	m_secret(_sec)
{
	int ok = secp256k1_ecdsa_seckey_verify(m_secret.data());
	if (!ok)
		return;

	byte pubkey[65];
	int pubkeylen = 65;
	ok = secp256k1_ecdsa_pubkey_create(pubkey, &pubkeylen, m_secret.data(), 0);
	if (!ok || pubkeylen != 65)
		return;

	ok = secp256k1_ecdsa_pubkey_verify(pubkey, 65);
	if (!ok)
		return;

	m_secret = m_secret;
	memcpy(m_public.data(), &(pubkey[1]), 64);
	m_address = right160(dev::sha3(bytesConstRef(&(pubkey[1]), 64)));

#if ETH_ADDRESS_DEBUG
	cout << "---- ADDRESS -------------------------------" << endl;
	cout << "SEC: " << m_secret << endl;
	cout << "PUB: " << m_public << endl;
	cout << "ADR: " << m_address << endl;
#endif
}

KeyPair KeyPair::fromEncryptedSeed(bytesConstRef _seed, std::string const& _password)
{
	return KeyPair(sha3(aesDecrypt(_seed, _password)));
}

void dev::encrypt(Public _k, bytesConstRef _plain, bytes& _cipher)
{
	(void)_k;
	_cipher = _plain.toBytes();
}

bool dev::decrypt(Secret _k, bytesConstRef _cipher, bytes& _plain)
{
	(void)_k;
	_plain = _cipher.toBytes();
	return true;
}

Public dev::recover(Signature _sig, h256 _message)
{
	secp256k1_start();

	byte pubkey[65];
	int pubkeylen = 65;
	if (!secp256k1_ecdsa_recover_compact(_message.data(), 32, _sig.data(), pubkey, &pubkeylen, 0, (int)_sig[64]))
		return Public();

	// right160(dev::sha3(bytesConstRef(&(pubkey[1]), 64)));
#if ETH_CRYPTO_TRACE
	h256* sig = (h256 const*)_sig.data();
	cout << "---- RECOVER -------------------------------" << endl;
	cout << "MSG: " << _message << endl;
	cout << "R S V: " << sig[0] << " " << sig[1] << " " << (int)(_sig[64] - 27) << "+27" << endl;
	cout << "PUB: " << toHex(bytesConstRef(&(pubkey[1]), 64)) << endl;
#endif

	return *(Public const*)&(pubkey[1]);
}

inline h256 kFromMessage(h256 _msg, h256 _priv)
{
	return _msg ^ _priv;
}

Signature dev::sign(Secret _k, h256 _message)
{
	int v = 0;

	secp256k1_start();

	SignatureStruct ret;
	h256 nonce = kFromMessage(_message, _k);

	if (!secp256k1_ecdsa_sign_compact(_message.data(), 32, ret.r.data(), _k.data(), nonce.data(), &v))
		return Signature();
#if ETH_ADDRESS_DEBUG
	cout << "---- SIGN -------------------------------" << endl;
	cout << "MSG: " << _message << endl;
	cout << "SEC: " << _k << endl;
	cout << "NON: " << nonce << endl;
	cout << "R S V: " << ret.r << " " << ret.s << " " << v << "+27" << endl;
#endif

	ret.v = v;
	return *(Signature const*)&ret;
}
