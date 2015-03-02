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
/** @file Common.cpp
 * @author Gav Wood <i@gavwood.com>
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <libdevcore/Guards.h>
#include "SHA3.h"
#include "FileSystem.h"
#include "CryptoPP.h"
#include "Common.h"
using namespace std;
using namespace dev;
using namespace dev::crypto;

static Secp256k1 s_secp256k1;

bool dev::SignatureStruct::isValid() const
{
	if (this->v > 1 ||
			this->r >= h256("0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141") ||
			this->s >= h256("0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"))
		return false;
	return true;
}

Address dev::ZeroAddress = Address();

Public dev::toPublic(Secret const& _secret)
{
	Public p;
	s_secp256k1.toPublic(_secret, p);
	return std::move(p);
}

Address dev::toAddress(Public const& _public)
{
	return s_secp256k1.toAddress(_public);
}

Address dev::toAddress(Secret const& _secret)
{
	Public p;
	s_secp256k1.toPublic(_secret, p);
	return s_secp256k1.toAddress(p);
}

void dev::encrypt(Public const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	bytes io = _plain.toBytes();
	s_secp256k1.encrypt(_k, io);
	o_cipher = std::move(io);
}

bool dev::decrypt(Secret const& _k, bytesConstRef _cipher, bytes& o_plaintext)
{
	bytes io = _cipher.toBytes();
	s_secp256k1.decrypt(_k, io);
	if (io.empty())
		return false;
	o_plaintext = std::move(io);
	return true;
}

void dev::encryptECIES(Public const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	bytes io = _plain.toBytes();
	s_secp256k1.encryptECIES(_k, io);
	o_cipher = std::move(io);
}

bool dev::decryptECIES(Secret const& _k, bytesConstRef _cipher, bytes& o_plaintext)
{
	bytes io = _cipher.toBytes();
	if (!s_secp256k1.decryptECIES(_k, io))
		return false;
	o_plaintext = std::move(io);
	return true;
}

void dev::encryptSym(Secret const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	// TOOD: @alex @subtly do this properly.
	encrypt(KeyPair(_k).pub(), _plain, o_cipher);
}

bool dev::decryptSym(Secret const& _k, bytesConstRef _cipher, bytes& o_plain)
{
	// TODO: @alex @subtly do this properly.
	return decrypt(_k, _cipher, o_plain);
}

h128 dev::encryptSymNoAuth(Secret const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	h128 iv(Nonce::get());
	return encryptSymNoAuth(_k, _plain, o_cipher, iv);
}

h128 dev::encryptSymNoAuth(Secret const& _k, bytesConstRef _plain, bytes& o_cipher, h128 const& _iv)
{
	const int c_aesBlockLen = 16;
	size_t extraBytes = _plain.size() % c_aesBlockLen;
	size_t trimmedSize = _plain.size() - extraBytes;
	size_t paddedSize = _plain.size() + ((16 - extraBytes) % 16);
	o_cipher.resize(paddedSize);
	
	bytes underflowBytes(16);
	if (o_cipher.size() != _plain.size())
		_plain.cropped(trimmedSize, extraBytes).copyTo(&underflowBytes);
	
	const int c_aesKeyLen = 32;
	SecByteBlock key(_k.data(), c_aesKeyLen);
	try
	{
		CTR_Mode<AES>::Encryption e;
		e.SetKeyWithIV(key, key.size(), _iv.data());
		if (trimmedSize)
			e.ProcessData(o_cipher.data(), _plain.data(), trimmedSize);
		if (extraBytes)
			e.ProcessData(o_cipher.data() + trimmedSize, underflowBytes.data(), underflowBytes.size());
		return _iv;
	}
	catch(CryptoPP::Exception& e)
	{
		cerr << e.what() << endl;
		o_cipher.resize(0);
		return h128();
	}
}

bool dev::decryptSymNoAuth(Secret const& _k, h128 const& _iv, bytesConstRef _cipher, bytes& o_plaintext)
{
	const int c_aesBlockLen = 16;
	asserts(_cipher.size() % c_aesBlockLen == 0);
	o_plaintext.resize(_cipher.size());
	
	const int c_aesKeyLen = 32;
	SecByteBlock key(_k.data(), c_aesKeyLen);
	try
	{
		CTR_Mode<AES>::Decryption d;
		d.SetKeyWithIV(key, key.size(), _iv.data());
		d.ProcessData(o_plaintext.data(), _cipher.data(), _cipher.size());
		return true;
	}
	catch(CryptoPP::Exception& e)
	{
		cerr << e.what() << endl;
		o_plaintext.resize(0);
		return false;
	}
}

Public dev::recover(Signature const& _sig, h256 const& _message)
{
	return s_secp256k1.recover(_sig, _message.ref());
}

Signature dev::sign(Secret const& _k, h256 const& _hash)
{
	return s_secp256k1.sign(_k, _hash);
}

bool dev::verify(Public const& _p, Signature const& _s, h256 const& _hash)
{
	return s_secp256k1.verify(_p, _s, _hash.ref(), true);
}

KeyPair KeyPair::create()
{
	static boost::thread_specific_ptr<mt19937_64> s_eng;
	static unsigned s_id = 0;
	if (!s_eng.get())
		s_eng.reset(new mt19937_64(time(0) + chrono::high_resolution_clock::now().time_since_epoch().count() + ++s_id));

	uniform_int_distribution<uint16_t> d(0, 255);

	for (int i = 0; i < 100; ++i)
	{
		KeyPair ret(FixedHash<32>::random(*s_eng.get()));
		if (ret.address())
			return ret;
	}
	return KeyPair();
}

KeyPair::KeyPair(h256 _sec):
	m_secret(_sec)
{
	if (s_secp256k1.verifySecret(m_secret, m_public))
		m_address = s_secp256k1.toAddress(m_public);
}

KeyPair KeyPair::fromEncryptedSeed(bytesConstRef _seed, std::string const& _password)
{
	return KeyPair(sha3(aesDecrypt(_seed, _password)));
}

h256 crypto::kdf(Secret const& _priv, h256 const& _hash)
{
	// H(H(r||k)^h)
	h256 s;
	sha3mac(Nonce::get().ref(), _priv.ref(), s.ref());
	s ^= _hash;
	sha3(s.ref(), s.ref());
	
	if (!s || !_hash || !_priv)
		BOOST_THROW_EXCEPTION(InvalidState());
	return std::move(s);
}

h256 Nonce::get(bool _commit)
{
	// todo: atomic efface bit, periodic save, kdf, rr, rng
	// todo: encrypt
	static h256 s_seed;
	static string s_seedFile(getDataDir() + "/seed");
	static mutex s_x;
	Guard l(s_x);
	if (!s_seed)
	{
		static Nonce s_nonce;
		bytes b = contents(s_seedFile);
		if (b.size() == 32)
			memcpy(s_seed.data(), b.data(), 32);
		else
		{
			// todo: replace w/entropy from user and system
			std::mt19937_64 s_eng(time(0) + chrono::high_resolution_clock::now().time_since_epoch().count());
			std::uniform_int_distribution<uint16_t> d(0, 255);
			for (unsigned i = 0; i < 32; ++i)
				s_seed[i] = (byte)d(s_eng);
		}
		if (!s_seed)
			BOOST_THROW_EXCEPTION(InvalidState());
		
		// prevent seed reuse if process terminates abnormally
		writeFile(s_seedFile, bytes());
	}
	h256 prev(s_seed);
	sha3(prev.ref(), s_seed.ref());
	if (_commit)
		writeFile(s_seedFile, s_seed.asBytes());
	return std::move(s_seed);
}

Nonce::~Nonce()
{
	Nonce::get(true);
}
