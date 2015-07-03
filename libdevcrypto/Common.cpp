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
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <libscrypt/libscrypt.h>
#include <libdevcore/Guards.h>
#include <libdevcore/SHA3.h>
#include <libdevcore/FileSystem.h>
#if ETH_HAVE_SECP256K1
#include <secp256k1/secp256k1.h>
#endif
#include "AES.h"
#include "CryptoPP.h"
#include "Exceptions.h"
using namespace std;
using namespace dev;
using namespace dev::crypto;

#ifdef ETH_HAVE_SECP256K1
struct Secp256k1Context
{
	Secp256k1Context() { secp256k1_start(); }
	~Secp256k1Context() { secp256k1_stop(); }
};
static Secp256k1Context s_secp256k1;
#endif

static Secp256k1PP s_secp256k1pp;

bool dev::SignatureStruct::isValid() const noexcept
{
	if (v > 1 ||
		r >= h256("0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141") ||
		s >= h256("0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141") ||
		s < h256(1) ||
		r < h256(1))
		return false;
	return true;
}

Address dev::ZeroAddress = Address();

Public dev::toPublic(Secret const& _secret)
{
#ifdef ETH_HAVE_SECP256K1
	bytes o(65);
	int pubkeylen;
	if (!secp256k1_ecdsa_pubkey_create(o.data(), &pubkeylen, _secret.data(), false))
		return Public();
	return FixedHash<64>(o.data()+1, Public::ConstructFromPointer);
#else
	Public p;
	s_secp256k1pp.toPublic(_secret, p);
	return p;
#endif
}

Address dev::toAddress(Public const& _public)
{
	return right160(sha3(_public.ref()));
}

Address dev::toAddress(Secret const& _secret)
{
	Public p;
	s_secp256k1pp.toPublic(_secret, p);
	return toAddress(p);
}

void dev::encrypt(Public const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	bytes io = _plain.toBytes();
	s_secp256k1pp.encrypt(_k, io);
	o_cipher = std::move(io);
}

bool dev::decrypt(Secret const& _k, bytesConstRef _cipher, bytes& o_plaintext)
{
	bytes io = _cipher.toBytes();
	s_secp256k1pp.decrypt(_k, io);
	if (io.empty())
		return false;
	o_plaintext = std::move(io);
	return true;
}

void dev::encryptECIES(Public const& _k, bytesConstRef _plain, bytes& o_cipher)
{
	bytes io = _plain.toBytes();
	s_secp256k1pp.encryptECIES(_k, io);
	o_cipher = std::move(io);
}

bool dev::decryptECIES(Secret const& _k, bytesConstRef _cipher, bytes& o_plaintext)
{
	bytes io = _cipher.toBytes();
	if (!s_secp256k1pp.decryptECIES(_k, io))
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

std::pair<bytes, h128> dev::encryptSymNoAuth(h128 const& _k, bytesConstRef _plain)
{
	h128 iv(Nonce::get());
	return make_pair(encryptSymNoAuth(_k, iv, _plain), iv);
}

bytes dev::encryptAES128CTR(bytesConstRef _k, h128 const& _iv, bytesConstRef _plain)
{
	if (_k.size() != 16 && _k.size() != 24 && _k.size() != 32)
		return bytes();
	SecByteBlock key(_k.data(), _k.size());
	try
	{
		CTR_Mode<AES>::Encryption e;
		e.SetKeyWithIV(key, key.size(), _iv.data());
		bytes ret(_plain.size());
		e.ProcessData(ret.data(), _plain.data(), _plain.size());
		return ret;
	}
	catch (CryptoPP::Exception& _e)
	{
		cerr << _e.what() << endl;
		return bytes();
	}
}

bytes dev::decryptAES128CTR(bytesConstRef _k, h128 const& _iv, bytesConstRef _cipher)
{
	if (_k.size() != 16 && _k.size() != 24 && _k.size() != 32)
		return bytes();
	SecByteBlock key(_k.data(), _k.size());
	try
	{
		CTR_Mode<AES>::Decryption d;
		d.SetKeyWithIV(key, key.size(), _iv.data());
		bytes ret(_cipher.size());
		d.ProcessData(ret.data(), _cipher.data(), _cipher.size());
		return ret;
	}
	catch (CryptoPP::Exception& _e)
	{
		cerr << _e.what() << endl;
		return bytes();
	}
}

Public dev::recover(Signature const& _sig, h256 const& _message)
{
#ifdef ETH_HAVE_SECP256K1
	bytes o(65);
	int pubkeylen;
	if (!secp256k1_ecdsa_recover_compact(_message.data(), h256::size, _sig.data(), o.data(), &pubkeylen, false, _sig[64]))
		return Public();
	return FixedHash<64>(o.data()+1, Public::ConstructFromPointer);
#else
	return s_secp256k1pp.recover(_sig, _message.ref());
#endif
}

Signature dev::sign(Secret const& _k, h256 const& _hash)
{
#ifdef ETH_HAVE_SECP256K1
	Signature s;
	if (!secp256k1_ecdsa_sign_compact(_hash.data(), h256::size, s.data(), _k.data(), Nonce::get().data(), (int*)(s.data()+64)))
		return Signature();
	return s;
#else
	return s_secp256k1pp.sign(_k, _hash);
#endif
}

bool dev::verify(Public const& _p, Signature const& _s, h256 const& _hash)
{
	if (!_p)
		return false;
#ifdef ETH_HAVE_SECP256K1
	return _p == recover(_s, _hash);
#else
	return s_secp256k1pp.verify(_p, _s, _hash.ref(), true);
#endif
}

bytes dev::pbkdf2(string const& _pass, bytes const& _salt, unsigned _iterations, unsigned _dkLen)
{
	bytes ret(_dkLen);
	if (PKCS5_PBKDF2_HMAC<SHA256>().DeriveKey(
		ret.data(),
		ret.size(),
		0,
		reinterpret_cast<byte const*>(_pass.data()),
		_pass.size(),
		_salt.data(),
		_salt.size(),
		_iterations
	) != _iterations)
		BOOST_THROW_EXCEPTION(CryptoException() << errinfo_comment("Key derivation failed."));
	return ret;
}

bytes dev::scrypt(std::string const& _pass, bytes const& _salt, uint64_t _n, uint32_t _r, uint32_t _p, unsigned _dkLen)
{
	bytes ret(_dkLen);
	if (libscrypt_scrypt(
		reinterpret_cast<uint8_t const*>(_pass.data()),
		_pass.size(),
		_salt.data(),
		_salt.size(),
		_n,
		_r,
		_p,
		ret.data(),
		ret.size()
	) != 0)
		BOOST_THROW_EXCEPTION(CryptoException() << errinfo_comment("Key derivation failed."));
	return ret;
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
	if (s_secp256k1pp.verifySecret(m_secret, m_public))
		m_address = toAddress(m_public);
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
	return s;
}

mutex Nonce::s_x;
static string s_seedFile;

h256 Nonce::get()
{
	// todo: atomic efface bit, periodic save, kdf, rr, rng
	// todo: encrypt
	Guard l(Nonce::s_x);
	return Nonce::singleton().next();
}

void Nonce::reset()
{
	Guard l(Nonce::s_x);
	Nonce::singleton().resetInternal();
}

void Nonce::setSeedFilePath(string const& _filePath)
{
	s_seedFile = _filePath;
}

Nonce::~Nonce()
{
	Guard l(Nonce::s_x);
	if (m_value)
		// this might throw
		resetInternal();
}

Nonce& Nonce::singleton()
{
	static Nonce s;
	return s;
}

void Nonce::initialiseIfNeeded()
{
	if (m_value)
		return;

	bytes b = contents(seedFile());
	if (b.size() == 32)
		memcpy(m_value.data(), b.data(), 32);
	else
	{
		// todo: replace w/entropy from user and system
		std::mt19937_64 s_eng(time(0) + chrono::high_resolution_clock::now().time_since_epoch().count());
		std::uniform_int_distribution<uint16_t> d(0, 255);
		for (unsigned i = 0; i < 32; ++i)
			m_value[i] = byte(d(s_eng));
	}
	if (!m_value)
		BOOST_THROW_EXCEPTION(InvalidState());

	// prevent seed reuse if process terminates abnormally
	// this might throw
	writeFile(seedFile(), bytes());
}

h256 Nonce::next()
{
	initialiseIfNeeded();
	m_value = sha3(m_value);
	return m_value;
}

void Nonce::resetInternal()
{
	// this might throw
	next();
	writeFile(seedFile(), m_value.asBytes());
	m_value = h256();
}

string const& Nonce::seedFile()
{
	if (s_seedFile.empty())
		s_seedFile = getDataDir() + "/seed";
	return s_seedFile;
}
