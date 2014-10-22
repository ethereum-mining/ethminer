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
/** @file EC.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * Shared EC classes and functions.
 */

#pragma once

#include "CryptoHeaders.h"
#include "Common.h"

namespace dev
{
namespace crypto
{

namespace pp
// cryptopp wrappers
{
/// RNG used by CryptoPP
inline CryptoPP::AutoSeededRandomPool& PRNG() { static CryptoPP::AutoSeededRandomPool prng; return prng; }

/// EC curve used by CryptoPP
inline CryptoPP::OID const& secp256k1() { static CryptoPP::OID curve = CryptoPP::ASN1::secp256k1(); return curve; }
	
Public exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k);
	
/**
 * @brief CryptoPP-specific EC keypair
 */
class ECKeyPair
{
public:
	/// Create a new, randomly generated keypair.
	Address const& address() const { return m_address; }
	
	Public const& publicKey() const { return m_public; }
	
protected:
	ECKeyPair();
	
	CryptoPP::ECIES<CryptoPP::ECP>::Decryptor m_decryptor;

	Address m_address;
	Public m_public;
};
}
	
/// ECDSA Signature
using Signature = FixedHash<65>;
	
/// Secret nonce from trusted key exchange.
using Nonce = h256;

/// Public key with nonce corresponding to trusted key exchange.
typedef std::pair<Nonce,Public> PublicTrust;

/**
 * @brief EC KeyPair
 * @todo remove secret access
 * @todo Integrate and/or replace KeyPair, move to common.h
 */
class ECKeyPair: public pp::ECKeyPair
{
	friend class ECDHETKeyExchange;
	friend class ECIESEncryptor;
	friend class ECIESDecryptor;
	
public:
	static ECKeyPair create();
	
	/// Replaces text with ciphertext.
	static void encrypt(bytes& _text, Public _key);
	
	/// @returns ciphertext.
	static bytes encrypt(bytesConstRef _text, Public _key);
	
	/// Recover public key from signature.
	static Public recover(Signature _sig, h256 _messageHash);
	
	/// Sign message.
	Signature sign(h256 _messageHash);
	
	/// Decrypt ciphertext.
	bytes decrypt(bytesConstRef _cipher);
	
	/// Encrypt using our own public key.
	void encrypt(bytes& _text);
	
private:
	ECKeyPair() {};

	std::map<Address,PublicTrust> m_trustEgress;
	std::set<Nonce> m_trustIngress;
};
	
}
}

