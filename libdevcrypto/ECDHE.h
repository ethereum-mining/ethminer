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
/** @file ECDHE.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * Elliptic curve Diffie-Hellman ephemeral key exchange
 */

#pragma once

#include "EC.h"

namespace dev
{
namespace crypto
{
	
/**
 * @brief Derive DH shared secret from EC keypairs.
 * @todo shield secret()
 */
class ECDHE
{
	friend class ECDHETKeyExchange;
public:
	ECDHE();

	/// Agree on ECDH parmaters and derive shared secret.
	void agree(Public _remote);

protected:
	Secret secret();
	
private:
	KeyPair m_ephemeral;		///< Ephemeral keypair
	Public m_remote;			///< Public key of remote
};

/**
 * @brief Secure exchange of static keys.
 * Key exchange is encrypted with public key of remote and then encrypted by block cipher. For a blind remote the ecdhe public key is used to encrypt exchange, and for a trusted remote the trusted public key is used. The block cipher key is derived from ecdhe shared secret.
 */
class ECDHETKeyExchange
{
public:
	/// Blind key exchange. KeyPair trusts will be updated if successful.
	ECDHETKeyExchange(ECDHE const& _ecdhe, ECKeyPair* _keyTrust);
	
	/// Trusted key exchange. Upon success, KeyPair trusts will be updated.
	ECDHETKeyExchange(ECDHE const& _ecdhe, ECKeyPair* _keyTrust, Address _remote);
	
	/// Authentication for trusted remote, blind trust, or disconnect.
	/// Returns key exchange. encrypted w/aes-ctr. key=ecdhe.m_shared[0-127]
	///
	/// @returns E(K,prefix||e(epub,m||v||sign(k,sha3(dhe-k||m)))||mac)
	///
	/// E = AES in CTR mode (todo: nonce)
	/// K = ecdhe.secret[0..127]
	/// ECDHETKeyExchange(ECDHE const&, ECKeyPair*):
	///   prefix = sha3(ecdhe.remote)
	///   epub = ecdhe.remote
	/// ECDHETKeyExchange(ECDHE const&, ECKeyPair* _k, Address _r):
	///   trust = _k.m_trustEgress.find(_r)
	///   sha3(trust.first)
	///   epub = trust.second
	/// e = ECIES encrypt()
	/// m = keypair.public
	/// v = 0x80
	/// k = keypair.secret
	/// mac = sha3(M||prefix||e()); M = ecdhe.secret[128..255]
	/// K = ecdhe.secret[0..127]
	bytes exchange();
	
	/// Decrypts payload, checks mac, checks trust, decrypts exchange, authenticates exchange, verifies version, verifies signature, and if no failures occur, updates or creats trust and derives trusted-shared-secret.
	bool authenticate(bytes _exchangeIn);
	
	/// Encrypts message; @returns e(k,m).
	void encrypt();
	
	/// Signs message then encrypts; @returns e(k,sign(k,sha3(m))||m).
	bytes signEncrypt(bytes _m);
	
private:
	bool blind;
	ECDHE const& m_ecdhe;
	ECKeyPair* m_keypair;
	PublicTrust m_trust;
	
};

}
}

