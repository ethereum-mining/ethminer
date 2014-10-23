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

#include "Common.h"

namespace dev
{
namespace crypto
{

/// Encrypts text (in place).
void encrypt(Public const& _k, bytes& _text);

/// Decrypts text (in place).
void decrypt(Secret const& _k, bytes& _text);

class SecretKeyRef
{
public:
	/// Creates random secret
	SecretKeyRef();
	
	/// Creates from secret (move).
	SecretKeyRef(Secret _s): m_secret(_s) {}
	
	/// Retrieve the secret key.
	Secret sec() const { return m_secret; }
	
	/// Retrieve the public key.
	Public pub() const;

	/// Retrieve the associated address of the public key.
	Address address() const;
	
private:
	Secret m_secret;
};


	
/// [ECDHE Trusted Key Exchange]:
	
/// ECDSA Signature
using Signature = FixedHash<65>;

/// Secret nonce from trusted key exchange.
using Nonce = h256;

/// Public key with nonce corresponding to trusted key exchange.
typedef std::pair<Nonce,Public> PublicTrust;

/**
 * @brief EC KeyPair
 * @deprecated
 */
class ECKeyPair
{
	
	/// TO BE REMOVED
	
	friend class ECDHETKeyExchange;
	std::map<Address,PublicTrust> m_trustEgress;
	std::set<Nonce> m_trustIngress;
};
	
}
}

