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

#include "CryptoPP.h"
#include "Common.h"

namespace dev
{
namespace crypto
{
	
/// ECDSA Signature
using Signature = FixedHash<65>;
	
/// Secret nonce from trusted key exchange.
using Nonce = h256;

/// Public key with nonce corresponding to trusted key exchange.
typedef std::pair<Nonce,Public> PublicTrust;

/// Recover public key from signature.
//Public recover(Signature const& _sig, h256 _messageHash);
	
/// Replaces text with ciphertext.
void encrypt(bytes& _text, Public const& _key);
	
/// @returns ciphertext.
//bytes encrypt(bytesConstRef _text, Public const& _key);
	
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

