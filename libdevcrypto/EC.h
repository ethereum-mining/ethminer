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
 * ECDSA, ECIES
 */

#pragma once

#include "Common.h"

namespace dev
{
namespace crypto
{

void toPublic(Secret const& _s, Public& o_public);
h256 kdf(Secret const& _priv, h256 const& _hash);
	
/// Encrypts text (in place).
void encrypt(Public const& _k, bytes& io_cipher);

/// Decrypts text (in place).
void decrypt(Secret const& _k, bytes& io_text);

/// Returns siganture of message.
Signature sign(Secret const& _k, bytesConstRef _message);
	
/// Returns compact siganture of message hash.
Signature sign(Secret const& _k, h256 const& _hash);

/// Verify compact signature (public key is extracted from message).
bool verify(Signature const& _signature, bytesConstRef _message);
	
/// Verify signature.
bool verify(Public const& _p, Signature const& _sig, bytesConstRef _message, bool _hashed = false);

/// Recovers public key from compact signature. Uses libsecp256k1.
Public recover(Signature _signature, bytesConstRef _message);

bool verifySecret(Secret const& _s, Public const& _p);
	
}
	
}

