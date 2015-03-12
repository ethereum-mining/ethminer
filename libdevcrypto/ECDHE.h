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

#include "AES.h"

namespace dev
{
namespace crypto
{
	
/// Public key of remote and corresponding shared secret.
using AliasSession = std::pair<Public,h256>;
	
/**
 * @brief An addressable EC key pair.
 */
class Alias
{
	friend class ECDHEKeyExchange; // todo: remove
public:
	Alias(Secret _s): m_secret(_s) {};
	
	AliasSession session(Address _a) { return m_sessions.count(_a) ? m_sessions.find(_a)->second : AliasSession(); }
	
private:
	std::map<Address,AliasSession> m_sessions;
	Secret m_secret;
};

namespace ecdh
{
void agree(Secret const& _s, Public const& _r, h256& o_s);
}
	
/**
 * @brief Derive DH shared secret from EC keypairs.
 * As ephemeral keys are single-use, agreement is limited to a single occurence.
 */
class ECDHE
{
public:
	/// Constructor (pass public key for ingress exchange).
	ECDHE(): m_ephemeral(KeyPair::create()) {};

	/// Public key sent to remote.
	Public pubkey() { return m_ephemeral.pub(); }
	
	Secret seckey() { return m_ephemeral.sec(); }
	
	/// Input public key for dh agreement, output generated shared secret.
	void agree(Public const& _remoteEphemeral, Secret& o_sharedSecret) const;
	
protected:
	KeyPair m_ephemeral;					///< Ephemeral keypair; generated.
	mutable Public m_remoteEphemeral;		///< Public key of remote; parameter. Set once when agree is called, otherwise immutable.
};

/**
 * @brief Secure exchange of static keys.
 * Key exchange is encrypted with public key of remote and then encrypted by block cipher. For a blind remote the ecdhe public key is used to encrypt exchange, and for a known remote the known public key is used. The block cipher key is derived from ecdhe shared secret.
 *
 * Usage: Agree -> Exchange -> Authenticate
 */
class ECDHEKeyExchange: private ECDHE
{
public:
	/// Exchange with unknown remote (pass public key for ingress exchange)
	ECDHEKeyExchange(Alias& _k): m_alias(_k) {}

	/// Exchange with known remote
	ECDHEKeyExchange(Alias& _k, AliasSession _known): m_alias(_k), m_known(_known) {}

	/// Provide public key for dh agreement to generate shared secret.
	void agree(Public const& _remoteEphemeral);
	
	/// @returns encrypted payload of key exchange
	void exchange(bytes& o_exchange);
	
	/// Decrypt payload, check mac, check trust, decrypt exchange, authenticate exchange, verify version, verify signature, and if no failure occurs, update or creats trust and derive session-shared-secret.
	bool authenticate(bytes _exchangeIn);

private:
	Secret m_ephemeralSecret;
	Alias m_alias;
	AliasSession m_known;
	Secret m_sharedAliasSecret;

	FixedHash<16> m_sharedC;
	FixedHash<16> m_sharedM;
};

}
}

