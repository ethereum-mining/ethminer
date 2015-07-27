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

}
}

