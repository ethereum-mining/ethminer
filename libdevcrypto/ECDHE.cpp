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
/** @file ECDHE.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "ECDHE.h"
#include <libdevcore/SHA3.h>
#include "CryptoPP.h"

using namespace std;
using namespace dev;
using namespace dev::crypto;

static Secp256k1PP s_secp256k1;

void dev::crypto::ecdh::agree(Secret const& _s, Public const& _r, h256& o_s)
{
	s_secp256k1.agree(_s, _r, o_s);
}

void ECDHE::agree(Public const& _remote, Secret& o_sharedSecret) const
{
	if (m_remoteEphemeral)
		// agreement can only occur once
		BOOST_THROW_EXCEPTION(InvalidState());
	
	m_remoteEphemeral = _remote;
	s_secp256k1.agree(m_ephemeral.sec(), m_remoteEphemeral, o_sharedSecret);
}

void ECDHEKeyExchange::agree(Public const& _remoteEphemeral)
{
	s_secp256k1.agree(m_ephemeral.sec(), _remoteEphemeral, m_ephemeralSecret);
}

void ECDHEKeyExchange::exchange(bytes& o_exchange)
{
	if (!m_ephemeralSecret)
		// didn't agree on public remote
		BOOST_THROW_EXCEPTION(InvalidState());

	// The key exchange payload is in two parts and is encrypted
	// using ephemeral keypair.
	//
	// The first part is the 'prefix' which is a zero-knowledge proof
	// allowing the remote to resume or emplace a previous session.
	// If a session previously exists:
	//	prefix is sha3(token) // todo: ephemeral entropy from both sides
	// If a session doesn't exist:
	//	prefix is sha3(m_ephemeralSecret)
	//
	// The second part is encrypted using the public key which relates to the prefix.
	
	Public encpk = m_known.first ? m_known.first : m_remoteEphemeral;
	bytes exchange(encpk.asBytes());
	
	// This is the public key which we would like the remote to use,
	// which maybe different than the previously-known public key.
	//
	// Here we should pick an appropriate alias or generate a new one,
	// but for now, we use static alias passed to constructor.
	//
	Public p = toPublic(m_alias.m_secret);
	exchange.resize(exchange.size() + sizeof(p));
	memcpy(&exchange[exchange.size() - sizeof(p)], p.data(), sizeof(p));
	
	// protocol parameters; should be fixed size
	bytes v(1, 0x80);
	exchange.resize(exchange.size() + v.size());
	memcpy(&exchange[exchange.size() - v.size()], v.data(), v.size());
	
	h256 auth;
	sha3mac(m_alias.m_secret.ref(), m_ephemeralSecret.ref(), auth.ref());
	Signature sig = s_secp256k1.sign(m_alias.m_secret, auth);
	exchange.resize(exchange.size() + sizeof(sig));
	memcpy(&exchange[exchange.size() - sizeof(sig)], sig.data(), sizeof(sig));
	
	aes::AuthenticatedStream aes(aes::Encrypt, m_ephemeralSecret, 0);
	h256 prefix(sha3(m_known.second ? m_known.second : (h256)m_remoteEphemeral));
	aes.update(prefix.ref());
	
	s_secp256k1.encrypt(encpk, exchange);
	aes.update(&exchange);

	aes.streamOut(o_exchange);
}



