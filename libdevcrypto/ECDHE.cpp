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

#include "SHA3.h"
#include "SHA3MAC.h"
#include "CryptoPP.h"
#include "ECDHE.h"

using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace dev::crypto::pp;

void ECDHE::agree(Public _remote)
{
	m_remoteEphemeral = _remote;
	ecdhAgree(m_ephemeral.sec(), m_remoteEphemeral, m_sharedSecret);
}

void ECDHEKeyExchange::exchange(bytes& o_exchange)
{
	if (!m_sharedSecret)
		// didn't agree on public remote
		BOOST_THROW_EXCEPTION(InvalidState());

	Public encpk = m_known.first|m_remoteEphemeral;
	bytes exchange(encpk.asBytes());
	
	// This is the public key which we would like the remote to use,
	// which maybe different than previously-known public key.
	// Here we would pick an appropriate alias or generate a new one,
	// but for now, we use static alias passed to constructor.
	//
	Public p;
	pp::exponentToPublic(pp::secretToExponent(m_alias.m_secret), p);
	exchange.resize(exchange.size() + sizeof(p));
	memcpy(exchange.data() - sizeof(p), p.data(), sizeof(p));
	
	// protocol parameters; should be fixed size
	bytes v(asBytes("\x80"));
	exchange.resize(exchange.size() + v.size());
	memcpy(exchange.data() - v.size(), v.data(), v.size());
	
	h256 auth;
	sha3mac(m_alias.m_secret.ref(), m_sharedSecret.ref(), auth.ref());
	Signature sig = crypto::sign(m_alias.m_secret, auth);
	exchange.resize(exchange.size() + sizeof(sig));
	memcpy(exchange.data() - sizeof(sig), sig.data(), sizeof(sig));
	
	aes::AuthenticatedStream aes(aes::Encrypt, m_sharedSecret, 0);
	h256 prefix(sha3((h256)(m_known.second|m_remoteEphemeral)));
	aes.update(prefix.ref());
	
	encrypt(encpk, exchange);
	aes.update(&exchange);

	aes.streamOut(o_exchange);
}



