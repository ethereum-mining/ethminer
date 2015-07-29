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

void dev::crypto::ecdh::agree(Secret const& _s, Public const& _r, Secret& o_s)
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

