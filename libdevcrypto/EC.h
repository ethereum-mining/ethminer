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

//class ECDHETKeyExchange;
	
// 256-bit sha3(k) || Public = 84
using PublicTrustNonce = h256;
typedef std::pair<PublicTrustNonce,Public> PublicTrust;
	
inline CryptoPP::AutoSeededRandomPool& PRNG() { static CryptoPP::AutoSeededRandomPool prng; return prng; }

inline CryptoPP::OID secp256k1() { return CryptoPP::ASN1::secp256k1(); }

class ECKeyPair
{
	friend class ECDHETKeyExchange;
	
public:
	static ECKeyPair create();
	
	/// deprecate
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pub() { return m_pub; }
	/// deprecate
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> sec() { return m_sec; }

private:
	ECKeyPair() {}
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> m_pub;
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> m_sec;
	
	std::map<Address,PublicTrust> m_trustEgress;
	std::set<PublicTrustNonce> m_trustIngress;
};

}
}

