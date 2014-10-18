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
 * Ethereum-specific data structures & algorithms.
 */

#pragma once

#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-function"
#include <osrng.h>
#include <oids.h>
#include <filters.h>
#include <eccrypto.h>
#include <ecp.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "Common.h"

namespace dev
{
namespace crypto
{

inline CryptoPP::AutoSeededRandomPool& PRNG()
{
	static CryptoPP::AutoSeededRandomPool prng;
	return prng;
}

inline CryptoPP::OID secp256k1()
{
	return CryptoPP::ASN1::secp256k1();
}
	
class ECKeyPair
{
public:
	static ECKeyPair create();
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pub() { return m_pub; } // deprecate
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> sec() { return m_sec; } // deprecate
	
private:
	ECKeyPair() {}
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> m_pub;
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> m_sec;
};

//class ECDHE;
//bytes ECSign(KeyPair, bytesConstRef);
//bool ECVerify(Public, bytesConstRef);
	
}
}
