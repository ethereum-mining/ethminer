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
/** @file CryptoPP.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * CryptoPP headers and primitive helper methods
 */

#pragma once

// need to leave this one disabled for link-time. blame cryptopp.
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#include <cryptopp/sha.h>
#include <cryptopp/sha3.h>
#include <cryptopp/ripemd.h>
#include <cryptopp/aes.h>
#include <cryptopp/pwdbased.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>
#include <cryptopp/eccrypto.h>
#include <cryptopp/ecp.h>
#include <cryptopp/files.h>
#include <cryptopp/osrng.h>
#include <cryptopp/oids.h>
#include <cryptopp/dsa.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "Common.h"

namespace dev
{
namespace crypto
{
namespace pp
{
	
using namespace CryptoPP;
	
/// CryptoPP random number pool
static CryptoPP::AutoSeededRandomPool PRNG;
	
/// CryptoPP EC Cruve
static const CryptoPP::OID secp256k1Curve = CryptoPP::ASN1::secp256k1();
	
static const CryptoPP::DL_GroupParameters_EC<CryptoPP::ECP> secp256k1Params(secp256k1Curve);
	
static ECP::Point publicToPoint(Public const& _p) { Integer x(_p.data(), 32); Integer y(_p.data() + 32, 32); return std::move(ECP::Point(x,y)); }
	
static Integer secretToExponent(Secret const& _s) { return std::move(Integer(_s.data(), Secret::size)); }

void exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& _p);

static void exportPrivateKey(CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> const& _k, Secret& _s) { _k.GetPrivateExponent().Encode(_s.data(), Secret::size); }
	
void exponentToPublic(Integer const& _e, Public& _p);
	
template <class T>
void initializeDLScheme(Secret const& _s, T& io_operator) { io_operator.AccessKey().Initialize(pp::secp256k1Params, secretToExponent(_s)); }
	
template <class T>
void initializeDLScheme(Public const& _p, T& io_operator) { io_operator.AccessKey().Initialize(pp::secp256k1Params, publicToPoint(_p)); }

}
}
}

