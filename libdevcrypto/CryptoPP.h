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
 * CryptoPP headers and helper methods
 */

#pragma once

#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#include <sha.h>
#include <sha3.h>
#include <ripemd.h>
#include <aes.h>
#include <pwdbased.h>
#include <modes.h>
#include <filters.h>
#include <eccrypto.h>
#include <ecp.h>
#include <files.h>
#include <osrng.h>
#include <oids.h>
#include <secp256k1/secp256k1.h>
#include <dsa.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "Common.h"

namespace dev
{
namespace crypto
{

namespace pp
{

/// CryptoPP random number pool
static CryptoPP::AutoSeededRandomPool PRNG;
	
/// CryptoPP EC Cruve
static const CryptoPP::OID secp256k1Curve = CryptoPP::ASN1::secp256k1();

/// Initialize signer with Secret
void initializeSigner(Secret const& _s, CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA3_256>::Signer& out_signer);
	
/// Initialize verifier with Public
void initializeVerifier(Public const& _p, CryptoPP::ECDSA<CryptoPP::ECP, CryptoPP::SHA3_256>::Verifier& _verifier);
	
/// Initialize cryptopp encryptor with Public
void initializeEncryptor(Public const& _p, CryptoPP::ECIES<CryptoPP::ECP>::Encryptor& out_encryptor);
	
/// Initialize cryptopp decryptor with Secret
void initializeDecryptor(Secret const& _s, CryptoPP::ECIES<CryptoPP::ECP>::Decryptor& out_decryptor);

/// Conversion from cryptopp public key to bytes
void exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& out_p);
	
/// Conversion from cryptopp private key to bytes
void exportPrivateKey(CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> const& _k, Secret& out_s);
	
}
}
}

