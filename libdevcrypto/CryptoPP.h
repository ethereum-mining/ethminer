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

#include <mutex>
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
#include <libdevcore/SHA3.h>
#include "Common.h"

namespace dev
{
namespace crypto
{

using namespace CryptoPP;

inline ECP::Point publicToPoint(Public const& _p) { Integer x(_p.data(), 32); Integer y(_p.data() + 32, 32); return ECP::Point(x,y); }

inline Integer secretToExponent(Secret const& _s) { return std::move(Integer(_s.data(), Secret::size)); }

/**
 * CryptoPP secp256k1 algorithms.
 * @todo Collect ECIES methods into class.
 */
class Secp256k1PP
{	
public:
	Secp256k1PP(): m_oid(ASN1::secp256k1()), m_params(m_oid), m_curve(m_params.GetCurve()), m_q(m_params.GetGroupOrder()), m_qs(m_params.GetSubgroupOrder()) {}

	void toPublic(Secret const& _s, Public& o_public) { exponentToPublic(Integer(_s.data(), sizeof(_s)), o_public); }
	
	/// Encrypts text (replace input). (ECIES w/XOR-SHA1)
	void encrypt(Public const& _k, bytes& io_cipher);
	
	/// Decrypts text (replace input). (ECIES w/XOR-SHA1)
	void decrypt(Secret const& _k, bytes& io_text);
	
	/// Encrypts text (replace input). (ECIES w/AES128-CTR-SHA256)
	void encryptECIES(Public const& _k, bytes& io_cipher);

	/// Decrypts text (replace input). (ECIES w/AES128-CTR-SHA256)
	bool decryptECIES(Secret const& _k, bytes& io_text);
	
	/// Key derivation function used by encryptECIES and decryptECIES.
	bytes eciesKDF(Secret const& _z, bytes _s1, unsigned kdBitLen = 256);
	
	/// @returns siganture of message.
	Signature sign(Secret const& _k, bytesConstRef _message);
	
	/// @returns compact siganture of provided hash.
	Signature sign(Secret const& _k, h256 const& _hash);
	
	/// Verify compact signature (public key is extracted from signature).
	bool verify(Signature const& _signature, bytesConstRef _message);
	
	/// Verify signature.
	bool verify(Public const& _p, Signature const& _sig, bytesConstRef _message, bool _hashed = false);
	
	/// Recovers public key from compact signature. Uses libsecp256k1.
	Public recover(Signature _signature, bytesConstRef _message);
	
	/// Verifies _s is a valid secret key and returns corresponding public key in o_p.
	bool verifySecret(Secret const& _s, Public& o_p);
	
	void agree(Secret const& _s, Public const& _r, Secret& o_s);
	
protected:
	void exportPrivateKey(DL_PrivateKey_EC<ECP> const& _k, Secret& o_s) { _k.GetPrivateExponent().Encode(o_s.writable().data(), Secret::size); }
	
	void exportPublicKey(DL_PublicKey_EC<ECP> const& _k, Public& o_p);
	
	void exponentToPublic(Integer const& _e, Public& o_p);
	
	template <class T> void initializeDLScheme(Secret const& _s, T& io_operator) { std::lock_guard<std::mutex> l(x_params); io_operator.AccessKey().Initialize(m_params, secretToExponent(_s)); }
	
	template <class T> void initializeDLScheme(Public const& _p, T& io_operator) { std::lock_guard<std::mutex> l(x_params); io_operator.AccessKey().Initialize(m_params, publicToPoint(_p)); }
	
private:
	OID m_oid;
	
	std::mutex x_rng;
	AutoSeededRandomPool m_rng;
	
	std::mutex x_params;
	DL_GroupParameters_EC<ECP> m_params;
	
	std::mutex x_curve;
	DL_GroupParameters_EC<ECP>::EllipticCurve m_curve;
	
	Integer m_q;
	Integer m_qs;
};

}
}

