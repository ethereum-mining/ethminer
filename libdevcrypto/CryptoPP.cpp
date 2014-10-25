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
/** @file CryptoPP.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "CryptoPP.h"

using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;

/// Conversion from bytes to cryptopp point
inline ECP::Point publicToPoint(Public const& _p);

/// Conversion from bytes to cryptopp exponent
inline Integer secretToExponent(Secret const& _s);

/// Conversion from cryptopp exponent Integer to bytes
inline void exponentToPublic(Integer const& _e, Public& _p);

void pp::initializeSigner(Secret const& _s, ECDSA<ECP, CryptoPP::SHA3_256>::Signer& _signer)
{
	_signer.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1Curve);
	_signer.AccessKey().SetPrivateExponent(secretToExponent(_s));
}

void pp::initializeVerifier(Public const& _p, ECDSA<ECP, CryptoPP::SHA3_256>::Verifier& _verifier)
{
	_verifier.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1Curve);
	_verifier.AccessKey().SetPublicElement(publicToPoint(_p));
}

void pp::initializeEncryptor(Public const& _p, CryptoPP::ECIES<CryptoPP::ECP>::Encryptor& _encryptor)
{
	_encryptor.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1Curve);
	_encryptor.AccessKey().SetPublicElement(publicToPoint(_p));
}

void pp::initializeDecryptor(Secret const& _s, CryptoPP::ECIES<CryptoPP::ECP>::Decryptor& _decryptor)
{
	_decryptor.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1Curve);
	_decryptor.AccessKey().SetPrivateExponent(secretToExponent(_s));
}

void pp::exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& _p)
{
	bytes prefixedKey(_k.GetGroupParameters().GetEncodedElementSize(true));
	_k.GetGroupParameters().GetCurve().EncodePoint(prefixedKey.data(), _k.GetPublicElement(), false);
	
	static_assert(Public::size == 64, "Public key must be 64 bytes.");
	assert(Public::size + 1 == _k.GetGroupParameters().GetEncodedElementSize(true));
	memcpy(_p.data(), &prefixedKey[1], Public::size);
}

void pp::exportPrivateKey(CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> const& _k, Secret& _s)
{
	_k.GetPrivateExponent().Encode(_s.data(), Secret::size);
}

/// Integer and Point Conversion:

inline ECP::Point publicToPoint(Public const& _p)
{
	ECP::Point p;
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pub;
	pub.AccessGroupParameters().Initialize(pp::secp256k1Curve);
	
	bytes prefixedKey(pub.GetGroupParameters().GetEncodedElementSize(true));
	prefixedKey[0] = 0x04;
	assert(Public::size == prefixedKey.size() - 1);
	memcpy(&prefixedKey[1], _p.data(), prefixedKey.size() - 1);
	
	pub.GetGroupParameters().GetCurve().DecodePoint(p, prefixedKey.data(), prefixedKey.size());
	return std::move(p);
}

inline Integer secretToExponent(Secret const& _s)
{
	static_assert(Secret::size == 32, "Secret key must be 32 bytes.");
	return std::move(Integer(_s.data(), Secret::size));
}

inline void exponentToPublic(Integer const& _e, Public& _p)
{
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> k;
	k.AccessGroupParameters().Initialize(pp::secp256k1Curve);
	k.SetPrivateExponent(_e);
	
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> p;
	p.AccessGroupParameters().Initialize(pp::secp256k1Curve);
	k.MakePublicKey(p);
	pp::exportPublicKey(p, _p);
}
