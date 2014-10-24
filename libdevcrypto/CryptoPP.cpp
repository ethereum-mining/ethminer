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

ECP::Point pp::PointFromPublic(Public const& _p)
{
	ECP::Point p;
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pub;
	pub.AccessGroupParameters().Initialize(pp::secp256k1());

	bytes prefixedKey(pub.GetGroupParameters().GetEncodedElementSize(true));
	prefixedKey[0] = 0x04;
	assert(Public::size == prefixedKey.size() - 1);
	memcpy(&prefixedKey[1], _p.data(), prefixedKey.size() - 1);
	
	pub.GetGroupParameters().GetCurve().DecodePoint(p, prefixedKey.data(), prefixedKey.size());
	return std::move(p);
}

Integer pp::ExponentFromSecret(Secret const& _s)
{
	return std::move(Integer(_s.data(), 32));
}

void pp::PublicFromExponent(Integer const& _e, Public& _p)
{
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> k;
	k.AccessGroupParameters().Initialize(secp256k1());
	k.SetPrivateExponent(_e);

	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> p;
	p.AccessGroupParameters().Initialize(secp256k1());
	k.MakePublicKey(p);
	pp::PublicFromDL_PublicKey_EC(p, _p);
}

void pp::PublicFromDL_PublicKey_EC(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& _p)
{
	bytes prefixedKey(_k.GetGroupParameters().GetEncodedElementSize(true));
	_k.GetGroupParameters().GetCurve().EncodePoint(prefixedKey.data(), _k.GetPublicElement(), false);
	
	static_assert(Public::size == 64, "Public key must be 64 bytes.");
	assert(Public::size + 1 == _k.GetGroupParameters().GetEncodedElementSize(true));
	memcpy(_p.data(), &prefixedKey[1], Public::size);
}

void pp::SecretFromDL_PrivateKey_EC(CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> const& _k, Secret& _s)
{
	_k.GetPrivateExponent().Encode(_s.data(), Secret::size);
}
