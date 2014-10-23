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
 * CryptoPP wrappers
 */

#include "CryptoPP.h"

using namespace dev;
using namespace dev::crypto;
using namespace pp;
using namespace CryptoPP;


ECP::Point pp::PointFromPublic(Public const& _p)
{
	bytes prefixedKey(65);
	prefixedKey[0] = 0x04;
	memcpy(&prefixedKey[1], _p.data(), 64);
	
	ECP::Point p;
	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> pub;
	pub.AccessGroupParameters().Initialize(pp::secp256k1());
	pub.GetGroupParameters().GetCurve().DecodePoint(p, prefixedKey.data(), 65);
	return std::move(p);
}

Integer pp::ExponentFromSecret(Secret const& _s)
{
	return std::move(Integer(_s.data(), 32));
}

void pp::PublicFromExponent(Integer const& _e, Public& _p) {
	CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> k;
	k.AccessGroupParameters().Initialize(secp256k1());
	k.SetPrivateExponent(_e);

	CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> p;
	p.AccessGroupParameters().Initialize(secp256k1());
	k.MakePublicKey(p);
	pp::PublicFromDL_PublicKey_EC(p, _p);
}

void pp::PublicFromDL_PublicKey_EC(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k, Public& _p) {
	bytes prefixedKey(65);
	_k.GetGroupParameters().GetCurve().EncodePoint(prefixedKey.data(), _k.GetPublicElement(), false);
	memcpy(_p.data(), &prefixedKey[1], 64);
}

void pp::SecretFromDL_PrivateKey_EC(CryptoPP::DL_PrivateKey_EC<CryptoPP::ECP> const& _k, Secret& _s) {
	_k.GetPrivateExponent().Encode(_s.data(), 32);
}
