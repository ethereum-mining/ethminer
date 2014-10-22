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

using namespace dev::crypto;
using namespace CryptoPP;

dev::Public pp::exportPublicKey(CryptoPP::DL_PublicKey_EC<CryptoPP::ECP> const& _k) {
	Public p;
	ECP::Point q(_k.GetPublicElement());
	q.x.Encode(&p.data()[0], 32);
	q.y.Encode(&p.data()[32], 32);
	return p;
}

pp::ECKeyPair::ECKeyPair():
m_decryptor(pp::PRNG(), pp::secp256k1())
{
}