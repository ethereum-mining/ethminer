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
/** @file ECIES.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * Ethereum-specific data structures & algorithms.
 */

#include "EC.h"
#include "ECIES.h"

using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;

ECIESEncryptor::ECIESEncryptor(ECKeyPair* _k)
{
	m_encryptor.AccessKey().AccessGroupParameters().Initialize(secp256k1());
	m_encryptor.AccessKey().SetPublicElement(_k->pub().GetPublicElement());
}

void ECIESEncryptor::encrypt(bytes& _message)
{
	std::string c;
	StringSource ss(_message.data(), _message.size(), true, new PK_EncryptorFilter(PRNG(), m_encryptor, new StringSink(c)));
	bzero(_message.data(), _message.size() * sizeof(byte));
	_message = std::move(bytesRef(c).toBytes());
}

ECIESDecryptor::ECIESDecryptor(ECKeyPair* _k)
{
	m_decryptor.AccessKey().AccessGroupParameters().Initialize(secp256k1());
	m_decryptor.AccessKey().SetPrivateExponent(_k->sec().GetPrivateExponent());
}

bytes ECIESDecryptor::decrypt(bytesConstRef& _c)
{
	std::string p;
	StringSource ss(_c.data(), _c.size(), true, new PK_DecryptorFilter(PRNG(), m_decryptor, new StringSink(p)));
	return std::move(bytesRef(p).toBytes());
}

