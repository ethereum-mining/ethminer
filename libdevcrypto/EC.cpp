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
/** @file EC.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * Shared EC classes and functions.
 */

#pragma warning(push)
#pragma warning(disable:4100 4244)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wextra"
#include <files.h>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "CryptoPP.h"
#include "SHA3.h"
#include "EC.h"

// CryptoPP and dev conflict so dev and pp namespace are used explicitly
using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace CryptoPP;

void dev::crypto::encrypt(Public const& _key, bytes& _plain)
{
	ECIES<ECP>::Encryptor e;
	e.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1());
	e.AccessKey().SetPublicElement(pp::PointFromPublic(_key));
	size_t plen = _plain.size();
	_plain.resize(e.CiphertextLength(plen));
	e.Encrypt(pp::PRNG(), _plain.data(), plen, _plain.data());
}

void dev::crypto::decrypt(Secret const& _k, bytes& _c)
{
	CryptoPP::ECIES<CryptoPP::ECP>::Decryptor m_decryptor;
	m_decryptor.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1());
	m_decryptor.AccessKey().SetPrivateExponent(pp::ExponentFromSecret(_k));
	size_t plen = _c.size();
	DecodingResult r = m_decryptor.Decrypt(pp::PRNG(), _c.data(), plen, _c.data());
	_c.resize(r.messageLength);
}






/// Old stuff :)

ECKeyPair ECKeyPair::create()
{
	ECKeyPair k;

	// export public key and set address
	ECIES<ECP>::Encryptor e(k.m_decryptor.GetKey());
	pp::PublicFromDL_PublicKey_EC(e.GetKey(), k.m_public);
	k.m_address = dev::right160(dev::sha3(k.m_public.ref()));
	
	return k;
}

void ECKeyPair::encrypt(bytes& _text)
{
	ECIES<ECP>::Encryptor e(m_decryptor);
	std::string c;
	StringSource ss(_text.data(), _text.size(), true, new PK_EncryptorFilter(pp::PRNG(), e, new StringSink(c)));
	bzero(_text.data(), _text.size() * sizeof(byte));
	_text = std::move(asBytes(c));
}

void ECKeyPair::decrypt(bytes& _c)
{
	DecodingResult r = m_decryptor.Decrypt(pp::PRNG(), _c.data(), _c.size(), _c.data());
	_c.resize(r.messageLength);
	
//	std::string p;
//	StringSource ss(_c.data(), _c.size(), true, new PK_DecryptorFilter(pp::PRNG(), m_decryptor, new StringSink(p)));
//	return std::move(asBytes(p));
}




