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

void dev::crypto::encrypt(Public const& _key, bytes& io_cipher)
{
	ECIES<ECP>::Encryptor e;
	e.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1());
	e.AccessKey().SetPublicElement(pp::PointFromPublic(_key));
	size_t plen = io_cipher.size();
	bytes c;
	c.resize(e.CiphertextLength(plen));
	// todo: use StringSource with _plain as input and output.
	e.Encrypt(pp::PRNG(), io_cipher.data(), plen, c.data());
	bzero(io_cipher.data(), io_cipher.size());
	io_cipher = std::move(c);
}

void dev::crypto::decrypt(Secret const& _k, bytes& io_text)
{
	CryptoPP::ECIES<CryptoPP::ECP>::Decryptor d;
	d.AccessKey().AccessGroupParameters().Initialize(pp::secp256k1());
	d.AccessKey().SetPrivateExponent(pp::ExponentFromSecret(_k));
	size_t clen = io_text.size();
	bytes p;
	p.resize(d.MaxPlaintextLength(io_text.size()));
	// todo: use StringSource with _c as input and output.
	DecodingResult r = d.Decrypt(pp::PRNG(), io_text.data(), clen, p.data());
	assert(r.messageLength);
	io_text.resize(r.messageLength);
	io_text = std::move(p);
}

