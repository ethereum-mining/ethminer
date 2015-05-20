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
/** @file AES.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include "AES.h"
#include <libdevcore/Common.h>
#include "CryptoPP.h"
using namespace std;
using namespace dev;
using namespace dev::crypto;
using namespace dev::crypto::aes;
using namespace CryptoPP;

struct aes::Aes128Ctr
{
	Aes128Ctr(h128 _k)
	{
		mode.SetKeyWithIV(_k.data(), sizeof(h128), Nonce::get().data());
	}
	CTR_Mode<AES>::Encryption mode;
};

Stream::Stream(StreamType, h128 _ckey):
	m_cSecret(_ckey)
{
	cryptor = new Aes128Ctr(_ckey);
}

Stream::~Stream()
{
	delete cryptor;
}

void Stream::update(bytesRef)
{

}

size_t Stream::streamOut(bytes&)
{
	return 0;
}

bytes dev::aesDecrypt(bytesConstRef _ivCipher, std::string const& _password, unsigned _rounds, bytesConstRef _salt)
{
	bytes pw = asBytes(_password);

	if (!_salt.size())
		_salt = &pw;

	bytes target(64);
	CryptoPP::PKCS5_PBKDF2_HMAC<CryptoPP::SHA256>().DeriveKey(target.data(), target.size(), 0, pw.data(), pw.size(), _salt.data(), _salt.size(), _rounds);

	try
	{
		CryptoPP::AES::Decryption aesDecryption(target.data(), 16);
		auto cipher = _ivCipher.cropped(16);
		auto iv = _ivCipher.cropped(0, 16);
		CryptoPP::CBC_Mode_ExternalCipher::Decryption cbcDecryption(aesDecryption, iv.data());
		std::string decrypted;
		CryptoPP::StreamTransformationFilter stfDecryptor(cbcDecryption, new CryptoPP::StringSink(decrypted));
		stfDecryptor.Put(cipher.data(), cipher.size());
		stfDecryptor.MessageEnd();
		return asBytes(decrypted);
	}
	catch (exception const& e)
	{
		cerr << e.what() << endl;
		return bytes();
	}
}
