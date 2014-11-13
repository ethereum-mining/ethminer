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

#include "CryptoPP.h"
#include "AES.h"

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
	CryptoPP::CTR_Mode<CryptoPP::AES>::Encryption mode;
};

Stream::Stream(StreamType _t, h128 _ckey):
	m_cSecret(_ckey)
{
	(void)_t; // encrypt and decrypt are same operation w/ctr
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

