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
using namespace dev::crypto::aes;
using namespace dev::crypto::pp;
using namespace CryptoPP;

Stream::Stream(StreamType _t, h128 _ckey):
	m_cSecret(_ckey)
{
	(void)_t; // encrypt and decrypt are same operation w/ctr mode
	cryptor = new Aes128Ctr(_ckey);
}

Stream::~Stream()
{
	delete cryptor;
}

void Stream::update(bytesRef io_bytes)
{

}

size_t Stream::streamOut(bytes& o_bytes)
{
	
}

