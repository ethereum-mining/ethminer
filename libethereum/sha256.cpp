/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file sha256.cpp
 * @author Gav Wood <i@gavwood.com>
 * @author Oliver Gay <olivier.gay@a3.epfl.ch>
 * @date 2014
 * @note Modified from an original version with BSD licence.
 */

/*
 * Updated to C++, zedwood.com 2012
 * Based on Olivier Gay's version
 * See Modified BSD License below:
 *
 * FIPS 180-2 SHA-224/256/384/512 implementation
 * Issue date:  04/30/2005
 * http://www.ouah.org/ogay/sha2/
 *
 * Copyright (C) 2005, 2007 Olivier Gay <olivier.gay@a3.epfl.ch>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <cstring>
#include <fstream>
#include "sha256.h"
using namespace std;
using namespace eth;

const uint32_t SHA256::sha256_k[64] = //UL = uint32_t
			{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
			 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
			 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
			 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
			 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
			 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
			 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
			 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
			 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
			 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
			 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
			 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
			 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void SHA256::transform(byte const* message, unsigned block_nb)
{
	uint32_t w[64];
	uint32_t wv[8];
	uint32_t t1, t2;
	const unsigned char *sub_block;
	int i;
	int j;
	for (i = 0; i < (int) block_nb; i++) {
		sub_block = message + (i << 6);
		for (j = 0; j < 16; j++) {
			SHA2_PACK32(&sub_block[j << 2], &w[j]);
		}
		for (j = 16; j < 64; j++) {
			w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
		}
		for (j = 0; j < 8; j++) {
			wv[j] = m_h[j];
		}
		for (j = 0; j < 64; j++) {
			t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
				+ sha256_k[j] + w[j];
			t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
			wv[7] = wv[6];
			wv[6] = wv[5];
			wv[5] = wv[4];
			wv[4] = wv[3] + t1;
			wv[3] = wv[2];
			wv[2] = wv[1];
			wv[1] = wv[0];
			wv[0] = t1 + t2;
		}
		for (j = 0; j < 8; j++) {
			m_h[j] += wv[j];
		}
	}
}

void SHA256::init()
{
	m_h[0] = 0x6a09e667;
	m_h[1] = 0xbb67ae85;
	m_h[2] = 0x3c6ef372;
	m_h[3] = 0xa54ff53a;
	m_h[4] = 0x510e527f;
	m_h[5] = 0x9b05688c;
	m_h[6] = 0x1f83d9ab;
	m_h[7] = 0x5be0cd19;
	m_len = 0;
	m_tot_len = 0;
}

void SHA256::update(byte const* message, unsigned len)
{
	unsigned int block_nb;
	unsigned int new_len, rem_len, tmp_len;
	const unsigned char *shifted_message;
	tmp_len = SHA224_256_BLOCK_SIZE - m_len;
	rem_len = len < tmp_len ? len : tmp_len;
	memcpy(&m_block[m_len], message, rem_len);
	if (m_len + len < SHA224_256_BLOCK_SIZE) {
		m_len += len;
		return;
	}
	new_len = len - rem_len;
	block_nb = new_len / SHA224_256_BLOCK_SIZE;
	shifted_message = message + rem_len;
	transform(m_block, 1);
	transform(shifted_message, block_nb);
	rem_len = new_len % SHA224_256_BLOCK_SIZE;
	memcpy(m_block, &shifted_message[block_nb << 6], rem_len);
	m_len = rem_len;
	m_tot_len += (block_nb + 1) << 6;
}

void SHA256::final(byte* digest)
{
	unsigned block_nb;
	unsigned pm_len;
	unsigned len_b;
	int i;
	block_nb = (1 + ((SHA224_256_BLOCK_SIZE - 9)
					 < (m_len % SHA224_256_BLOCK_SIZE)));
	len_b = (m_tot_len + m_len) << 3;
	pm_len = block_nb << 6;
	memset(m_block + m_len, 0, pm_len - m_len);
	m_block[m_len] = 0x80;
	SHA2_UNPACK32(len_b, m_block + pm_len - 4);
	transform(m_block, block_nb);

	for (i = 0 ; i < 8; i++) {
		SHA2_UNPACK32(m_h[i], &digest[i << 2]);
	}
}

std::string eth::sha256(std::string const& _input, bool _hex)
{
	if (!_hex)
	{
		string ret(SHA256::DIGEST_SIZE, '\0');
		SHA256 ctx = SHA256();
		ctx.init();
		ctx.update( (unsigned char*)_input.c_str(), _input.length());
		ctx.final((byte*)ret.data());
		return ret;
	}

	unsigned char digest[SHA256::DIGEST_SIZE];
	memset(digest,0,SHA256::DIGEST_SIZE);

	SHA256 ctx = SHA256();
	ctx.init();
	ctx.update( (unsigned char*)_input.c_str(), _input.length());
	ctx.final(digest);

	char buf[2*SHA256::DIGEST_SIZE+1];
	buf[2*SHA256::DIGEST_SIZE] = 0;
	for (unsigned int i = 0; i < SHA256::DIGEST_SIZE; i++)
		sprintf(buf+i*2, "%02x", digest[i]);
	return std::string(buf);
}

bytes eth::sha256Bytes(bytesConstRef _input)
{
	bytes ret(SHA256::DIGEST_SIZE);
	SHA256 ctx = SHA256();
	ctx.init();
	ctx.update((byte*)_input.data(), _input.size());
	ctx.final(ret.data());
	return ret;
}

u256 eth::sha256(bytesConstRef _input)
{
	u256 ret = 0;

	SHA256 ctx = SHA256();
	ctx.init();
	ctx.update(_input.data(), _input.size());
	uint8_t buf[SHA256::DIGEST_SIZE];
	ctx.final(buf);
	for (unsigned i = 0; i < 32; ++i)
		ret = (ret << 8) | buf[i];
	return ret;
}
