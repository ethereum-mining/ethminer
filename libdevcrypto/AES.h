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
/** @file AES.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 *
 * AES
 * todo: use openssl
 */

#pragma once

#include <atomic>
#include "Common.h"

namespace dev
{
namespace crypto
{
namespace aes
{

struct Aes128Ctr;
enum StreamType { Encrypt, Decrypt };
	
/**
 * @brief Encrypted stream
 */
class Stream
{
public:
	// streamtype maybe irrelevant w/ctr
	Stream(StreamType _t, h128 _ckey);
	~Stream();
	
	virtual void update(bytesRef io_bytes);
	
	/// Move ciphertext to _bytes.
	virtual size_t streamOut(bytes& o_bytes);
	
private:
	Stream(Stream const&) = delete;
	Stream& operator=(Stream const&) = delete;
	
	h128 m_cSecret;
	bytes m_text;

	Aes128Ctr* cryptor;
};
	

/**
 * @brief Encrypted stream with inband SHA3 mac at specific interval.
 */
class AuthenticatedStream: public Stream
{
public:
	AuthenticatedStream(StreamType _t, h128 _ckey, h128 _mackey, unsigned _interval): Stream(_t, _ckey), m_macSecret(_mackey) { m_macInterval = _interval; }
	
	AuthenticatedStream(StreamType _t, Secret const& _s, unsigned _interval): Stream(_t, h128(_s)), m_macSecret(FixedHash<16>(_s[0]+16)) { m_macInterval = _interval; }
	
	/// Adjust mac interval. Next mac will be xored with value.
	void adjustInterval(unsigned _interval) { m_macInterval = _interval; };
	
private:
	AuthenticatedStream(AuthenticatedStream const&) = delete;
	AuthenticatedStream& operator=(AuthenticatedStream const&) = delete;
	
	std::atomic<unsigned> m_macInterval;
	h128 m_macSecret;
};

}
}
}