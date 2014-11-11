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
 */

#pragma once

#include "Common.h"

namespace dev
{
namespace crypto
{
namespace aes
{

using Secret128 = FixedHash<16>;
enum StreamType { Encrypt, Decrypt };
	
/**
 * @brief Encrypted stream
 */
class Stream
{
public:
	Stream(StreamType _t, Secret128 _encS, bool _zero = true): m_type(_t), m_zeroInput(_zero), m_encSecret(_encS) {};
	
	virtual void update(bytesRef io_bytes) {};
	
	/// Move ciphertext to _bytes.
	virtual size_t streamOut(bytes& o_bytes) {};
	
private:
	StreamType m_type;
	bool m_zeroInput;
	Secret128 m_encSecret;
	bytes m_text;
};

/**
 * @brief Encrypted stream with inband SHA3 mac at specific interval.
 */
class AuthenticatedStream: public Stream
{
public:
	AuthenticatedStream(StreamType _t, Secret128 _encS, Secret128 _macS, unsigned _interval, bool _zero = true): Stream(_t, _encS, _zero), m_macSecret(_macS) { m_macInterval = _interval; }
	
	AuthenticatedStream(StreamType _t, Secret const& _s, unsigned _interval, bool _zero = true): Stream(_t, Secret128(_s), _zero), m_macSecret(FixedHash<16>(_s[0]+16)) { m_macInterval = _interval; }
	
	/// Adjust mac interval. Next mac will be xored with value.
	void adjustInterval(unsigned _interval) { m_macInterval = _interval; };
	
private:
	std::atomic<unsigned> m_macInterval;
	Secret128 m_macSecret;
};

}
}
}