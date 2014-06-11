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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Very common stuff (i.e. that every other header needs except vector_ref.h).
 */

#pragma once

// way to many uint to size_t warnings in 32 bit build
#ifdef _M_IX86
#pragma warning(disable:4244)
#endif

#ifdef _MSC_VER
#define _ALLOW_KEYWORD_MACROS
#define noexcept throw()
#endif

#include <map>
#include <vector>
#include <set>
#include <boost/multiprecision/cpp_int.hpp>
#include "vector_ref.h"

// CryptoPP defines byte in the global namespace, so so must we.
using byte = uint8_t;

// Quote a given token stream to turn it into a string.
#define ETH_QUOTED_HELPER(s) #s
#define ETH_QUOTED(s) ETH_QUOTED_HELPER(s)

namespace eth
{

extern char const* EthVersion;

// Binary data types.
using bytes = std::vector<byte>;
using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

// Numeric types.
using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using uint = uint64_t;
using sint = int64_t;
using u256s = std::vector<u256>;
using u160s = std::vector<u160>;
using u256Set = std::set<u256>;
using u160Set = std::set<u160>;

// Map types.
using StringMap = std::map<std::string, std::string>;
using u256Map = std::map<u256, u256>;
using HexMap = std::map<bytes, std::string>;

// Null/Invalid values for convenience.
static const u256 Invalid256 = ~(u256)0;
static const bytes NullBytes;
static const std::map<u256, u256> EmptyMapU256U256;

inline s256 u2s(u256 _u)
{
    static const bigint c_end = (bigint)1 << 256;
    static const u256 c_send = (u256)1 << 255;
    if (_u < c_send)
        return (s256)_u;
    else
        return (s256)-(c_end - _u);
}

inline u256 s2u(s256 _u)
{
    static const bigint c_end = (bigint)1 << 256;
    if (_u >= 0)
        return (u256)_u;
    else
        return (u256)(c_end + _u);
}

}
