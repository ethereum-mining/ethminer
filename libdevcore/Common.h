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

// way to many unsigned to size_t warnings in 32 bit build
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
#include <functional>
#pragma warning(push)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <boost/multiprecision/cpp_int.hpp>
#pragma warning(pop)
#pragma GCC diagnostic pop
#include "vector_ref.h"
#include "debugbreak.h"

// CryptoPP defines byte in the global namespace, so must we.
using byte = uint8_t;

// Quote a given token stream to turn it into a string.
#define DEV_QUOTED_HELPER(s) #s
#define DEV_QUOTED(s) DEV_QUOTED_HELPER(s)

namespace dev
{

extern char const* Version;

// Binary data types.
using bytes = std::vector<byte>;
using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

// Numeric types.
using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u128 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<128, 128, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u512 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<512, 512, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using u256s = std::vector<u256>;
using u160s = std::vector<u160>;
using u256Set = std::set<u256>;
using u160Set = std::set<u160>;

// Map types.
using StringMap = std::map<std::string, std::string>;
using u256Map = std::map<u256, u256>;
using HexMap = std::map<bytes, std::string>;

// String types.
using strings = std::vector<std::string>;

// Fixed-length string types.
using string32 = std::array<char, 32>;

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

inline unsigned int toLog2(u256 _x)
{
	unsigned ret;
	for (ret = 0; _x >>= 1; ++ret) {}
	return ret;
}

/// RAII utility class whose destructor calls a given function.
class ScopeGuard {
public:
	ScopeGuard(std::function<void(void)> _f): m_f(_f) {}
	~ScopeGuard() { m_f(); }
private:
	std::function<void(void)> m_f;
};

// Assertions...

#if defined(_MSC_VER)
#define ETH_FUNC __FUNCSIG__
#elif defined(__GNUC__)
#define ETH_FUNC __PRETTY_FUNCTION__
#else
#define ETH_FUNC __func__
#endif

#define asserts(A) ::dev::assertAux(A, #A, __LINE__, __FILE__, ETH_FUNC)
#define assertsEqual(A, B) ::dev::assertEqualAux(A, B, #A, #B, __LINE__, __FILE__, ETH_FUNC)

inline bool assertAux(bool _a, char const* _aStr, unsigned _line, char const* _file, char const* _func)
{
	bool ret = _a;
	if (!ret)
	{
		std::cerr << "Assertion failed:" << _aStr << " [func=" << _func << ", line=" << _line << ", file=" << _file << "]" << std::endl;
#if ETH_DEBUG
		debug_break();
#endif
	}
	return !ret;
}

template<class A, class B>
inline bool assertEqualAux(A const& _a, B const& _b, char const* _aStr, char const* _bStr, unsigned _line, char const* _file, char const* _func)
{
	bool ret = _a == _b;
	if (!ret)
	{
		std::cerr << "Assertion failed: " << _aStr << " == " << _bStr << " [func=" << _func << ", line=" << _line << ", file=" << _file << "]" << std::endl;
		std::cerr << "   Fail equality: " << _a << "==" << _b << std::endl;
#if ETH_DEBUG
		debug_break();
#endif
	}
	return !ret;
}

}
