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
/** @file Common.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Shared algorithms and data types.
 */

#pragma once

#include <map>
#include <string>
#include <cassert>
#include <random>
#include <sstream>
#include <cstdint>
#include <type_traits>
#include <boost/multiprecision/cpp_int.hpp>
#include "vector_ref.h"

namespace eth
{

using byte = uint8_t;
using bytes = std::vector<byte>;

using bytesRef = vector_ref<byte>;
using bytesConstRef = vector_ref<byte const>;

using bigint = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<>>;
using u256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s256 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<256, 256, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using u160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;
using s160 =  boost::multiprecision::number<boost::multiprecision::cpp_int_backend<160, 160, boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
using uint = uint64_t;
using sint = int64_t;
using u256s = std::vector<u256>;
using u160s = std::vector<u160>;

using StringMap = std::map<std::string, std::string>;
using u256Map = std::map<u256, u256>;
using HexMap = std::map<bytes, std::string>;

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

inline std::string asString(bytes const& _b) { return std::string((char const*)_b.data(), (char const*)(_b.data() + _b.size())); }

template <class _T> inline std::string asHex(_T const& _data, int _w = 2)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << std::hex << std::setfill('0') << std::setw(_w) << (int)(typename std::make_unsigned<decltype(i)>::type)i;
	return ret.str();
}

template <class _T> void trimFront(_T& _t, uint _elements)
{
	memmove(_t.data(), _t.data() + _elements, (_t.size() - _elements) * sizeof(_t[0]));
	_t.resize(_t.size() - _elements);
}

template <class _T, class _U> void pushFront(_T& _t, _U _e)
{
	_t.push_back(_e);
	memmove(_t.data() + 1, _t.data(), (_t.size() - 1) * sizeof(_e));
	_t[0] = _e;
}

class BadHexCharacter: public std::exception {};

std::string randomWord();
std::string escaped(std::string const& _s, bool _all = true);
int fromHex(char _i);
bytes fromUserHex(std::string const& _s);
bytes toHex(std::string const& _s);

template <class _T, class _Out>
inline void toBigEndian(_T _val, _Out& o_out)
{
	auto s = o_out.size();
	for (uint i = 0; i < s; ++i, _val >>= 8)
		o_out[s - 1 - i] = (typename _Out::value_type)(uint8_t)_val;
}

template <class _T, class _In>
inline _T fromBigEndian(_In const& _bytes)
{
	_T ret = 0;
	for (auto i: _bytes)
		ret = (ret << 8) | (byte)(typename std::make_unsigned<typename _In::value_type>::type)i;
	return ret;
}

inline std::string toBigEndianString(u256 _val) { std::string ret(32, '\0'); toBigEndian(_val, ret); return ret; }
inline std::string toBigEndianString(u160 _val) { std::string ret(20, '\0'); toBigEndian(_val, ret); return ret; }

inline bytes toBigEndian(u256 _val) { bytes ret(32); toBigEndian(_val, ret); return ret; }
inline bytes toBigEndian(u160 _val) { bytes ret(20); toBigEndian(_val, ret); return ret; }

template <class _T>
inline std::string toCompactBigEndianString(_T _val)
{
	int i = 0;
	for (_T v = _val; v; ++i, v >>= 8) {}
	std::string ret(i, '\0');
	toBigEndian(_val, ret);
	return ret;
}

template <class _T, class _U> uint commonPrefix(_T const& _t, _U const& _u)
{
	uint s = std::min<uint>(_t.size(), _u.size());
	for (uint i = 0;; ++i)
		if (i == s || _t[i] != _u[i])
			return i;
	return s;
}

u256 ripemd160(bytesConstRef _message);

template <class _T>
inline u160 low160(_T const& _t)
{
	return (u160)(_t & ((((_T)1) << 160) - 1));
}

template <class _T>
inline u160 as160(_T const& _t)
{
	return (u160)(_t & ((((_T)1) << 160) - 1));
}

template <class _T> inline std::vector<_T>& operator+=(std::vector<_T>& _a, std::vector<_T> const& _b)
{
	auto s = _a.size();
	_a.resize(_a.size() + _b.size());
	memcpy(_a.data() + s, _b.data(), _b.size() * sizeof(_T));
	return _a;

}
template <class _T> inline std::vector<_T> operator+(std::vector<_T> const& _a, std::vector<_T> const& _b) { std::vector<_T> ret(_a); return ret += _b; }

}
