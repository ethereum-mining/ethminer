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
/** @file CommonData.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * Shared algorithms and data types.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <string>
#include "Common.h"

namespace dev
{

// String conversion functions, mainly to/from hex/nibble/byte representations.

enum class ThrowType
{
	NoThrow = 0,
	Throw = 1,
};

/// Convert a series of bytes to the corresponding string of hex duplets.
/// @param _w specifies the width of each of the elements. Defaults to two - enough to represent a byte.
/// @example toHex("A\x69") == "4169"
template <class _T>
std::string toHex(_T const& _data, int _w = 2)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << std::hex << std::setfill('0') << std::setw(_w) << (int)(typename std::make_unsigned<decltype(i)>::type)i;
	return ret.str();
}

/// Converts a (printable) ASCII hex character into the correspnding integer value.
/// @example fromHex('A') == 10 && fromHex('f') == 15 && fromHex('5') == 5
int fromHex(char _i);

/// Converts a (printable) ASCII hex string into the corresponding byte stream.
/// @example fromHex("41626261") == asBytes("Abba")
/// If _throw = ThrowType::NoThrow, it replaces bad hex characters with 0's, otherwise it will throw an exception.
bytes fromHex(std::string const& _s, ThrowType _throw = ThrowType::NoThrow);

#if 0
std::string toBase58(bytesConstRef _data);
bytes fromBase58(std::string const& _s);
#endif

/// Converts byte array to a string containing the same (binary) data. Unless
/// the byte array happens to contain ASCII data, this won't be printable.
inline std::string asString(bytes const& _b)
{
	return std::string((char const*)_b.data(), (char const*)(_b.data() + _b.size()));
}

/// Converts a string to a byte array containing the string's (byte) data.
inline bytes asBytes(std::string const& _b)
{
	return bytes((byte const*)_b.data(), (byte const*)(_b.data() + _b.size()));
}

/// Converts a string into the big-endian base-16 stream of integers (NOT ASCII).
/// @example asNibbles("A")[0] == 4 && asNibbles("A")[1] == 1
bytes asNibbles(std::string const& _s);


// Big-endian to/from host endian conversion functions.

/// Converts a templated integer value to the big-endian byte-stream represented on a templated collection.
/// The size of the collection object will be unchanged. If it is too small, it will not represent the
/// value properly, if too big then the additional elements will be zeroed out.
/// @a _Out will typically be either std::string or bytes.
/// @a _T will typically by unsigned, u160, u256 or bigint.
template <class _T, class _Out>
inline void toBigEndian(_T _val, _Out& o_out)
{
	for (auto i = o_out.size(); i-- != 0; _val >>= 8)
		o_out[i] = (typename _Out::value_type)(uint8_t)_val;
}

/// Converts a big-endian byte-stream represented on a templated collection to a templated integer value.
/// @a _In will typically be either std::string or bytes.
/// @a _T will typically by unsigned, u160, u256 or bigint.
template <class _T, class _In>
inline _T fromBigEndian(_In const& _bytes)
{
	_T ret = 0;
	for (auto i: _bytes)
		ret = (ret << 8) | (byte)(typename std::make_unsigned<typename _In::value_type>::type)i;
	return ret;
}

/// Convenience functions for toBigEndian
inline std::string toBigEndianString(u256 _val) { std::string ret(32, '\0'); toBigEndian(_val, ret); return ret; }
inline std::string toBigEndianString(u160 _val) { std::string ret(20, '\0'); toBigEndian(_val, ret); return ret; }
inline bytes toBigEndian(u256 _val) { bytes ret(32); toBigEndian(_val, ret); return ret; }
inline bytes toBigEndian(u160 _val) { bytes ret(20); toBigEndian(_val, ret); return ret; }

/// Convenience function for conversion of a u256 to hex
inline std::string toHex(u256 val) { return toHex(toBigEndian(val)); }

/// Convenience function for toBigEndian.
/// @returns a byte array just big enough to represent @a _val.
template <class _T>
inline bytes toCompactBigEndian(_T _val, unsigned _min = 0)
{
	int i = 0;
	for (_T v = _val; v; ++i, v >>= 8) {}
	bytes ret(std::max<unsigned>(_min, i), 0);
	toBigEndian(_val, ret);
	return ret;
}
inline bytes toCompactBigEndian(byte _val, unsigned _min = 0)
{
	return (_min || _val) ? bytes{ _val } : bytes{};
}

/// Convenience function for toBigEndian.
/// @returns a string just big enough to represent @a _val.
template <class _T>
inline std::string toCompactBigEndianString(_T _val)
{
	int i = 0;
	for (_T v = _val; v; ++i, v >>= 8) {}
	std::string ret(i, '\0');
	toBigEndian(_val, ret);
	return ret;
}


// Algorithms for string and string-like collections.

/// Escapes a string into the C-string representation.
/// @p _all if true will escape all characters, not just the unprintable ones.
std::string escaped(std::string const& _s, bool _all = true);

/// Determines the length of the common prefix of the two collections given.
/// @returns the number of elements both @a _t and @a _u share, in order, at the beginning.
/// @example commonPrefix("Hello world!", "Hello, world!") == 5
template <class _T, class _U>
unsigned commonPrefix(_T const& _t, _U const& _u)
{
	unsigned s = std::min<unsigned>(_t.size(), _u.size());
	for (unsigned i = 0;; ++i)
		if (i == s || _t[i] != _u[i])
			return i;
	return s;
}

/// Creates a random, printable, word.
std::string randomWord();


// General datatype convenience functions.

/// Determine bytes required to encode the given integer value. @returns 0 if @a _i is zero.
template <class _T>
inline unsigned bytesRequired(_T _i)
{
	unsigned i = 0;
	for (; _i != 0; ++i, _i >>= 8) {}
	return i;
}

/// Trims a given number of elements from the front of a collection.
/// Only works for POD element types.
template <class _T>
void trimFront(_T& _t, unsigned _elements)
{
	static_assert(std::is_pod<typename _T::value_type>::value, "");
	memmove(_t.data(), _t.data() + _elements, (_t.size() - _elements) * sizeof(_t[0]));
	_t.resize(_t.size() - _elements);
}

/// Pushes an element on to the front of a collection.
/// Only works for POD element types.
template <class _T, class _U>
void pushFront(_T& _t, _U _e)
{
	static_assert(std::is_pod<typename _T::value_type>::value, "");
	_t.push_back(_e);
	memmove(_t.data() + 1, _t.data(), (_t.size() - 1) * sizeof(_e));
	_t[0] = _e;
}

/// Concatenate two vectors of elements of POD types.
template <class _T>
inline std::vector<_T>& operator+=(std::vector<typename std::enable_if<std::is_pod<_T>::value, _T>::type>& _a, std::vector<_T> const& _b)
{
	auto s = _a.size();
	_a.resize(_a.size() + _b.size());
	memcpy(_a.data() + s, _b.data(), _b.size() * sizeof(_T));
	return _a;

}

/// Concatenate two vectors of elements.
template <class _T>
inline std::vector<_T>& operator+=(std::vector<typename std::enable_if<!std::is_pod<_T>::value, _T>::type>& _a, std::vector<_T> const& _b)
{
	_a.reserve(_a.size() + _b.size());
	for (auto& i: _b)
		_a.push_back(i);
	return _a;
}

/// Concatenate two vectors of elements.
template <class _T>
inline std::vector<_T> operator+(std::vector<_T> const& _a, std::vector<_T> const& _b)
{
	std::vector<_T> ret(_a);
	return ret += _b;
}

/// Merge two sets of elements.
template <class _T>
inline std::set<_T>& operator+=(std::set<_T>& _a, std::set<_T> const& _b)
{
	for (auto& i: _b)
		_a.insert(i);
	return _a;
}

/// Merge two sets of elements.
template <class _T>
inline std::set<_T> operator+(std::set<_T> const& _a, std::set<_T> const& _b)
{
	std::set<_T> ret(_a);
	return ret += _b;
}

/// Make normal string from fixed-length string.
std::string toString(string32 const& _s);

template<class T, class U>
std::vector<T> keysOf(std::map<T, U> const& _m)
{
	std::vector<T> ret;
	for (auto const& i: _m)
		ret.push_back(i.first);
	return ret;
}

}
