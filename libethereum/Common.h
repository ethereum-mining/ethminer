#pragma once

#include <cassert>
#include <sstream>
#include <cstdint>
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

template <class _T> std::string toString(_T const& _t) { std::ostringstream o; o << _t; return o.str(); }

template <class _T> inline std::string asHex(_T const& _data, int _w = 2)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << std::hex << std::setfill('0') << std::setw(_w) << (int)i;
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

inline bytes toHex(std::string const& _s)
{
	std::vector<uint8_t> ret;
	ret.reserve(_s.size() * 2);
	for (auto i: _s)
	{
		ret.push_back(i / 16);
		ret.push_back(i % 16);
	}
	return ret;
}

template <class _T>
inline std::string toBigEndianString(_T _val, uint _s)
{
	std::string ret;
	ret.resize(_s);
	for (uint i = 0; i < _s; ++i, _val >>= 8)
		ret[_s - 1 - i] = (char)(uint8_t)_val;
	return ret;
}

inline std::string toBigEndianString(u256 _val) { return toBigEndianString(_val, 32); }
inline std::string toBigEndianString(u160 _val) { return toBigEndianString(_val, 20); }

template <class _T>
inline std::string toCompactBigEndianString(_T _val)
{
	int i;
	for (i = 0; _val; ++i, _val >>= 8) {}
	return toBigEndianString(_val, i);
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

}
