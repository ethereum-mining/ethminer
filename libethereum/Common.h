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

// way to many uint to size_t warnings in 32 bit build
#ifdef _M_IX86
#pragma warning(disable:4244)
#endif

#include <ctime>
#include <iomanip>
#include <chrono>
#include <array>
#include <map>
#include <set>
#include <string>
#include <cassert>
#include <sstream>
#include <cstdint>
#include <type_traits>
#include <boost/multiprecision/cpp_int.hpp>
#include "vector_ref.h"

namespace eth
{

// Binary data types.
using byte = uint8_t;
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

template <class T, class Out> inline void toBigEndian(T _val, Out& o_out);
template <class T, class In> inline T fromBigEndian(In const& _bytes);

template <unsigned N>
class FixedHash
{
	using Arith = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;

public:
	enum { size = N };

	FixedHash() { m_data.fill(0); }
	FixedHash(Arith const& _arith) { toBigEndian(_arith, m_data); }
	explicit FixedHash(bytes const& _b) { memcpy(m_data.data(), _b.data(), std::min<uint>(_b.size(), N)); }
	explicit FixedHash(byte const* _bs) { memcpy(m_data.data(), _bs, N); }

	operator Arith() const { return fromBigEndian<Arith>(m_data); }

	operator bool() const { return ((Arith)*this) != 0; }

	bool operator==(FixedHash const& _c) const { return m_data == _c.m_data; }
	bool operator!=(FixedHash const& _c) const { return m_data != _c.m_data; }
	bool operator<(FixedHash const& _c) const { return m_data < _c.m_data; }

	FixedHash& operator^=(FixedHash const& _c) { for (auto i = 0; i < N; ++i) m_data[i] ^= _c.m_data[i]; return *this; }
	FixedHash operator^(FixedHash const& _c) const { return FixedHash(*this) ^= _c; }
	FixedHash& operator|=(FixedHash const& _c) { for (auto i = 0; i < N; ++i) m_data[i] |= _c.m_data[i]; return *this; }
	FixedHash operator|(FixedHash const& _c) const { return FixedHash(*this) |= _c; }
	FixedHash& operator&=(FixedHash const& _c) { for (auto i = 0; i < N; ++i) m_data[i] &= _c.m_data[i]; return *this; }
	FixedHash operator&(FixedHash const& _c) const { return FixedHash(*this) &= _c; }
	FixedHash& operator~() { for (auto i = 0; i < N; ++i) m_data[i] = ~m_data[i]; return *this; }

	byte& operator[](unsigned _i) { return m_data[_i]; }
	byte operator[](unsigned _i) const { return m_data[_i]; }

	bytesRef ref() { return bytesRef(m_data.data(), N); }
	bytesConstRef ref() const { return bytesConstRef(m_data.data(), N); }

	byte* data() { return m_data.data(); }
	byte const* data() const { return m_data.data(); }

	bytes asBytes() const { return bytes(data(), data() + N); }
	std::array<byte, N>& asArray() { return m_data; }
	std::array<byte, N> const& asArray() const { return m_data; }

private:
	std::array<byte, N> m_data;
};

template <unsigned N>
inline std::ostream& operator<<(std::ostream& _out, FixedHash<N> const& _h)
{
	_out << std::noshowbase << std::hex << std::setfill('0');
	for (unsigned i = 0; i < N; ++i)
		_out << std::setw(2) << (int)_h[i];
	_out << std::dec;
	return _out;
}

using h256 = FixedHash<32>;
using h160 = FixedHash<20>;
using h256s = std::vector<h256>;
using h160s = std::vector<h160>;
using h256Set = std::set<h256>;
using h160Set = std::set<h160>;

using Secret = h256;
using Address = h160;
using Addresses = h160s;

// Map types.
using StringMap = std::map<std::string, std::string>;
using u256Map = std::map<u256, u256>;
using HexMap = std::map<bytes, std::string>;

// Null/Invalid values for convenience.
static const u256 Invalid256 = ~(u256)0;
static const bytes NullBytes;


/// Logging
class NullOutputStream
{
public:
	template <class T> NullOutputStream& operator<<(T const&) { return *this; }
};

extern std::map<std::type_info const*, bool> g_logOverride;
extern thread_local std::string t_logThreadName;

inline void setThreadName(std::string const& _n) { t_logThreadName = _n; }

struct LogChannel { static const char constexpr* name = "   "; static const int verbosity = 1; };
struct LeftChannel: public LogChannel { static const char constexpr* name = "<<<"; };
struct RightChannel: public LogChannel { static const char constexpr* name = ">>>"; };
struct WarnChannel: public LogChannel { static const char constexpr* name = "!!!"; static const int verbosity = 0; };
struct NoteChannel: public LogChannel { static const char constexpr* name = "***"; };
struct DebugChannel: public LogChannel { static const char constexpr*  name = "---"; static const int verbosity = 0; };

extern int g_logVerbosity;
extern std::function<void(std::string const&, char const*)> g_logPost;

void simpleDebugOut(std::string const&, char const* );

template <class Id, bool _AutoSpacing = true>
class LogOutputStream
{
public:
	LogOutputStream(bool _term = true)
	{
		std::type_info const* i = &typeid(Id);
		auto it = g_logOverride.find(i);
		if ((it != g_logOverride.end() && it->second == true) || (it == g_logOverride.end() && Id::verbosity <= g_logVerbosity))
		{
			time_t rawTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			sstr << Id::name << " [ " << rawTime /*put_time("%T", std::localtime(&rawTime)) */<< " | " << t_logThreadName << (_term ? " ] " : "");
		}
	}
	~LogOutputStream() { if (Id::verbosity <= g_logVerbosity) g_logPost(sstr.str(), Id::name); }
	template <class T> LogOutputStream& operator<<(T const& _t) { if (Id::verbosity <= g_logVerbosity) { if (_AutoSpacing && sstr.str().size() && sstr.str().back() != ' ') sstr << " "; sstr << _t; } return *this; }
	std::stringstream sstr;
};

// Dirties the global namespace, but oh so convenient...
#define cnote eth::LogOutputStream<eth::NoteChannel, true>()
#define cwarn eth::LogOutputStream<eth::WarnChannel, true>()

#define ndebug if (true) {} else eth::NullOutputStream()
#define nlog(X) if (true) {} else eth::NullOutputStream()
#define nslog(X) if (true) {} else eth::NullOutputStream()

#if NDEBUG
#define cdebug ndebug
#else
#define cdebug eth::LogOutputStream<eth::DebugChannel, true>()
#endif

#if NLOG
#define clog(X) nlog(X)
#define cslog(X) nslog(X)
#else
#define clog(X) eth::LogOutputStream<X, true>()
#define cslog(X) eth::LogOutputStream<X, false>()
#endif







/// User-friendly string representation of the amount _b in wei.
std::string formatBalance(u256 _b);

/// Converts arbitrary value to string representation using std::stringstream.
template <class _T>
std::string toString(_T const& _t)
{
	std::ostringstream o;
	o << _t;
	return o.str();
}

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

/// Convert a series of bytes to the corresponding string of hex duplets.
/// @param _w specifies the width of each of the elements. Defaults to two - enough to represent a byte.
/// @example asHex("A\x69") == "4169"
template <class _T>
std::string asHex(_T const& _data, int _w = 2)
{
	std::ostringstream ret;
	for (auto i: _data)
		ret << std::hex << std::setfill('0') << std::setw(_w) << (int)(typename std::make_unsigned<decltype(i)>::type)i;
	return ret.str();
}

/// Trims a given number of elements from the front of a collection.
/// Only works for POD element types.
template <class _T>
void trimFront(_T& _t, uint _elements)
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

/// Creates a random, printable, word.
std::string randomWord();

/// Escapes a string into the C-string representation.
/// @p _all if true will escape all characters, not just the unprintable ones.
std::string escaped(std::string const& _s, bool _all = true);

/// Converts a (printable) ASCII hex character into the correspnding integer value.
/// @example fromHex('A') == 10 && fromHex('f') == 15 && fromHex('5') == 5
int fromHex(char _i);

/// Converts a (printable) ASCII hex string into the corresponding byte stream.
/// @example fromUserHex("41626261") == asBytes("Abba")
bytes fromUserHex(std::string const& _s);

/// Converts a string into the big-endian base-16 stream of integers (NOT ASCII).
/// @example toHex("A")[0] == 4 && toHex("A")[1] == 1
bytes toHex(std::string const& _s);

/// Converts a templated integer value to the big-endian byte-stream represented on a templated collection.
/// The size of the collection object will be unchanged. If it is too small, it will not represent the
/// value properly, if too big then the additional elements will be zeroed out.
/// @a _Out will typically be either std::string or bytes.
/// @a _T will typically by uint, u160, u256 or bigint.
template <class _T, class _Out>
inline void toBigEndian(_T _val, _Out& o_out)
{
	for (auto i = o_out.size(); i-- != 0; _val >>= 8)
		o_out[i] = (typename _Out::value_type)(uint8_t)_val;
}

/// Converts a big-endian byte-stream represented on a templated collection to a templated integer value.
/// @a _In will typically be either std::string or bytes.
/// @a _T will typically by uint, u160, u256 or bigint.
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

/// Determines the length of the common prefix of the two collections given.
/// @returns the number of elements both @a _t and @a _u share, in order, at the beginning.
/// @example commonPrefix("Hello world!", "Hello, world!") == 5
template <class _T, class _U>
uint commonPrefix(_T const& _t, _U const& _u)
{
	uint s = std::min<uint>(_t.size(), _u.size());
	for (uint i = 0;; ++i)
		if (i == s || _t[i] != _u[i])
			return i;
	return s;
}

/// Convert the given value into h160 (160-bit unsigned integer) using the right 20 bytes.
inline h160 right160(h256 const& _t)
{
	h160 ret;
	memcpy(ret.data(), _t.data() + 10, 20);
	return ret;
}

/// Convert the given value into h160 (160-bit unsigned integer) using the left 20 bytes.
inline h160 left160(h256 const& _t)
{
	h160 ret;
	memcpy(&ret[0], _t.data(), 20);
	return ret;
}

/// Convert the given value into u160 (160-bit unsigned integer) by taking the lowest order 160-bits and discarding the rest.
inline u160 low160(u256 const& _t)
{
	return (u160)(_t & ((((u256)1) << 160) - 1));
}

inline u160 low160(bigint const& _t)
{
	return (u160)(_t & ((((bigint)1) << 160) - 1));
}

/// Convert the given value into u160 (160-bit unsigned integer) by taking the lowest order 160-bits and discarding the rest.
inline u160 high160(u256 const& _t)
{
	return (u160)(_t >> 96);
}


/// Concatenate two vectors of elements. _T must be POD.
template <class _T>
inline std::vector<_T>& operator+=(std::vector<typename std::enable_if<std::is_pod<_T>::value, _T>::type>& _a, std::vector<_T> const& _b)
{
	auto s = _a.size();
	_a.resize(_a.size() + _b.size());
	memcpy(_a.data() + s, _b.data(), _b.size() * sizeof(_T));
	return _a;

}

/// Concatenate two vectors of elements. _T must be POD.
template <class _T>
inline std::vector<_T> operator+(std::vector<typename std::enable_if<std::is_pod<_T>::value, _T>::type> const& _a, std::vector<_T> const& _b)
{
	std::vector<_T> ret(_a);
	return ret += _b;
}

/// SHA-3 convenience routines.
void sha3(bytesConstRef _input, bytesRef _output);
std::string sha3(std::string const& _input, bool _hex);
bytes sha3Bytes(bytesConstRef _input);
inline bytes sha3Bytes(std::string const& _input) { return sha3Bytes((std::string*)&_input); }
inline bytes sha3Bytes(bytes const& _input) { return sha3Bytes((bytes*)&_input); }
h256 sha3(bytesConstRef _input);
inline h256 sha3(bytes const& _input) { return sha3(bytesConstRef((bytes*)&_input)); }
inline h256 sha3(std::string const& _input) { return sha3(bytesConstRef(_input)); }

/// Get information concerning the currency denominations.
std::vector<std::pair<u256, std::string>> const& units();

/// Convert a private key into the public key equivalent.
/// @returns 0 if it's not a valid private key.
Address toAddress(h256 _private);

class KeyPair
{
public:
	KeyPair() {}
	KeyPair(Secret _k): m_secret(_k), m_address(toAddress(_k)) {}

	static KeyPair create();

	Secret const& secret() const { return m_secret; }
	Address const& address() const { return m_address; }

private:
	Secret m_secret;
	Address m_address;
};

}
