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
/** @file FixedHash.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * The FixedHash fixed-size "hash" container type.
 */

#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <random>
#include "CommonData.h"

namespace dev
{

extern std::random_device s_fixedHashEngine;

/// Fixed-size raw-byte array container type, with an API optimised for storing hashes.
/// Transparently converts to/from the corresponding arithmetic type; this will
/// assume the data contained in the hash is big-endian.
template <unsigned N>
class FixedHash
{
public:

#if defined(_WIN32)
	const char* k_ellipsis = "...";
#else
	const char* k_ellipsis = "\342\200\246";
#endif

	/// The corresponding arithmetic type.
	using Arith = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;

	/// The size of the container.
	enum { size = N };

	/// A dummy flag to avoid accidental construction from pointer.
	enum ConstructFromPointerType { ConstructFromPointer };

	/// Method to convert from a string.
	enum ConstructFromHashType { AlignLeft, AlignRight, FailIfDifferent };

	/// Construct an empty hash.
	FixedHash() { m_data.fill(0); }

	/// Construct from another hash, filling with zeroes or cropping as necessary.
	template <unsigned M> explicit FixedHash(FixedHash<M> const& _h, ConstructFromHashType _t = AlignLeft) { m_data.fill(0); unsigned c = std::min(M, N); for (unsigned i = 0; i < c; ++i) m_data[_t == AlignRight ? N - 1 - i : i] = _h[_t == AlignRight ? M - 1 - i : i]; }

	/// Convert from the corresponding arithmetic type.
	FixedHash(Arith const& _arith) { toBigEndian(_arith, m_data); }

	/// Convert from unsigned
	explicit FixedHash(unsigned _u) { toBigEndian(_u, m_data); }

	/// Explicitly construct, copying from a byte array.
	explicit FixedHash(bytes const& _b, ConstructFromHashType _t = FailIfDifferent) { if (_b.size() == N) memcpy(m_data.data(), _b.data(), std::min<unsigned>(_b.size(), N)); else { m_data.fill(0); if (_t != FailIfDifferent) { auto c = std::min<unsigned>(_b.size(), N); for (unsigned i = 0; i < c; ++i) m_data[_t == AlignRight ? N - 1 - i : i] = _b[_t == AlignRight ? _b.size() - 1 - i : i]; } } }

	/// Explicitly construct, copying from a byte array.
	explicit FixedHash(bytesConstRef _b, ConstructFromHashType _t = FailIfDifferent) { if (_b.size() == N) memcpy(m_data.data(), _b.data(), std::min<unsigned>(_b.size(), N)); else { m_data.fill(0); if (_t != FailIfDifferent) { auto c = std::min<unsigned>(_b.size(), N); for (unsigned i = 0; i < c; ++i) m_data[_t == AlignRight ? N - 1 - i : i] = _b[_t == AlignRight ? _b.size() - 1 - i : i]; } } }

	/// Explicitly construct, copying from a bytes in memory with given pointer.
	explicit FixedHash(byte const* _bs, ConstructFromPointerType) { memcpy(m_data.data(), _bs, N); }

	/// Explicitly construct, copying from a  string.
	explicit FixedHash(std::string const& _s): FixedHash(fromHex(_s, WhenError::Throw), FailIfDifferent) {}

	/// Convert to arithmetic type.
	operator Arith() const { return fromBigEndian<Arith>(m_data); }

	/// @returns true iff this is the empty hash.
	explicit operator bool() const { return std::any_of(m_data.begin(), m_data.end(), [](byte _b) { return _b != 0; }); }

	// The obvious comparison operators.
	bool operator==(FixedHash const& _c) const { return m_data == _c.m_data; }
	bool operator!=(FixedHash const& _c) const { return m_data != _c.m_data; }
	bool operator<(FixedHash const& _c) const { for (unsigned i = 0; i < N; ++i) if (m_data[i] < _c.m_data[i]) return true; else if (m_data[i] > _c.m_data[i]) return false; return false; }
	bool operator>=(FixedHash const& _c) const { return !operator<(_c); }
	bool operator<=(FixedHash const& _c) const { return operator==(_c) || operator<(_c); }
	bool operator>(FixedHash const& _c) const { return !operator<=(_c); }

	// The obvious binary operators.
	FixedHash& operator^=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] ^= _c.m_data[i]; return *this; }
	FixedHash operator^(FixedHash const& _c) const { return FixedHash(*this) ^= _c; }
	FixedHash& operator|=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] |= _c.m_data[i]; return *this; }
	FixedHash operator|(FixedHash const& _c) const { return FixedHash(*this) |= _c; }
	FixedHash& operator&=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] &= _c.m_data[i]; return *this; }
	FixedHash operator&(FixedHash const& _c) const { return FixedHash(*this) &= _c; }
	FixedHash operator~() const { FixedHash ret; for (unsigned i = 0; i < N; ++i) ret[i] = ~m_data[i]; return ret; }

	// Big-endian increment.
	FixedHash& operator++() { for (unsigned i = size; i > 0 && !++m_data[--i]; ) {} return *this; }

	/// @returns a particular byte from the hash.
	byte& operator[](unsigned _i) { return m_data[_i]; }
	/// @returns a particular byte from the hash.
	byte operator[](unsigned _i) const { return m_data[_i]; }

	/// @returns an abridged version of the hash as a user-readable hex string.

	std::string abridged() const { return toHex(ref().cropped(0, 4)) + k_ellipsis; }

	/// @returns the hash as a user-readable hex string.
	std::string hex() const { return toHex(ref()); }

	/// @returns a mutable byte vector_ref to the object's data.
	bytesRef ref() { return bytesRef(m_data.data(), N); }

	/// @returns a constant byte vector_ref to the object's data.
	bytesConstRef ref() const { return bytesConstRef(m_data.data(), N); }

	/// @returns a mutable byte pointer to the object's data.
	byte* data() { return m_data.data(); }

	/// @returns a constant byte pointer to the object's data.
	byte const* data() const { return m_data.data(); }

	/// Populate with random data.
	template <class Engine>
	void randomize(Engine& _eng)
	{
		for (auto& i: m_data)
			i = (uint8_t)std::uniform_int_distribution<uint16_t>(0, 255)(_eng);
	}

	/// @returns a random valued object.
	static FixedHash random() { FixedHash ret; ret.randomize(s_fixedHashEngine); return ret; }

	struct hash
	{
		/// Make a hash of the object's data.
		size_t operator()(FixedHash const& _value) const { return boost::hash_range(_value.m_data.cbegin(), _value.m_data.cend()); }
	};

	void clear() { m_data.fill(0); }

private:
	std::array<byte, N> m_data;		///< The binary data.
};

/// Fast equality operator for h256.
template<> inline bool FixedHash<32>::operator==(FixedHash<32> const& _other) const
{
	const uint64_t* hash1 = (const uint64_t*)data();
	const uint64_t* hash2 = (const uint64_t*)_other.data();
	return (hash1[0] == hash2[0]) && (hash1[1] == hash2[1]) && (hash1[2] == hash2[2]) && (hash1[3] == hash2[3]);
}

/// Fast std::hash compatible hash function object for h256.
template<> inline size_t FixedHash<32>::hash::operator()(FixedHash<32> const& value) const
{
	uint64_t const* data = reinterpret_cast<uint64_t const*>(value.data());
	return boost::hash_range(data, data + 4);
}

/// Stream I/O for the FixedHash class.
template <unsigned N>
inline std::ostream& operator<<(std::ostream& _out, FixedHash<N> const& _h)
{
	_out << std::noshowbase << std::hex << std::setfill('0');
	for (unsigned i = 0; i < N; ++i)
		_out << std::setw(2) << (int)_h[i];
	_out << std::dec;
	return _out;
}

// Common types of FixedHash.
using h2048 = FixedHash<256>;
using h1024 = FixedHash<128>;
using h520 = FixedHash<65>;
using h512 = FixedHash<64>;
using h256 = FixedHash<32>;
using h160 = FixedHash<20>;
using h128 = FixedHash<16>;
using h64 = FixedHash<8>;
using h512s = std::vector<h512>;
using h256s = std::vector<h256>;
using h160s = std::vector<h160>;

inline std::string toString(h256s const& _bs)
{
	std::ostringstream out;
	out << "[ ";
	for (auto i: _bs)
		out << i.abridged() << ", ";
	out << "]";
	return out.str();
}

}

namespace std
{
	/// Forward std::hash<dev::FixedHash> to dev::FixedHash::hash.
	template<> struct hash<dev::h64>: dev::h64::hash {};
	template<> struct hash<dev::h128>: dev::h128::hash {};
	template<> struct hash<dev::h160>: dev::h160::hash {};
	template<> struct hash<dev::h256>: dev::h256::hash {};
	template<> struct hash<dev::h512>: dev::h512::hash {};
}
