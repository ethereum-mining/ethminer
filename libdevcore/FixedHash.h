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
#include <random>
#include <algorithm>
#include "CommonData.h"

namespace dev
{

extern std::mt19937_64 s_fixedHashEngine;

/// Fixed-size raw-byte array container type, with an API optimised for storing hashes.
/// Transparently converts to/from the corresponding arithmetic type; this will
/// assume the data contained in the hash is big-endian.
template <unsigned N>
class FixedHash
{
public:
	/// The corresponding arithmetic type.
	using Arith = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>;

	/// The size of the container.
	enum { size = N };

	/// A dummy flag to avoid accidental construction from pointer.
	enum ConstructFromPointerType { ConstructFromPointer };

	/// Method to convert from a string.
	enum ConstructFromStringType { FromHex, FromBinary };

	/// Method to convert from a string.
	enum ConstructFromHashType { AlignLeft, AlignRight };

	/// Construct an empty hash.
	FixedHash() { m_data.fill(0); }

	/// Construct from another hash, filling with zeroes or cropping as necessary.
	template <unsigned M> explicit FixedHash(FixedHash<M> const& _h, ConstructFromHashType _t = AlignLeft) { m_data.fill(0); unsigned c = std::min(M, N); for (unsigned i = 0; i < c; ++i) m_data[_t == AlignRight ? N - 1 - i : i] = _h[_t == AlignRight ? M - 1 - i : i]; }

	/// Convert from the corresponding arithmetic type.
	FixedHash(Arith const& _arith) { toBigEndian(_arith, m_data); }

	/// Explicitly construct, copying from a byte array.
	explicit FixedHash(bytes const& _b) { if (_b.size() == N) memcpy(m_data.data(), _b.data(), std::min<unsigned>(_b.size(), N)); }

	/// Explicitly construct, copying from a byte array.
	explicit FixedHash(bytesConstRef _b) { if (_b.size() == N) memcpy(m_data.data(), _b.data(), std::min<unsigned>(_b.size(), N)); }

	/// Explicitly construct, copying from a bytes in memory with given pointer.
	explicit FixedHash(byte const* _bs, ConstructFromPointerType) { memcpy(m_data.data(), _bs, N); }

	/// Explicitly construct, copying from a  string.
	explicit FixedHash(std::string const& _s, ConstructFromStringType _t = FromHex): FixedHash(_t == FromHex ? fromHex(_s) : dev::asBytes(_s)) {}

	/// Convert to arithmetic type.
	operator Arith() const { return fromBigEndian<Arith>(m_data); }

	/// @returns true iff this is the empty hash.
	operator bool() const { return ((Arith)*this) != 0; }

	// The obvious comparison operators.
	bool operator==(FixedHash const& _c) const { return m_data == _c.m_data; }
	bool operator!=(FixedHash const& _c) const { return m_data != _c.m_data; }
	bool operator<(FixedHash const& _c) const { return m_data < _c.m_data; }
	bool operator>=(FixedHash const& _c) const { return m_data >= _c.m_data; }

	// The obvious binary operators.
	FixedHash& operator^=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] ^= _c.m_data[i]; return *this; }
	FixedHash operator^(FixedHash const& _c) const { return FixedHash(*this) ^= _c; }
	FixedHash& operator|=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] |= _c.m_data[i]; return *this; }
	FixedHash operator|(FixedHash const& _c) const { return FixedHash(*this) |= _c; }
	FixedHash& operator&=(FixedHash const& _c) { for (unsigned i = 0; i < N; ++i) m_data[i] &= _c.m_data[i]; return *this; }
	FixedHash operator&(FixedHash const& _c) const { return FixedHash(*this) &= _c; }
	FixedHash& operator~() { for (unsigned i = 0; i < N; ++i) m_data[i] = ~m_data[i]; return *this; }

	/// @returns true if all bytes in @a _c are set in this object.
	bool contains(FixedHash const& _c) const { return (*this & _c) == _c; }

	/// @returns a particular byte from the hash.
	byte& operator[](unsigned _i) { return m_data[_i]; }
	/// @returns a particular byte from the hash.
	byte operator[](unsigned _i) const { return m_data[_i]; }

	/// @returns an abridged version of the hash as a user-readable hex string.
	std::string abridged() const { return toHex(ref().cropped(0, 4)) + "\342\200\246"; }

	/// @returns a mutable byte vector_ref to the object's data.
	bytesRef ref() { return bytesRef(m_data.data(), N); }

	/// @returns a constant byte vector_ref to the object's data.
	bytesConstRef ref() const { return bytesConstRef(m_data.data(), N); }

	/// @returns a mutable byte pointer to the object's data.
	byte* data() { return m_data.data(); }

	/// @returns a constant byte pointer to the object's data.
	byte const* data() const { return m_data.data(); }

	/// @returns a copy of the object's data as a byte vector.
	bytes asBytes() const { return bytes(data(), data() + N); }

	/// @returns a mutable reference to the object's data as an STL array.
	std::array<byte, N>& asArray() { return m_data; }

	/// @returns a constant reference to the object's data as an STL array.
	std::array<byte, N> const& asArray() const { return m_data; }

	/// @returns a randomly-valued hash
	template <class Engine>
	static FixedHash random(Engine& _eng)
	{
		FixedHash ret;
		for (auto& i: ret.m_data)
			i = std::uniform_int_distribution<uint16_t>(0, 255)(_eng);
		return ret;
	}

	static FixedHash random() { return random(s_fixedHashEngine); }

	/// A generic std::hash compatible function object.
	struct hash
	{
		/// Make a hash of the object's data.
		size_t operator()(FixedHash const& value) const
		{
			size_t h = 0;
			for (auto i: value.m_data)
				h = (h << (5 - h)) + i;
			return h;
		}
	};

	inline FixedHash<32> bloom() const
	{
		FixedHash<32> ret;
		for (auto i: m_data)
			ret[i / 8] |= 1 << (i % 8);
		return ret;
	}

	template <unsigned P, unsigned M> inline FixedHash& shiftBloom(FixedHash<M> const& _h)
	{
		return (*this |= _h.template nbloom<P, N>());
	}

	template <unsigned P, unsigned M> inline bool containsBloom(FixedHash<M> const& _h)
	{
		return contains(_h.template nbloom<P, N>());
	}

	template <unsigned P, unsigned M> inline FixedHash<M> nbloom() const
	{
		static const unsigned c_bloomBits = M * 8;
		unsigned mask = c_bloomBits - 1;
		unsigned bloomBytes = (dev::toLog2(c_bloomBits) + 7) / 8;
		FixedHash<M> ret;
		byte const* p = data();
		for (unsigned i = 0; i < P; ++i)
		{
			unsigned index = 0;
			for (unsigned j = 0; j < bloomBytes; ++j, ++p)
				index = (index << 8) | *p;
			index &= mask;
			ret[M - 1 - index / 8] |= (1 << (index % 8));
		}
		return ret;
	}

	/// Returns the index of the first bit set to one, or size() * 8 if no bits are set.
	inline unsigned firstBitSet() const
	{
		unsigned ret = 0;
		for (auto d: m_data)
			if (d)
				for (;; ++ret, d <<= 1)
					if (d & 0x80)
						return ret;
					else {}
			else
				ret += 8;
		return ret;
	}

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
	const uint64_t*data = (const uint64_t*)value.data();
	uint64_t hash = data[0];
	hash ^= data[1];
	hash ^= data[2];
	hash ^= data[3];
	return (size_t)hash;
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
using h520 = FixedHash<65>;
using h512 = FixedHash<64>;
using h256 = FixedHash<32>;
using h160 = FixedHash<20>;
using h128 = FixedHash<16>;
using h512s = std::vector<h512>;
using h256s = std::vector<h256>;
using h160s = std::vector<h160>;
using h256Set = std::set<h256>;
using h160Set = std::set<h160>;

/// Convert the given value into h160 (160-bit unsigned integer) using the right 20 bytes.
inline h160 right160(h256 const& _t)
{
	h160 ret;
	memcpy(ret.data(), _t.data() + 12, 20);
	return ret;
}

/// Convert the given value into h160 (160-bit unsigned integer) using the left 20 bytes.
inline h160 left160(h256 const& _t)
{
	h160 ret;
	memcpy(&ret[0], _t.data(), 20);
	return ret;
}

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
	/// Forward std::hash<dev::h256> to dev::h256::hash.
	template<> struct hash<dev::h256>: dev::h256::hash {};
}
