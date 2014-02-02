/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 2 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file RLP.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 *
 * RLP (de-)serialisation.
 */

#pragma once

#include <vector>
#include <array>
#include <exception>
#include <iostream>
#include <iomanip>
#include "vector_ref.h"
#include "Common.h"

namespace eth
{

class RLP;
typedef std::vector<RLP> RLPs;

template <class _T> struct intTraits { static const uint maxSize = sizeof(_T); };
template <> struct intTraits<u160> { static const uint maxSize = 20; };
template <> struct intTraits<u256> { static const uint maxSize = 32; };
template <> struct intTraits<bigint> { static const uint maxSize = ~(uint)0; };

/**
 * @brief Class for interpreting Recursive Linear-Prefix Data.
 * @by Gav Wood, 2013
 *
 * Class for reading byte arrays of data in RLP format.
 */
class RLP
{
public:
	class BadCast: public std::exception {};

	/// Construct a null node.
	RLP() {}

	/// Construct a node of value given in the bytes.
	explicit RLP(bytesConstRef _d): m_data(_d) {}

	/// Construct a node of value given in the bytes.
	explicit RLP(bytes const& _d): m_data(&_d) {}

	/// Construct a node to read RLP data in the bytes given.
	RLP(byte const* _b, uint _s): m_data(bytesConstRef(_b, _s)) {}

	/// Construct a node to read RLP data in the string.
	explicit RLP(std::string const& _s): m_data(bytesConstRef((byte const*)_s.data(), _s.size())) {}

	bytesConstRef data() const { return m_data; }

	/// @returns true if the RLP is non-null.
	explicit operator bool() const { return !isNull(); }

	/// No value.
	bool isNull() const { return m_data.size() == 0; }

	/// Contains a zero-length string or zero-length list.
	bool isEmpty() const { return !isNull() && (m_data[0] == 0x40 || m_data[0] == 0x80); }

	/// String value.
	bool isString() const { return !isNull() && m_data[0] >= 0x40 && m_data[0] < 0x80; }

	/// List value.
	bool isList() const { return !isNull() && m_data[0] >= 0x80 && m_data[0] < 0xc0; }

	/// Integer value. Either isSlimInt(), isFatInt() or isBigInt().
	bool isInt() const { return !isNull() && m_data[0] < 0x40; }

	/// Fits into eth::uint type. Can use toSlimInt() to read (as well as toFatInt() or toBigInt() ).
	bool isSlimInt() const { return !isNull() && m_data[0] < 0x20; }

	/// Fits into eth::u256 or eth::bigint type. Use only toFatInt() or toBigInt() to read.
	bool isFatInt() const { return !isNull() && m_data[0] >= 0x20 && m_data[0] < 0x38; }

	/// Fits into eth::u256 type, though might fit into eth::uint type.
	bool isFixedInt() const { return !isNull() && m_data[0] < 0x38; }

	/// Fits only into eth::bigint type. Use only toBigInt() to read.
	bool isBigInt() const { return !isNull() && m_data[0] >= 0x38 && m_data[0] < 0x40; }

	/// @returns the number of items in the list, or zero if it isn't a list.
	uint itemCount() const { return isList() ? items() : 0; }
	uint itemCountStrict() const { if (!isList()) throw BadCast(); return items(); }

	/// @returns the number of characters in the string, or zero if it isn't a string.
	uint stringSize() const { return isString() ? items() : 0; }

	/// Equality operators; does best-effort conversion and checks for equality.
	bool operator==(char const* _s) const { return isString() && toString() == _s; }
	bool operator!=(char const* _s) const { return isString() && toString() != _s; }
	bool operator==(std::string const& _s) const { return isString() && toString() == _s; }
	bool operator!=(std::string const& _s) const { return isString() && toString() != _s; }
	template <unsigned _N> bool operator==(FixedHash<_N> const& _h) const { return isString() && toHash<_N>() == _h; }
	template <unsigned _N> bool operator!=(FixedHash<_N> const& _s) const { return isString() && toHash<_N>() != _s; }
	bool operator==(uint const& _i) const { return (isInt() || isString()) && toSlimInt() == _i; }
	bool operator!=(uint const& _i) const { return (isInt() || isString()) && toSlimInt() != _i; }
	bool operator==(u256 const& _i) const { return (isInt() || isString()) && toFatInt() == _i; }
	bool operator!=(u256 const& _i) const { return (isInt() || isString()) && toFatInt() != _i; }
	bool operator==(bigint const& _i) const { return (isInt() || isString()) && toBigInt() == _i; }
	bool operator!=(bigint const& _i) const { return (isInt() || isString()) && toBigInt() != _i; }

	/// Subscript operator.
	/// @returns the list item @a _i if isList() and @a _i < listItems(), or RLP() otherwise.
	/// @note if used to access items in ascending order, this is efficient.
	RLP operator[](uint _i) const;

	typedef RLP element_type;

	/// @brief Iterator class for iterating through items of RLP list.
	class iterator
	{
		friend class RLP;

	public:
		typedef RLP value_type;
		typedef RLP element_type;

		iterator& operator++();
		iterator operator++(int) { auto ret = *this; operator++(); return ret; }
		RLP operator*() const { return RLP(m_lastItem); }
		bool operator==(iterator const& _cmp) const { return m_lastItem == _cmp.m_lastItem; }
		bool operator!=(iterator const& _cmp) const { return !operator==(_cmp); }

	private:
		iterator() {}
		iterator(RLP const& _parent, bool _begin);

		uint m_remaining = 0;
		bytesConstRef m_lastItem;
	};

	/// @brief Iterator into beginning of sub-item list (valid only if we are a list).
	iterator begin() const { return iterator(*this, true); }

	/// @brief Iterator into end of sub-item list (valid only if we are a list).
	iterator end() const { return iterator(*this, false); }

	/// Best-effort conversion operators.
	explicit operator std::string() const { return toString(); }
	explicit operator RLPs() const { return toList(); }
	explicit operator byte() const { return toInt<byte>(); }
	explicit operator uint() const { return toInt<uint>(); }
	explicit operator u256() const { return toInt<u256>(); }
	explicit operator bigint() const { return toInt<bigint>(); }
	template <unsigned _N> explicit operator FixedHash<_N>() const { return toHash<FixedHash<_N>>(); }

	/// Converts to bytearray. @returns the empty byte array if not a string.
	bytes toBytes() const { if (!isString()) return bytes(); return bytes(payload().data(), payload().data() + items()); }
	/// Converts to bytearray. @returns the empty byte array if not a string.
	bytesConstRef toBytesConstRef() const { if (!isString()) return bytesConstRef(); return payload().cropped(0, items()); }
	/// Converts to string. @returns the empty string if not a string.
	std::string toString() const { if (!isString()) return std::string(); return payload().cropped(0, items()).toString(); }
	/// Converts to string. @throws BadCast if not a string.
	std::string toStringStrict() const { if (!isString()) throw BadCast(); return payload().cropped(0, items()).toString(); }

	template <class T> std::vector<T> toVector() const { std::vector<T> ret; if (isList()) { ret.reserve(itemCount()); for (auto const& i: *this) ret.push_back((T)i); } return ret; }
	template <class T, size_t N> std::array<T, N> toArray() const { std::array<T, N> ret; if (itemCount() != N) throw BadCast(); if (isList()) for (uint i = 0; i < N; ++i) ret[i] = (T)operator[](i); return ret; }

	/// Int conversion flags
	enum
	{
		AllowString = 1,
		AllowInt = 2,
		ThrowOnFail = 4,
		FailIfTooBig = 8,
		Strict = AllowString | AllowInt | ThrowOnFail | FailIfTooBig,
		StrictlyString = AllowString | ThrowOnFail | FailIfTooBig,
		StrictlyInt = AllowInt | ThrowOnFail | FailIfTooBig,
		LaisezFaire = AllowString | AllowInt
	};

	/// Converts to int of type given; if isString(), decodes as big-endian bytestream. @returns 0 if not an int or string.
	template <class _T = uint> _T toInt(int _flags = Strict) const
	{
		if ((isString() && !(_flags & AllowString)) || (isInt() && !(_flags & AllowInt)) || isList() || isNull())
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return 0;
		else {}

		if (isDirectValueInt())
			return m_data[0];

		auto s = isInt() ? intSize() - lengthSize() : isString() ? items() : 0;
		if (s > intTraits<_T>::maxSize && (_flags & FailIfTooBig))
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return 0;
		else {}

		_T ret = 0;
		uint o = lengthSize() + 1;
		for (uint i = 0; i < s; ++i)
			ret = (ret << 8) | m_data[i + o];
		return ret;
	}

	template <class _N> _N toHash(int _flags = Strict) const
	{
		if (!isString() || (items() > _N::size && (_flags & FailIfTooBig)))
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return _N();
		else{}

		_N ret;
		size_t s = std::min((size_t)_N::size, (size_t)items());
		memcpy(ret.data() + _N::size - s, payload().data(), s);
		return ret;
	}

	/// Converts to eth::uint. @see toInt()
	uint toSlimInt(int _flags = Strict) const { return toInt<uint>(_flags); }

	/// Converts to eth::u256. @see toInt()
	u256 toFatInt(int _flags = Strict) const { return toInt<u256>(_flags); }

	/// Converts to eth::bigint. @see toInt()
	bigint toBigInt(int _flags = Strict) const { return toInt<bigint>(_flags); }

	/// Converts to RLPs collection object. Useful if you need random access to sub items or will iterate over multiple times.
	RLPs toList() const;

	/// @returns the data payload. Valid for all types.
	bytesConstRef payload() const { auto n = (m_data[0] & 0x3f); return m_data.cropped(1 + (n < 0x38 ? 0 : (n - 0x37))); }

private:
	/// Direct value integer.
	bool isDirectValueInt() const { assert(!isNull()); return m_data[0] < 0x18; }

	/// Indirect-value integer.
	bool isIndirectValueInt() const { assert(!isNull()); return m_data[0] >= 0x18 && m_data[0] < 0x38; }

	/// Indirect addressed integer.
	bool isIndirectAddressedInt() const { assert(!isNull()); return m_data[0] < 0x40 && m_data[0] >= 0x38; }

	/// Direct-length string.
	bool isSmallString() const { assert(!isNull()); return m_data[0] >= 0x40 && m_data[0] < 0x78; }

	/// Direct-length list.
	bool isSmallList() const { assert(!isNull()); return m_data[0] >= 0x80 && m_data[0] < 0xb8; }

	/// @returns the theoretical size of this item; if it's a list, will require a deep traversal which could take a while.
	/// @note Under normal circumstances, is equivalent to m_data.size() - use that unless you know it won't work.
	uint actualSize() const;

	/// @returns the total additional bytes used to encode the integer. Includes the data-size and potentially the length-size. Returns 0 if not isInt().
	uint intSize() const { return (!isInt() || isDirectValueInt()) ? 0 : isIndirectAddressedInt() ? lengthSize() + items() : (m_data[0] - 0x17); }

	/// @returns the bytes used to encode the length of the data. Valid for all types.
	uint lengthSize() const { auto n = (m_data[0] & 0x3f); return n > 0x37 ? n - 0x37 : 0; }

	/// @returns the number of data items (bytes in the case of strings & ints, items in the case of lists). Valid for all types.
	uint items() const;

	/// Our byte data.
	bytesConstRef m_data;

	/// The list-indexing cache.
	mutable uint m_lastIndex = (uint)-1;
	mutable uint m_lastEnd = 0;
	mutable bytesConstRef m_lastItem;
};

/**
 * @brief Class for writing to an RLP bytestream.
 */
class RLPStream
{
public:
	/// Initializes empty RLPStream.
	RLPStream() {}

	/// Initializes the RLPStream as a list of @a _listItems items.
	explicit RLPStream(uint _listItems) { appendList(_listItems); }

	/// Append given data to the byte stream.
	RLPStream& append(uint _s);
	RLPStream& append(u160 _s);
	RLPStream& append(u256 _s);
	RLPStream& append(h160 _s, bool _compact = true) { return appendFixed(_s, _compact); }
	RLPStream& append(h256 _s, bool _compact = true) { return appendFixed(_s, _compact); }
	RLPStream& append(bigint _s);
	RLPStream& appendList(uint _count);
	RLPStream& appendString(bytesConstRef _s);
	RLPStream& appendString(bytes const& _s) { return appendString(bytesConstRef(&_s)); }
	RLPStream& appendString(std::string const& _s);
	RLPStream& appendRaw(bytesConstRef _rlp);
	RLPStream& appendRaw(bytes const& _rlp) { return appendRaw(&_rlp); }
	RLPStream& appendRaw(RLP const& _rlp) { return appendRaw(_rlp.data()); }

	/// Shift operators for appending data items.
	RLPStream& operator<<(uint _i) { return append(_i); }
	RLPStream& operator<<(u160 _i) { return append(_i); }
	RLPStream& operator<<(u256 _i) { return append(_i); }
	RLPStream& operator<<(h160 _i) { return append(_i); }
	RLPStream& operator<<(h256 _i) { return append(_i); }
	RLPStream& operator<<(bigint _i) { return append(_i); }
	RLPStream& operator<<(char const* _s) { return appendString(std::string(_s)); }
	RLPStream& operator<<(std::string const& _s) { return appendString(_s); }
	RLPStream& operator<<(RLP const& _i) { return appendRaw(_i); }
	template <class _T> RLPStream& operator<<(std::vector<_T> const& _s) { appendList(_s.size()); for (auto const& i: _s) append(i); return *this; }
	template <class _T, size_t S> RLPStream& operator<<(std::array<_T, S> const& _s) { appendList(_s.size()); for (auto const& i: _s) append(i); return *this; }

	void clear() { m_out.clear(); }

	/// Read the byte stream.
	bytes const& out() const { return m_out; }

	void swapOut(bytes& _dest) { swap(m_out, _dest); }

private:
	/// Push the node-type byte (using @a _base) along with the item count @a _count.
	/// @arg _count is number of characters for strings, data-bytes for ints, or items for lists.
	void pushCount(uint _count, byte _base);

	/// Push an integer as a raw big-endian byte-stream.
	template <class _T> void pushInt(_T _i, uint _br)
	{
		m_out.resize(m_out.size() + _br);
		byte* b = &m_out.back();
		for (; _i; _i >>= 8)
			*(b--) = (byte)_i;
	}

	template <unsigned _N>
	RLPStream& appendFixed(FixedHash<_N> const& _s, bool _compact)
	{
		uint s = _N;
		byte const* d = _s.data();
		if (_compact)
			for (unsigned i = 0; i < _N && !*d; ++i, --s, ++d) {}

		if (s < 0x38)
			m_out.push_back((byte)(s | 0x40));
		else
			pushCount(s, 0x40);
		uint os = m_out.size();
		m_out.resize(os + s);
		memcpy(m_out.data() + os, d, s);
		return *this;
	}

	/// Determine bytes required to encode the given integer value. @returns 0 if @a _i is zero.
	template <class _T> static uint bytesRequired(_T _i)
	{
		_i >>= 8;
		uint i = 1;
		for (; _i != 0; ++i, _i >>= 8) {}
		return i;
	}

	/// Our output byte stream.
	bytes m_out;
};

template <class _T> void rlpListAux(RLPStream& _out, _T _t) { _out << _t; }
template <class _T, class ... _Ts> void rlpListAux(RLPStream& _out, _T _t, _Ts ... _ts) { rlpListAux(_out << _t, _ts...); }

/// Export a single item in RLP format, returning a byte array.
template <class _T> bytes rlp(_T _t) { return (RLPStream() << _t).out(); }

/// Export a list of items in RLP format, returning a byte array.
inline bytes rlpList() { return RLPStream(0).out(); }
template <class ... _Ts> bytes rlpList(_Ts ... _ts)
{
	RLPStream out(sizeof ...(_Ts));
	rlpListAux(out, _ts...);
	return out.out();
}

/// The empty string in RLP format.
extern bytes RLPNull;

/// The empty list in RLP format.
extern bytes RLPEmptyList;

/// Human readable version of RLP.
std::ostream& operator<<(std::ostream& _out, eth::RLP const& _d);

}
