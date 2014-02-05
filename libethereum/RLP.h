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

static const byte c_rlpMaxLengthBytes = 8;
static const byte c_rlpDataImmLenStart = 0x80;
static const byte c_rlpListStart = 0xc0;

static const byte c_rlpDataImmLenCount = c_rlpListStart - c_rlpDataImmLenStart - c_rlpMaxLengthBytes;
static const byte c_rlpDataIndLenZero = c_rlpDataImmLenStart + c_rlpDataImmLenCount - 1;
static const byte c_rlpListImmLenCount = 256 - c_rlpListStart - c_rlpMaxLengthBytes;
static const byte c_rlpListIndLenZero = c_rlpListStart + c_rlpListImmLenCount - 1;

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

	/// The bare data of the RLP.
	bytesConstRef data() const { return m_data; }

	/// @returns true if the RLP is non-null.
	explicit operator bool() const { return !isNull(); }

	/// No value.
	bool isNull() const { return m_data.size() == 0; }

	/// Contains a zero-length string or zero-length list.
	bool isEmpty() const { return !isNull() && (m_data[0] == c_rlpDataImmLenStart || m_data[0] == c_rlpListStart); }

	/// String value.
	bool isData() const { return !isNull() && m_data[0] < c_rlpListStart; }

	/// List value.
	bool isList() const { return !isNull() && m_data[0] >= c_rlpListStart; }

	/// Integer value. Must not have a leading zero.
	bool isInt() const;

	/// @returns the number of items in the list, or zero if it isn't a list.
	uint itemCount() const { return isList() ? items() : 0; }
	uint itemCountStrict() const { if (!isList()) throw BadCast(); return items(); }

	/// @returns the number of bytes in the data, or zero if it isn't data.
	uint size() const { return isData() ? length() : 0; }
	uint sizeStrict() const { if (!isData()) throw BadCast(); return length(); }

	/// Equality operators; does best-effort conversion and checks for equality.
	bool operator==(char const* _s) const { return isData() && toString() == _s; }
	bool operator!=(char const* _s) const { return isData() && toString() != _s; }
	bool operator==(std::string const& _s) const { return isData() && toString() == _s; }
	bool operator!=(std::string const& _s) const { return isData() && toString() != _s; }
	template <unsigned _N> bool operator==(FixedHash<_N> const& _h) const { return isData() && toHash<_N>() == _h; }
	template <unsigned _N> bool operator!=(FixedHash<_N> const& _s) const { return isData() && toHash<_N>() != _s; }
	bool operator==(uint const& _i) const { return isInt() && toInt<uint>() == _i; }
	bool operator!=(uint const& _i) const { return isInt() && toInt<uint>() != _i; }
	bool operator==(u256 const& _i) const { return isInt() && toInt<u256>() == _i; }
	bool operator!=(u256 const& _i) const { return isInt() && toInt<u256>() != _i; }
	bool operator==(bigint const& _i) const { return isInt() && toInt<bigint>() == _i; }
	bool operator!=(bigint const& _i) const { return isInt() && toInt<bigint>() != _i; }

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
	bytes toBytes() const { if (!isData()) return bytes(); return bytes(payload().data(), payload().data() + length()); }
	/// Converts to bytearray. @returns the empty byte array if not a string.
	bytesConstRef toBytesConstRef() const { if (!isData()) return bytesConstRef(); return payload().cropped(0, length()); }
	/// Converts to string. @returns the empty string if not a string.
	std::string toString() const { if (!isData()) return std::string(); return payload().cropped(0, length()).toString(); }
	/// Converts to string. @throws BadCast if not a string.
	std::string toStringStrict() const { if (!isData()) throw BadCast(); return payload().cropped(0, length()).toString(); }

	template <class T> std::vector<T> toVector() const { std::vector<T> ret; if (isList()) { ret.reserve(itemCount()); for (auto const& i: *this) ret.push_back((T)i); } return ret; }
	template <class T, size_t N> std::array<T, N> toArray() const { std::array<T, N> ret; if (itemCount() != N) throw BadCast(); if (isList()) for (uint i = 0; i < N; ++i) ret[i] = (T)operator[](i); return ret; }

	/// Int conversion flags
	enum
	{
		AllowNonCanon = 1,
		ThrowOnFail = 4,
		FailIfTooBig = 8,
		Strict = ThrowOnFail | FailIfTooBig,
		LaisezFaire = AllowNonCanon
	};

	/// Converts to int of type given; if isString(), decodes as big-endian bytestream. @returns 0 if not an int or string.
	template <class _T = uint> _T toInt(int _flags = Strict) const
	{
		if ((!isInt() && !(_flags & AllowNonCanon)) || isList() || isNull())
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return 0;
		else {}

		auto p = payload();
		if (p.size() > intTraits<_T>::maxSize && (_flags & FailIfTooBig))
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return 0;
		else {}

		return fromBigEndian<_T>(p);
	}

	template <class _N> _N toHash(int _flags = Strict) const
	{
		if (!isData() || (length() > _N::size && (_flags & FailIfTooBig)))
			if (_flags & ThrowOnFail)
				throw BadCast();
			else
				return _N();
		else{}

		_N ret;
		size_t s = std::min((size_t)_N::size, (size_t)length());
		memcpy(ret.data() + _N::size - s, payload().data(), s);
		return ret;
	}

	/// Converts to RLPs collection object. Useful if you need random access to sub items or will iterate over multiple times.
	RLPs toList() const;

	/// @returns the data payload. Valid for all types.
	bytesConstRef payload() const { return isSingleByte() ? m_data.cropped(0, 1) : m_data.cropped(1 + lengthSize()); }

private:
	/// Single-byte data payload.
	bool isSingleByte() const { return !isNull() && m_data[0] < c_rlpDataImmLenStart; }

	/// @returns the theoretical size of this item; if it's a list, will require a deep traversal which could take a while.
	/// @note Under normal circumstances, is equivalent to m_data.size() - use that unless you know it won't work.
	uint actualSize() const;

	/// @returns the bytes used to encode the length of the data. Valid for all types.
	uint lengthSize() const { if (isData() && m_data[0] > c_rlpDataIndLenZero) return m_data[0] - c_rlpDataIndLenZero; if (isList() && m_data[0] > c_rlpListIndLenZero) return m_data[0] - c_rlpListIndLenZero; return 0; }

	/// @returns the size in bytes of the payload, as given by the RLP as opposed to as inferred from m_data.
	uint length() const;

	/// @returns the number of data items.
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

	~RLPStream() {}

	/// Append given datum to the byte stream.
	RLPStream& append(uint _s) { return append(bigint(_s)); }
	RLPStream& append(u160 _s) { return append(bigint(_s)); }
	RLPStream& append(u256 _s) { return append(bigint(_s)); }
	RLPStream& append(bigint _s);
	RLPStream& append(bytesConstRef _s, bool _compact = false);
	RLPStream& append(bytes const& _s) { return append(bytesConstRef(&_s)); }
	RLPStream& append(std::string const& _s) { return append(bytesConstRef(_s)); }
	RLPStream& append(char const* _s) { return append(std::string(_s)); }
	RLPStream& append(h160 _s, bool _compact = false) { return append(_s.ref(), _compact); }
	RLPStream& append(h256 _s, bool _compact = false) { return append(_s.ref(), _compact); }

	/// Appends an arbitrary RLP fragment - this *must* be a single item.
	RLPStream& append(RLP const& _rlp, uint _itemCount = 1) { return appendRaw(_rlp.data(), _itemCount); }

	/// Appends a sequence of data to the stream as a list.
	template <class _T> RLPStream& append(std::vector<_T> const& _s) { appendList(_s.size()); for (auto const& i: _s) append(i); return *this; }
	template <class _T, size_t S> RLPStream& append(std::array<_T, S> const& _s) { appendList(_s.size()); for (auto const& i: _s) append(i); return *this; }

	/// Appends a list.
	RLPStream& appendList(unsigned _items);
	RLPStream& appendList(bytesConstRef _rlp);
	RLPStream& appendList(bytes const& _rlp) { return appendList(&_rlp); }
	RLPStream& appendList(RLPStream const& _s) { return appendList(&_s.out()); }

	/// Appends raw (pre-serialised) RLP data. Use with caution.
	RLPStream& appendRaw(bytesConstRef _rlp, uint _itemCount = 1);
	RLPStream& appendRaw(bytes const& _rlp, uint _itemCount = 1) { return appendRaw(&_rlp, _itemCount); }

	/// Shift operators for appending data items.
	template <class T> RLPStream& operator<<(T _data) { return append(_data); }

	/// Clear the output stream so far.
	void clear() { m_out.clear(); m_listStack.clear(); }

	/// Read the byte stream.
	bytes const& out() const { assert(m_listStack.empty()); return m_out; }

	/// Swap the contents of the output stream out for some other byte array.
	void swapOut(bytes& _dest) { assert(m_listStack.empty()); swap(m_out, _dest); }

private:
	void noteAppended(uint _itemCount = 1);

	/// Push the node-type byte (using @a _base) along with the item count @a _count.
	/// @arg _count is number of characters for strings, data-bytes for ints, or items for lists.
	void pushCount(uint _count, byte _offset);

	/// Push an integer as a raw big-endian byte-stream.
	template <class _T> void pushInt(_T _i, uint _br)
	{
		m_out.resize(m_out.size() + _br);
		byte* b = &m_out.back();
		for (; _i; _i >>= 8)
			*(b--) = (byte)_i;
	}

	/// Determine bytes required to encode the given integer value. @returns 0 if @a _i is zero.
	template <class _T> static uint bytesRequired(_T _i)
	{
		uint i = 0;
		for (; _i != 0; ++i, _i >>= 8) {}
		return i;
	}

	/// Our output byte stream.
	bytes m_out;

	std::vector<std::pair<uint, uint>> m_listStack;
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
