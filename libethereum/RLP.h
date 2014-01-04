#pragma once

#include <exception>
#include <iostream>
#include <iomanip>
#include "vector_ref.h"
#include "Common.h"

namespace eth
{

class RLP;
typedef std::vector<RLP> RLPs;

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
	explicit RLP(bytes const& _d): m_data(const_cast<bytes*>(&_d)) {}	// a bit horrible, but we know we won't be altering the data. TODO: allow vector<T> const* to be passed to vector_ref<T const>.

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
	bool isEmpty() const { return m_data[0] == 0x40 || m_data[0] == 0x80; }

	/// String value.
	bool isString() const { assert(!isNull()); return m_data[0] >= 0x40 && m_data[0] < 0x80; }

	/// List value.
	bool isList() const { assert(!isNull()); return m_data[0] >= 0x80 && m_data[0] < 0xc0; }

	/// Integer value. Either isSlimInt() or isBigInt().
	bool isInt() const { assert(!isNull()); return m_data[0] < 0x40; }

	/// Fits into eth::uint type. Can use toInt() to read (as well as toBigInt() or toHugeInt() ).
	bool isSlimInt() const { assert(!isNull()); return m_data[0] < 0x20; }

	/// Fits only into eth::u256 type. Use only toFatInt() or toBigInt() to read.
	bool isFatInt() const { assert(!isNull()); return m_data[0] >= 0x20 && m_data[0] < 0x38; }

	/// Fits into eth::u256 type, though might fit into eth::uint type.
	bool isFixedInt() const { assert(!isNull()); return m_data[0] < 0x38; }

	/// Fits only into eth::bigint type. Use only toBigInt() to read.
	bool isBigInt() const { assert(!isNull()); return m_data[0] >= 0x38 && m_data[0] < 0x40; }

	/// @returns the number of items in the list, or zero if it isn't a list.
	uint itemCount() const { return isList() ? items() : 0; }
	uint itemCountStrict() const { if (!isList()) throw BadCast(); return items(); }

	/// @returns the number of characters in the string, or zero if it isn't a string.
	uint stringSize() const { return isString() ? items() : 0; }

	bool operator==(char const* _s) const { return isString() && toString() == _s; }
	bool operator==(std::string const& _s) const { return isString() && toString() == _s; }
	bool operator==(uint const& _i) const { return toSlimInt() == _i; }
	bool operator==(u256 const& _i) const { return toFatInt() == _i; }
	bool operator==(bigint const& _i) const { return toBigInt() == _i; }
	RLP operator[](uint _i) const
	{
		if (!isList() || itemCount() <= _i)
			return RLP();
		if (_i < m_lastIndex)
		{
			m_lastEnd = RLP(payload()).actualSize();
			m_lastItem = payload().cropped(m_lastEnd);
			m_lastIndex = 0;
		}
		for (; m_lastIndex < _i; ++m_lastIndex)
		{
			m_lastItem = payload().cropped(m_lastEnd);
			m_lastItem = m_lastItem.cropped(0, RLP(m_lastItem).actualSize());
			m_lastEnd += m_lastItem.size();
		}
		return RLP(m_lastItem);
	}

	typedef RLP element_type;

	class iterator
	{
		friend class RLP;

	public:
		typedef RLP value_type;
		typedef RLP element_type;

		iterator& operator++()
		{
			if (m_remaining)
			{
				m_lastItem.retarget(m_lastItem.next().data(), m_remaining);
				m_lastItem = m_lastItem.cropped(0, RLP(m_lastItem).actualSize());
				m_remaining -= std::min<uint>(m_remaining, m_lastItem.size());
			}
			return *this;
		}
		iterator operator++(int) { auto ret = *this; operator++(); return ret; }
		RLP operator*() const { return RLP(m_lastItem); }
		bool operator==(iterator const& _cmp) const { return m_lastItem == _cmp.m_lastItem; }
		bool operator!=(iterator const& _cmp) const { return !operator==(_cmp); }

	private:
		iterator() {}
		iterator(bytesConstRef _payload, bool _begin)
		{
			if (_begin)
			{
				m_lastItem = _payload.cropped(RLP(_payload).actualSize());
				m_remaining = _payload.size() - m_lastItem.size();
			}
			else
			{
				m_lastItem = _payload.cropped(m_lastItem.size());
				m_remaining = 0;
			}
		}
		uint m_remaining = 0;
		bytesConstRef m_lastItem;
	};
	friend class iterator;

	iterator begin() const { return iterator(payload(), true); }
	iterator end() const { return iterator(payload(), false); }

	explicit operator std::string() const { return toString(); }
	explicit operator RLPs() const { return toList(); }
	explicit operator uint() const { return toSlimInt(); }
	explicit operator u256() const { return toFatInt(); }
	explicit operator bigint() const { return toBigInt(); }

	std::string toString() const
	{
		if (!isString())
			return std::string();
		return payload().cropped(0, items()).toString();
	}

	template <class _T = uint> _T toInt() const
	{
		if (!isString() && !isInt())
			return 0;
		if (isDirectValueInt())
			return m_data[0];
		_T ret = 0;
		auto s = isInt() ? intSize() - lengthSize() : isString() ? items() : 0;
		uint o = lengthSize() + 1;
		for (uint i = 0; i < s; ++i)
			ret = (ret << 8) | m_data[i + o];
		return ret;
	}

	uint toSlimInt() const { return toInt<uint>(); }
	u256 toFatInt() const { return toInt<u256>(); }
	bigint toBigInt() const { return toInt<bigint>(); }
	uint toSlimIntStrict() const { if (!isSlimInt()) throw BadCast(); return toInt<uint>(); }
	u256 toFatIntStrict() const { if (!isFatInt() && !isSlimInt()) throw BadCast(); return toInt<u256>(); }
	bigint toBigIntStrict() const { if (!isInt()) throw BadCast(); return toInt<bigint>(); }
	uint toSlimIntFromString() const { if (!isString()) throw BadCast(); return toInt<uint>(); }
	u256 toFatIntFromString() const { if (!isString()) throw BadCast(); return toInt<u256>(); }
	bigint toBigIntFromString() const { if (!isString()) throw BadCast(); return toInt<bigint>(); }

	RLPs toList() const
	{
		RLPs ret;
		if (!isList())
			return ret;
		uint64_t c = items();
		for (uint64_t i = 0; i < c; ++i)
			ret.push_back(operator[](i));
		return ret;
	}

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

	uint actualSize() const
	{
		if (isNull())
			return 0;
		if (isInt())
			return 1 + intSize();
		if (isString())
			return payload().data() - m_data.data() + items();
		if (isList())
		{
			bytesConstRef d = payload();
			uint64_t c = items();
			for (uint64_t i = 0; i < c; ++i, d = d.cropped(RLP(d).actualSize())) {}
			return d.data() - m_data.data();
		}
		return 0;
	}

	uint intSize() const { return (!isInt() || isDirectValueInt()) ? 0 : isIndirectAddressedInt() ? lengthSize() + items() : (m_data[0] - 0x17); }

	uint lengthSize() const { auto n = (m_data[0] & 0x3f); return n > 0x37 ? n - 0x37 : 0; }
	uint items() const
	{
		auto n = (m_data[0] & 0x3f);
		if (n < 0x38)
			return n;
		uint ret = 0;
		for (int i = 0; i < n - 0x37; ++i)
			ret = (ret << 8) | m_data[i + 1];
		return ret;
	}

	bytesConstRef payload() const
	{
		assert(isString() || isList());
		auto n = (m_data[0] & 0x3f);
		return m_data.cropped(1 + (n < 0x38 ? 0 : (n - 0x37)));
	}

	bytesConstRef m_data;
	mutable uint m_lastIndex = (uint)-1;
	mutable uint m_lastEnd;
	mutable bytesConstRef m_lastItem;
};

struct RLPList { RLPList(uint _count): count(_count) {} uint count; };

class RLPStream
{
public:
	RLPStream() {}

	void append(uint _s) { appendNumeric(_s); }
	void append(u256 _s) { appendNumeric(_s); }
	void append(bigint _s) { appendNumeric(_s); }

	void append(std::string const& _s)
	{
		if (_s.size() < 0x38)
			m_out.push_back(_s.size() | 0x40);
		else
			pushCount(_s.size(), 0x40);
		uint os = m_out.size();
		m_out.resize(os + _s.size());
		memcpy(m_out.data() + os, _s.data(), _s.size());
	}

	void appendList(uint _count)
	{
		if (_count < 0x38)
			m_out.push_back(_count | 0x80);
		else
			pushCount(_count, 0x80);
	}

	RLPStream& operator<<(uint _i) { append(_i); return *this; }
	RLPStream& operator<<(u256 _i) { append(_i); return *this; }
	RLPStream& operator<<(bigint _i) { append(_i); return *this; }
	RLPStream& operator<<(char const* _s) { append(std::string(_s)); return *this; }
	RLPStream& operator<<(std::string const& _s) { append(_s); return *this; }
	RLPStream& operator<<(RLPList _l) { appendList(_l.count); return *this; }
	template <class _T> RLPStream& operator<<(std::vector<_T> const& _s) { appendList(_s.size()); for (auto const& i: _s) append(i); return *this; }

	bytes const& out() const { return m_out; }
	std::string str() const { return std::string((char const*)m_out.data(), (char const*)(m_out.data() + m_out.size())); }

private:
	void appendNumeric(uint _i)
	{
		if (_i < 0x18)
			m_out.push_back(_i);
		else
		{
			auto br = bytesRequired(_i);
			m_out.push_back(br + 0x17);	// max 8 bytes.
			pushInt(_i, br);
		}
	}

	void appendNumeric(u256 _i)
	{
		if (_i < 0x18)
			m_out.push_back((byte)_i);
		else
		{
			auto br = bytesRequired(_i);
			m_out.push_back(br + 0x17);	// max 8 bytes.
			pushInt(_i, br);
		}
	}

	void appendNumeric(bigint _i)
	{
		if (_i < 0x18)
			m_out.push_back((byte)_i);
		else
		{
			uint br = bytesRequired(_i);
			if (br <= 32)
				m_out.push_back(bytesRequired(_i) + 0x17);	// max 32 bytes.
			else
			{
				auto brbr = bytesRequired(br);
				m_out.push_back(0x37 + brbr);
				pushInt(br, brbr);
			}
			pushInt(_i, br);
		}
	}

	template <class _T> void pushInt(_T _i, uint _br)
	{
		m_out.resize(m_out.size() + _br);
		byte* b = &m_out.back();
		for (; _i; _i >>= 8)
			*(b--) = (byte)_i;
	}

	void pushCount(uint _count, byte _base)
	{
		auto br = bytesRequired(_count);
		m_out.push_back(br + 0x37 + _base);	// max 8 bytes.
		pushInt(_count, br);
	}

	template <class _T> static uint bytesRequired(_T _i)
	{
		_i >>= 8;
		uint i = 1;
		for (; _i != 0; ++i, _i >>= 8) {}
		return i;
	}

	bytes m_out;
};

template <class _T> void rlpListAux(RLPStream& _out, _T _t)
{
	_out << _t;
}

template <class _T, class ... _Ts> void rlpListAux(RLPStream& _out, _T _t, _Ts ... _ts)
{
	_out << _t;
	rlpListAux(_out, _ts...);
}

template <class _T> std::string rlp(_T _t)
{
	RLPStream out;
	out << _t;
	return out.str();
}

template <class _T> bytes rlpBytes(_T _t)
{
	RLPStream out;
	out << _t;
	return out.out();
}

template <class ... _Ts> std::string rlpList(_Ts ... _ts)
{
	RLPStream out;
	out << RLPList(sizeof ...(_Ts));
	rlpListAux(out, _ts...);
	return out.str();
}

template <class ... _Ts> bytes rlpListBytes(_Ts ... _ts)
{
	RLPStream out;
	out << RLPList(sizeof ...(_Ts));
	rlpListAux(out, _ts...);
	return out.out();
}

extern bytes RLPNull;

}

inline std::string escaped(std::string const& _s, bool _all = true)
{
	std::string ret;
	ret.reserve(_s.size());
	ret.push_back('"');
	for (auto i: _s)
		if (i == '"' && !_all)
			ret += "\\\"";
		else if (i == '\\' && !_all)
			ret += "\\\\";
		else if (i < ' ' || i > 127 || _all)
		{
			ret += "\\x";
			ret.push_back("0123456789abcdef"[(uint8_t)i / 16]);
			ret.push_back("0123456789abcdef"[(uint8_t)i % 16]);
		}
		else
			ret.push_back(i);
	ret.push_back('"');
	return ret;
}

inline std::ostream& operator<<(std::ostream& _out, eth::RLP _d)
{
	if (_d.isNull())
		_out << "null";
	else if (_d.isInt())
		_out << std::showbase << std::hex << std::nouppercase << _d.toBigInt();
	else if (_d.isString())
		_out << escaped(_d.toString(), true);
	else if (_d.isList())
	{
		_out << "[";
		int j = 0;
		for (auto i: _d)
			_out << (j++ ? ", " : " ") << i;
		_out << " ]";
	}

	return _out;
}

