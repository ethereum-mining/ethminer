#pragma once

#include <iostream>
#include "foreign.h"
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
	/// Construct a null node.
	RLP() {}

	/// Construct a node of value given in the bytes.
	explicit RLP(fConstBytes _d): m_data(_d) {}

	/// Construct a node to read RLP data in the bytes given.
	RLP(byte const* _b, uint _s): m_data(fConstBytes(_b, _s)) {}

	/// Construct a node to read RLP data in the string.
	explicit RLP(std::string const& _s): m_data(fConstBytes((byte const*)_s.data(), _s.size())) {}

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

	/// Fits only into eth::bigint type. Use only toBigInt() to read.
	bool isBigInt() const { assert(!isNull()); return m_data[0] >= 0x38 && m_data[0] < 0x40; }

	/// @returns the number of items in the list, or zero if it isn't a list.
	uint itemCount() const { return isList() ? items() : 0; }

	/// @returns the number of characters in the string, or zero if it isn't a string.
	uint stringSize() const { return isString() ? items() : 0; }

	std::string toString() const
	{
		if (!isString())
			return std::string();
		return payload().cropped(0, items()).toString();
	}

	template <class _T = uint> _T toInt(_T _def = 0) const
	{
		if (!isInt())
			return _def;
		if (isDirectValueInt())
			return m_data[0];
		_T ret = 0;
		auto s = intSize();
		for (uint i = 0; i < s; ++i)
			ret = (ret << 8) | m_data[i + 1];
		return ret;
	}

	uint toSlimInt(uint _def = 0) const { return toInt<uint>(_def); }
	u256 toFatInt(u256 _def = 0) const { return toInt<u256>(_def); }
	bigint toBigInt(bigint _def = 0) const { return toInt<bigint>(_def); }

	RLPs toList() const
	{
		RLPs ret;
		if (!isList())
			return ret;
		uint64_t c = items();
		fConstBytes d = payload();
		for (uint64_t i = 0; i < c; ++i, d = d.cropped(RLP(d).size()))
			ret.push_back(RLP(d));
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

	uint size() const
	{
		if (isInt())
			return 1 + intSize();
		if (isString())
			return payload().data() - m_data.data() + items();
		if (isList())
		{
			fConstBytes d = payload();
			uint64_t c = items();
			for (uint64_t i = 0; i < c; ++i, d = d.cropped(RLP(d).size())) {}
			return d.data() - m_data.data();
		}
		return 0;
	}

	uint intLengthSize() const { return isIndirectAddressedInt() ? m_data[0] - 0x37 : 0; }
	uint intSize() const { return (!isInt() || isDirectValueInt()) ? 0 : isIndirectAddressedInt() ? intLengthSize() + items() : (m_data[0] - 0x17); }

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

	fConstBytes payload() const
	{
		assert(isString() || isList());
		auto n = (m_data[0] & 0x3f);
		return m_data.cropped(1 + (n < 0x38 ? 0 : (n - 0x37)));
	}

	fConstBytes m_data;
};

}

inline std::ostream& operator<<(std::ostream& _out, eth::RLP _d)
{
	if (_d.isNull())
		_out << "null";
	else if (_d.isInt())
		_out << _d.toBigInt();
	else if (_d.isString())
		_out << "\"" << _d.toString() << "\"";
	else if (_d.isList())
	{
		_out << "[";
		int j = 0;
		for (auto i: _d.toList())
			_out << (j++ ? ", " : " ") << i;
		_out << " ]";
	}

	return _out;
}

