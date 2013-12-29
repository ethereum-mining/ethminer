#pragma once

#include <map>
#include "RLP.h"

namespace eth
{

using StringMap = std::map<std::string, std::string>;
using HexMap = std::map<bytes, std::string>;

/*
 * Hex-prefix Notation. First nibble has flags: oddness = 2^0 & termination = 2^1
 * [0,0,1,2,3,4,5]   0x10012345
 * [0,1,2,3,4,5]     0x00012345
 * [1,2,3,4,5]       0x112345
 * [0,0,1,2,3,4]     0x00001234
 * [0,1,2,3,4]       0x101234
 * [1,2,3,4]         0x001234
 * [0,0,1,2,3,4,5,T] 0x30012345
 * [0,0,1,2,3,4,T]   0x20001234
 * [0,1,2,3,4,5,T]   0x20012345
 * [1,2,3,4,5,T]     0x312345
 * [1,2,3,4,T]       0x201234
 */

std::string fromHex(bytes const& _hexVector, bool _forceTerminated = false)
{
	uint begin = 0;
	uint end = _hexVector.size();
	bool termed = _forceTerminated;
	bool odd = _hexVector.size() % 2;

	std::string ret(((termed ? 2 : 0) | (odd ? 1 : 0)) * 16, 1);
	if (odd)
	{
		ret[0] |= _hexVector[0];
		begin = 1;
	}
	else if (leadingZero)
	for (uint i = begin; i < end; i += 2)
		ret += _hexVector[i] * 16 + _hexVector[i + 1];
	return ret;
}

template <class _T, class ... _Ts> bytes encodeRLP(_T _t, _Ts ... _ts)
{

}

struct rlplist { rlplist(uint _count): count(_count) {} uint count; };

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
			m_out.push_back(_count | 0x40);
		else
			pushCount(_count, 0x40);
		uint os = m_out.size();
		m_out.resize(m_out.size() + _s.size());
		memcpy(m_out.data() + os, _s.data(), _s.size());
	}

	void appendList(uint _count)
	{
		if (_s.size() < 0x38)
			m_out.push_back(_count | 0x80);
		else
			pushCount(_count, 0x80);
	}

	RLPStream operator<<(uint _t) { append(_i); }
	RLPStream operator<<(u256 _t) { append(_i); }
	RLPStream operator<<(bigint _t) { append(_i); }
	RLPStream operator<<(std::string const& _s) { append(_s); }
	RLPStream operator<<(rlplist _l) { m_lists.push_back(_l.count); appendList(_l.count); }

private:
	void appendNumeric(uint _i)
	{
		if (_i < 0x18)
			m_out.push_back(_i);
		else
			m_out.push_back(bytesRequired(_i) + 0x17);	// max 8 bytes.
	}

	void appendNumeric(u256 _i)
	{
		if (_i < 0x18)
			m_out.push_back(_i);
		else
			m_out.push_back(bytesRequired(_i) + 0x17);	// max 32 bytes.
	}

	void appendNumeric(bigint _i)
	{
		if (_i < 0x18)
			m_out.push_back(_i);
		else
		{
			uint br = bytesRequired(_i);
			if (br <= 32)
				m_out.push_back(bytesRequired(_i) + 0x17);	// max 32 bytes.
			else
				m_out.push_back(0x37 + bytesRequired(br));
		}
		for (uint i = 0; i < )
		m_out.push_back()
	}

	void pushCount(uint _count, byte _base)
	{
		m_out.push_back(bytesRequired(_i) + 0x17);	// max 8 bytes.
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

template <> bytes encodeRLP(_T _t)
{

}

u256 hash256aux(HexMap const& _s, HexMap::const_iterator _begin, HexMap::const_iterator _end, unsigned _preLen)
{
	unsigned c = 0;
	for (auto i = _begin; i != _end; ++i, ++c) {}

	assert(c > 0);
	if (c == 1)
		return sha256(encodeRLP());



	for (auto i = 0; i < 16; ++i)
	{
	}
}

bytes toHex(std::string const& _s)
{
	std::vector<uint8_t> ret(_s.size() * 2 + 1);
	for (auto i: _s)
	{
		ret.push_back(i / 16);
		ret.push_back(i % 16);
	}
	ret.push_back(16);
	return ret;
}

u256 hash256(StringMap const& _s)
{
	// build patricia tree.
	if (_s.empty())
		return 0;
	HexMap hexMap;
	for (auto const& i: _s)
		hexMap[toHex(i.first)] = i.second;
	return hash256aux(hexMap, hexMap.cbegin(), hexMap.cend(), 0);
}

/**
 * @brief Merkle Patricia Tree: a modifed base-16 Radix tree.
 */
class PatriciaTree
{
public:
	PatriciaTree() {}
	~PatriciaTree() {}

	void fromRLP(RLP const& _data);
	std::string toRLP();

private:

};

}


