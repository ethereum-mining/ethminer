#pragma once

#include <map>
#include "RLP.h"
#include "sha256.h"

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

inline std::string fromHex(bytes const& _hexVector, bool _terminated = false, int _begin = 0, int _end = -1)
{
	uint begin = _begin;
	uint end = _end < 0 ? _hexVector.size() + 1 + _end : _end;
	bool termed = _terminated;
	bool odd = (end - begin) % 2;

	std::string ret(1, ((termed ? 2 : 0) | (odd ? 1 : 0)) * 16);
	if (odd)
	{
		ret[0] |= _hexVector[0];
		++begin;
	}
	for (uint i = begin; i < end; i += 2)
		ret += _hexVector[i] * 16 + _hexVector[i + 1];
	return ret;
}

inline u256 hash256aux(HexMap const& _s, HexMap::const_iterator _begin, HexMap::const_iterator _end, unsigned _preLen)
{
	unsigned c = 0;
	for (auto i = _begin; i != _end; ++i, ++c) {}

	assert(c > 0);
	RLPStream rlp;
	if (c == 1)
	{
		// only one left - terminate with the pair.
		rlp << RLPList(2) << fromHex(_begin->first, true, _preLen) << _begin->second;
	}
	else
	{
		// if they all have the same next nibble, we also want a pair.

		// otherwise enumerate all 16+1 entries.
		for (auto i = 0; i < 16; ++i)
		{
		}
	}
	return sha256(rlp.out());
}

inline bytes toHex(std::string const& _s)
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

inline u256 hash256(StringMap const& _s)
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


