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

inline std::string hexPrefixEncode(bytes const& _hexVector, bool _terminated = false, int _begin = 0, int _end = -1)
{
	uint begin = _begin;
	uint end = _end < 0 ? _hexVector.size() + 1 + _end : _end;
	bool termed = _terminated;
	bool odd = (end - begin) % 2;

	std::string ret(1, ((termed ? 2 : 0) | (odd ? 1 : 0)) * 16);
	if (odd)
	{
		ret[0] |= _hexVector[begin];
		++begin;
	}
	for (uint i = begin; i < end; i += 2)
		ret += _hexVector[i] * 16 + _hexVector[i + 1];
	return ret;
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

inline std::string toBigEndianString(u256 _val)
{
	std::string ret;
	ret.resize(32);
	for (int i = 0; i <32; ++i, _val >>= 8)
		ret[31 - i] = (char)(uint8_t)_val;
	return ret;
}

inline u256 hash256aux(HexMap const& _s, HexMap::const_iterator _begin, HexMap::const_iterator _end, unsigned _preLen)
{
	RLPStream rlp;
	if (_begin == _end)
	{
		rlp << "";	// NULL
	}
	else if (std::next(_begin) == _end)
	{
		// only one left - terminate with the pair.
		rlp << RLPList(2) << hexPrefixEncode(_begin->first, true, _preLen) << _begin->second;
	}
	else
	{
		// find the number of common prefix nibbles shared
		// i.e. the minimum number of nibbles shared at the beginning between the first hex string and each successive.
		uint sharedPre = (uint)-1;
		uint c = 0;
		for (auto i = std::next(_begin); i != _end && sharedPre; ++i, ++c)
		{
			uint x = std::min(sharedPre, std::min(_begin->first.size(), i->first.size()));
			uint shared = _preLen;
			for (; shared < x && _begin->first[shared] == i->first[shared]; ++shared) {}
			sharedPre = std::min(shared, sharedPre);
		}
		if (sharedPre > _preLen)
		{
			// if they all have the same next nibble, we also want a pair.
			rlp << RLPList(2) << hexPrefixEncode(_begin->first, false, _preLen, sharedPre) << toBigEndianString(hash256aux(_s, _begin, _end, sharedPre));
		}
		else
		{
			// otherwise enumerate all 16+1 entries.
			rlp << RLPList(17);
			auto b = _begin;
			if (_preLen == b->first.size())
				++b;
			for (auto i = 0; i < 16; ++i)
			{
				auto n = b;
				for (; n != _end && n->first[_preLen] == i; ++n) {}
				if (b == n)
					rlp << "";
				else
					rlp << toBigEndianString(hash256aux(_s, b, n, _preLen + 1));
				b = n;
			}
			if (_preLen == _begin->first.size())
				rlp << _begin->second;
			else
				rlp << "";
		}
	}
//	std::cout << std::hex << sha256(rlp.out()) << ": " << asHex(rlp.out()) << ": " << RLP(rlp.out()) << std::endl;
	return sha256(rlp.out());
}

inline u256 hash256(StringMap const& _s)
{
	// build patricia tree.
	if (_s.empty())
		return 0;
	HexMap hexMap;
	for (auto i = _s.rbegin(); i != _s.rend(); ++i)
		hexMap[toHex(i->first)] = i->second;
//	for (auto const& i: _s)
//		hexMap[toHex(i.first)] = i.second;
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


