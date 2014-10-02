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
/** @file TrieCommon.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>

namespace dev
{
namespace eth
{

inline byte nibble(bytesConstRef _data, unsigned _i)
{
	return (_i & 1) ? (_data[_i / 2] & 15) : (_data[_i / 2] >> 4);
}

inline unsigned sharedNibbles(bytesConstRef _a, unsigned _ab, unsigned _ae, bytesConstRef _b, unsigned _bb, unsigned _be)
{
	unsigned ret = 0;
	for (unsigned ai = _ab, bi = _bb; ai < _ae && bi < _be && nibble(_a, ai) == nibble(_b, bi); ++ai, ++bi, ++ret) {}
	return ret;
}

struct NibbleSlice
{
	bytesConstRef data;
	unsigned offset;

	NibbleSlice(bytesConstRef _d = bytesConstRef(), unsigned _o = 0): data(_d), offset(_o) {}
	byte operator[](unsigned _index) const { return nibble(data, offset + _index); }
	unsigned size() const { return data.size() * 2 - offset; }
	NibbleSlice mid(unsigned _index) const { return NibbleSlice(data, offset + _index); }

	bool contains(NibbleSlice _k) const { return shared(_k) == _k.size(); }
	unsigned shared(NibbleSlice _k) const { return sharedNibbles(data, offset, offset + size(), _k.data, _k.offset, _k.offset + _k.size()); }
	bool operator==(NibbleSlice _k) const { return _k.size() == size() && shared(_k) == _k.size(); }
	bool operator!=(NibbleSlice _s) const { return !operator==(_s); }
};

inline std::ostream& operator<<(std::ostream& _out, NibbleSlice const& _m)
{
	for (unsigned i = 0; i < _m.size(); ++i)
		_out << std::hex << (int)_m[i] << std::dec;
	return _out;
}

inline bool isLeaf(RLP const& _twoItem)
{
	assert(_twoItem.isList() && _twoItem.itemCount() == 2);
	auto pl = _twoItem[0].payload();
	return (pl[0] & 0x20) != 0;
}

inline NibbleSlice keyOf(bytesConstRef _hpe)
{
	if (!_hpe.size())
		return NibbleSlice(_hpe, 0);
	if (_hpe[0] & 0x10)
		return NibbleSlice(_hpe, 1);
	else
		return NibbleSlice(_hpe, 2);
}

inline NibbleSlice keyOf(RLP const& _twoItem)
{
	return keyOf(_twoItem[0].payload());
}

byte uniqueInUse(RLP const& _orig, byte except);
std::string hexPrefixEncode(bytes const& _hexVector, bool _leaf = false, int _begin = 0, int _end = -1);
std::string hexPrefixEncode(bytesConstRef _data, bool _leaf, int _beginNibble, int _endNibble, unsigned _offset);
std::string hexPrefixEncode(bytesConstRef _d1, unsigned _o1, bytesConstRef _d2, unsigned _o2, bool _leaf);

inline std::string hexPrefixEncode(NibbleSlice _s, bool _leaf, int _begin = 0, int _end = -1)
{
	return hexPrefixEncode(_s.data, _leaf, _begin, _end, _s.offset);
}

inline std::string hexPrefixEncode(NibbleSlice _s1, NibbleSlice _s2, bool _leaf)
{
	return hexPrefixEncode(_s1.data, _s1.offset, _s2.data, _s2.offset, _leaf);
}

}
}
