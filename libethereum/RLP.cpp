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
/** @file RLP.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "RLP.h"
using namespace std;
using namespace eth;

bytes eth::RLPNull = rlp("");
bytes eth::RLPEmptyList = rlpList();

RLP::iterator& RLP::iterator::operator++()
{
	if (m_remaining)
	{
		m_lastItem.retarget(m_lastItem.next().data(), m_remaining);
		m_lastItem = m_lastItem.cropped(0, RLP(m_lastItem).actualSize());
		m_remaining -= std::min<uint>(m_remaining, m_lastItem.size());
	}
	else
		m_lastItem.retarget(m_lastItem.next().data(), 0);
	return *this;
}

RLP::iterator::iterator(RLP const& _parent, bool _begin)
{
	if (_begin && _parent.isList())
	{
		auto pl = _parent.payload();
		m_lastItem = pl.cropped(0, RLP(pl).actualSize());
		m_remaining = pl.size() - m_lastItem.size();
	}
	else
	{
		m_lastItem = _parent.data().cropped(_parent.data().size());
		m_remaining = 0;
	}
}

RLP RLP::operator[](uint _i) const
{
	if (_i < m_lastIndex)
	{
		m_lastEnd = RLP(payload()).actualSize();
		m_lastItem = payload().cropped(0, m_lastEnd);
		m_lastIndex = 0;
	}
	for (; m_lastIndex < _i && m_lastItem.size(); ++m_lastIndex)
	{
		m_lastItem = payload().cropped(m_lastEnd);
		m_lastItem = m_lastItem.cropped(0, RLP(m_lastItem).actualSize());
		m_lastEnd += m_lastItem.size();
	}
	return RLP(m_lastItem);
}

RLPs RLP::toList() const
{
	RLPs ret;
	if (!isList())
		return ret;
	for (auto const& i: *this)
		ret.push_back(i);
	return ret;
}

eth::uint RLP::actualSize() const
{
	if (isNull())
		return 0;
	if (isSingleByte())
		return 1;
	if (isData() || isList())
		return payload().data() - m_data.data() + length();
	return 0;
}

bool RLP::isInt() const
{
	if (isNull())
		return false;
	byte n = m_data[0];
	if (n < c_rlpDataImmLenStart)
		return !!n;
	else if (n == c_rlpDataImmLenStart)
		return true;
	else if (n <= c_rlpDataIndLenZero)
	{
		if (m_data.size() <= 1)
			throw BadRLP();
		return m_data[1] != 0;
	}
	else if (n < c_rlpListStart)
	{
		if ((int)m_data.size() <= 1 + n - c_rlpDataIndLenZero)
			throw BadRLP();
		return m_data[1 + n - c_rlpDataIndLenZero] != 0;
	}
	else
		return false;
	return false;
}

eth::uint RLP::length() const
{
	if (isNull())
		return 0;
	uint ret = 0;
	byte n = m_data[0];
	if (n < c_rlpDataImmLenStart)
		return 1;
	else if (n <= c_rlpDataIndLenZero)
		return n - c_rlpDataImmLenStart;
	else if (n < c_rlpListStart)
	{
		if ((int)m_data.size() <= n - c_rlpDataIndLenZero)
			throw BadRLP();
		for (int i = 0; i < n - c_rlpDataIndLenZero; ++i)
			ret = (ret << 8) | m_data[i + 1];
	}
	else if (n <= c_rlpListIndLenZero)
		return n - c_rlpListStart;
	else
	{
		if ((int)m_data.size() <= n - c_rlpListIndLenZero)
			throw BadRLP();
		for (int i = 0; i < n - c_rlpListIndLenZero; ++i)
			ret = (ret << 8) | m_data[i + 1];
	}
	return ret;
}

eth::uint RLP::items() const
{
	if (isList())
	{
		bytesConstRef d = payload().cropped(0, length());
		eth::uint i = 0;
		for (; d.size(); ++i)
			d = d.cropped(RLP(d).actualSize());
		return i;
	}
	return 0;
}

RLPStream& RLPStream::appendRaw(bytesConstRef _s, uint _itemCount)
{
	uint os = m_out.size();
	m_out.resize(os + _s.size());
	memcpy(m_out.data() + os, _s.data(), _s.size());
	noteAppended(_itemCount);
	return *this;
}

void RLPStream::noteAppended(uint _itemCount)
{
	if (!_itemCount)
		return;
//	cdebug << "noteAppended(" << _itemCount << ")";
	while (m_listStack.size())
	{
		assert(m_listStack.back().first >= _itemCount);
		m_listStack.back().first -= _itemCount;
		if (m_listStack.back().first)
			break;
		else
		{
			auto p = m_listStack.back().second;
			m_listStack.pop_back();
			uint s = m_out.size() - p;		// list size
			auto brs = bytesRequired(s);
			uint encodeSize = s < c_rlpListImmLenCount ? 1 : (1 + brs);
//			cdebug << "s: " << s << ", p: " << p << ", m_out.size(): " << m_out.size() << ", encodeSize: " << encodeSize << " (br: " << brs << ")";
			auto os = m_out.size();
			m_out.resize(os + encodeSize);
			memmove(m_out.data() + p + encodeSize, m_out.data() + p, os - p);
			if (s < c_rlpListImmLenCount)
				m_out[p] = (byte)(c_rlpListStart + s);
			else
			{
				m_out[p] = (byte)(c_rlpListIndLenZero + brs);
				byte* b = &(m_out[p + brs]);
				for (; s; s >>= 8)
					*(b--) = (byte)s;
			}
		}
		_itemCount = 1;	// for all following iterations, we've effectively appended a single item only since we completed a list.
	}
}

RLPStream& RLPStream::appendList(uint _items)
{
//	cdebug << "appendList(" << _items << ")";
	if (_items)
		m_listStack.push_back(std::make_pair(_items, m_out.size()));
	else
		appendList(bytes());
	return *this;
}

RLPStream& RLPStream::appendList(bytesConstRef _rlp)
{
	if (_rlp.size() < c_rlpListImmLenCount)
		m_out.push_back((byte)(_rlp.size() + c_rlpListStart));
	else
		pushCount(_rlp.size(), c_rlpListIndLenZero);
	appendRaw(_rlp, 1);
	return *this;
}

RLPStream& RLPStream::append(bytesConstRef _s, bool _compact)
{
	uint s = _s.size();
	byte const* d = _s.data();
	if (_compact)
		for (unsigned i = 0; i < _s.size() && !*d; ++i, --s, ++d) {}

	if (s == 1 && *d < c_rlpDataImmLenStart)
		m_out.push_back(*d);
	else
	{
		if (s < c_rlpDataImmLenCount)
			m_out.push_back((byte)(s + c_rlpDataImmLenStart));
		else
			pushCount(s, c_rlpDataIndLenZero);
		appendRaw(bytesConstRef(d, s), 0);
	}
	noteAppended();
	return *this;
}

RLPStream& RLPStream::append(bigint _i)
{
	if (!_i)
		m_out.push_back(c_rlpDataImmLenStart);
	else if (_i < c_rlpDataImmLenStart)
		m_out.push_back((byte)_i);
	else
	{
		uint br = bytesRequired(_i);
		if (br < c_rlpDataImmLenCount)
			m_out.push_back((byte)(br + c_rlpDataImmLenStart));
		else
		{
			auto brbr = bytesRequired(br);
			m_out.push_back((byte)(c_rlpDataIndLenZero + brbr));
			pushInt(br, brbr);
		}
		pushInt(_i, br);
	}
	noteAppended();
	return *this;
}

void RLPStream::pushCount(uint _count, byte _base)
{
	auto br = bytesRequired(_count);
	m_out.push_back((byte)(br + _base));	// max 8 bytes.
	pushInt(_count, br);
}

std::ostream& eth::operator<<(std::ostream& _out, eth::RLP const& _d)
{
	if (_d.isNull())
		_out << "null";
	else if (_d.isInt())
		_out << std::showbase << std::hex << std::nouppercase << _d.toInt<bigint>(RLP::LaisezFaire) << dec;
	else if (_d.isData())
		_out << eth::escaped(_d.toString(), false);
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
