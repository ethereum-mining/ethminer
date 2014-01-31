/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
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

		uint t = 0;
		for (uint i = 0; i < _parent.itemCount(); ++i)
			t += _parent[i].actualSize();
		if (pl.size() != t)
			cout << _parent.itemCount() << " " << asHex(pl);
		assert(pl.size() == t);

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
	if (!isList() || itemCount() <= _i)
		return RLP();
	if (_i < m_lastIndex)
	{
		m_lastEnd = RLP(payload()).actualSize();
		m_lastItem = payload().cropped(0, m_lastEnd);
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

RLPs RLP::toList() const
{
	RLPs ret;
	if (!isList())
		return ret;
	uint64_t c = items();
	for (uint64_t i = 0; i < c; ++i)
		ret.push_back(operator[](i));
	return ret;
}

eth::uint RLP::actualSize() const
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

eth::uint RLP::items() const
{
	auto n = (m_data[0] & 0x3f);
	if (n < 0x38)
		return n;
	uint ret = 0;
	for (int i = 0; i < n - 0x37; ++i)
		ret = (ret << 8) | m_data[i + 1];
	return ret;
}

RLPStream& RLPStream::appendString(bytesConstRef _s)
{
	if (_s.size() < 0x38)
		m_out.push_back((byte)(_s.size() | 0x40));
	else
		pushCount(_s.size(), 0x40);
	uint os = m_out.size();
	m_out.resize(os + _s.size());
	memcpy(m_out.data() + os, _s.data(), _s.size());
	return *this;
}

RLPStream& RLPStream::appendString(string const& _s)
{
	if (_s.size() < 0x38)
		m_out.push_back((byte)(_s.size() | 0x40));
	else
		pushCount(_s.size(), 0x40);
	uint os = m_out.size();
	m_out.resize(os + _s.size());
	memcpy(m_out.data() + os, _s.data(), _s.size());
	return *this;
}

RLPStream& RLPStream::appendRaw(bytesConstRef _s)
{
	uint os = m_out.size();
	m_out.resize(os + _s.size());
	memcpy(m_out.data() + os, _s.data(), _s.size());
	return *this;
}

RLPStream& RLPStream::appendList(uint _count)
{
	if (_count < 0x38)
		m_out.push_back((byte)(_count | 0x80));
	else
		pushCount(_count, 0x80);
	return *this;
}

RLPStream& RLPStream::append(uint _i)
{
	if (_i < 0x18)
		m_out.push_back((byte)_i);
	else
	{
		auto br = bytesRequired(_i);
		m_out.push_back((byte)(br + 0x17));	// max 8 bytes.
		pushInt(_i, br);
	}
	return *this;
}

RLPStream& RLPStream::append(u160 _i)
{
	if (_i < 0x18)
		m_out.push_back((byte)_i);
	else
	{
		auto br = bytesRequired(_i);
		m_out.push_back((byte)(br + 0x17));	// max 8 bytes.
		pushInt(_i, br);
	}
	return *this;
}

RLPStream& RLPStream::append(u256 _i)
{
	if (_i < 0x18)
		m_out.push_back((byte)_i);
	else
	{
		auto br = bytesRequired(_i);
		m_out.push_back((byte)(br + 0x17));	// max 8 bytes.
		pushInt(_i, br);
	}
	return *this;
}

RLPStream& RLPStream::append(bigint _i)
{
	if (_i < 0x18)
		m_out.push_back((byte)_i);
	else
	{
		uint br = bytesRequired(_i);
		if (br <= 32)
			m_out.push_back((byte)(bytesRequired(_i) + 0x17));	// max 32 bytes.
		else
		{
			auto brbr = bytesRequired(br);
			m_out.push_back((byte)(0x37 + brbr));
			pushInt(br, brbr);
		}
		pushInt(_i, br);
	}
	return *this;
}

void RLPStream::pushCount(uint _count, byte _base)
{
	auto br = bytesRequired(_count);
	m_out.push_back((byte)(br + 0x37 + _base));	// max 8 bytes.
	pushInt(_count, br);
}

std::ostream& eth::operator<<(std::ostream& _out, eth::RLP const& _d)
{
	if (_d.isNull())
		_out << "null";
	else if (_d.isInt())
		_out << std::showbase << std::hex << std::nouppercase << _d.toBigInt(RLP::LaisezFaire) << dec;
	else if (_d.isString())
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
