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
/** @file Message.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Message.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

Message::Message(Envelope const& _e, Secret const& _s)
{
	try
	{
		bytes b;
		if (_s)
			decrypt(_s, &(_e.data()), b);
		populate(_s ? b : _e.data());
		m_to = KeyPair(_s).pub();
	}
	catch (...)	// Invalid secret? TODO: replace ... with InvalidSecret
	{
	}
}

void Message::populate(bytes const& _data)
{
	if (!_data.size())
		return;

	byte flags = _data[0];
	if (!!(flags & ContainsSignature) && _data.size() > sizeof(Signature) + 1)	// has a signature
	{
		bytesConstRef payload = bytesConstRef(&_data).cropped(sizeof(Signature) + 1);
		h256 h = sha3(payload);
		m_from = recover(*(Signature const*)&(_data[1]), h);
		m_payload = payload.toBytes();
	}
	else
		m_payload = bytesConstRef(&_data).cropped(1).toBytes();
}

Envelope Message::seal(Secret _from, Topic const& _topic, unsigned _ttl, unsigned _workToProve)
{
	Envelope ret(time(0) + _ttl, _ttl, _topic);

	bytes input(1 + m_payload.size());
	input[0] = 0;
	memcpy(input.data() + 1, m_payload.data(), m_payload.size());

	if (_from)		// needs a sig
	{
		input.resize(1 + m_payload.size() + sizeof(Signature));
		input[0] |= ContainsSignature;
		*(Signature*)&(input[1 + m_payload.size()]) = sign(_from, sha3(m_payload));
	}

	if (m_to)
		encrypt(m_to, &input, ret.m_data);
	else
		swap(ret.m_data, input);

	ret.proveWork(_workToProve);
	return ret;
}

Message Envelope::open(Secret const& _s) const
{
	return Message(*this, _s);
}

Message Envelope::open() const
{
	return Message(*this);
}

unsigned Envelope::workProved() const
{
	h256 d[2];
	d[0] = sha3NoNonce();
	d[1] = m_nonce;
	return dev::sha3(bytesConstRef(d[0].data(), 64)).firstBitSet();
}

void Envelope::proveWork(unsigned _ms)
{
	// PoW
	h256 d[2];
	d[0] = sha3NoNonce();
	uint32_t& n = *(uint32_t*)&(d[1][28]);
	unsigned bestBitSet = 0;
	bytesConstRef chuck(d[0].data(), 64);

	chrono::high_resolution_clock::time_point then = chrono::high_resolution_clock::now() + chrono::milliseconds(_ms);
	for (n = 0; chrono::high_resolution_clock::now() < then; )
		// do it rounds of 1024 for efficiency
		for (unsigned i = 0; i < 1024; ++i, ++n)
		{
			auto fbs = dev::sha3(chuck).firstBitSet();
			if (fbs > bestBitSet)
			{
				bestBitSet = fbs;
				m_nonce = n;
			}
		}
}
