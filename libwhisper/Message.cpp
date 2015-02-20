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

Message::Message(Envelope const& _e, FullTopic const& _fk, Secret const& _s)
{
	try
	{
		bytes b;
		if (_s)
			if (!decrypt(_s, &(_e.data()), b))
				return;
			else{}
		else
		{
			// public - need to get the key through combining with the topic/topicIndex we know.
			unsigned topicIndex = 0;
			Secret topicSecret;

			// determine topicSecret/topicIndex from knowledge of the collapsed topics (which give the order) and our full-size filter topic.
			CollapsedTopic knownTopic = collapse(_fk);
			for (unsigned ti = 0; ti < _fk.size() && !topicSecret; ++ti)
				for (unsigned i = 0; i < _e.topic().size(); ++i)
					if (_e.topic()[i] == knownTopic[ti])
					{
						topicSecret = _fk[ti];
						topicIndex = i;
						break;
					}

			if (_e.data().size() < _e.topic().size() * 32)
				return;

			// get key from decrypted topic key: just xor
			h256 tk = h256(bytesConstRef(&(_e.data())).cropped(32 * topicIndex, 32));
			bytesConstRef cipherText = bytesConstRef(&(_e.data())).cropped(32 * _e.topic().size());
			cnote << "Decrypting(" << topicIndex << "): " << topicSecret << tk << (topicSecret ^ tk) << toHex(cipherText);
			if (!decryptSym(topicSecret ^ tk, cipherText, b))
				return;
			cnote << "Got: " << toHex(b);
		}

		if (populate(b))
			if (_s)
				m_to = KeyPair(_s).pub();
	}
	catch (...)	// Invalid secret? TODO: replace ... with InvalidSecret
	{
	}
}

bool Message::populate(bytes const& _data)
{
	if (!_data.size())
		return false;

	byte flags = _data[0];
	if (!!(flags & ContainsSignature) && _data.size() >= sizeof(Signature) + 1)	// has a signature
	{
		bytesConstRef payload = bytesConstRef(&_data).cropped(1, _data.size() - sizeof(Signature) - 1);
		h256 h = sha3(payload);
		Signature const& sig = *(Signature const*)&(_data[1 + payload.size()]);
		m_from = recover(sig, h);
		if (!m_from)
			return false;
		m_payload = payload.toBytes();
	}
	else
		m_payload = bytesConstRef(&_data).cropped(1).toBytes();
	return true;
}

Envelope Message::seal(Secret _from, FullTopic const& _fullTopic, unsigned _ttl, unsigned _workToProve) const
{
	CollapsedTopic topic = collapse(_fullTopic);
	Envelope ret(time(0) + _ttl, _ttl, topic);

	bytes input(1 + m_payload.size());
	input[0] = 0;
	memcpy(input.data() + 1, m_payload.data(), m_payload.size());

	if (_from)		// needs a sig
	{
		input.resize(1 + m_payload.size() + sizeof(Signature));
		input[0] |= ContainsSignature;
		*(Signature*)&(input[1 + m_payload.size()]) = sign(_from, sha3(m_payload));
		// If this fails, the something is wrong with the sign-recover round-trip.
		assert(recover(*(Signature*)&(input[1 + m_payload.size()]), sha3(m_payload)) == KeyPair(_from).pub());
	}

	if (m_to)
		encrypt(m_to, &input, ret.m_data);
	else
	{
		// create the shared secret and encrypt
		Secret s = Secret::random();
		for (h256 const& t: _fullTopic)
			ret.m_data += (t ^ s).asBytes();
		bytes d;
		encryptSym(s, &input, d);
		ret.m_data += d;

		for (unsigned i = 0; i < _fullTopic.size(); ++i)
		{
			bytes b;
			h256 tk = h256(bytesConstRef(&(ret.m_data)).cropped(32 * i, 32));
			bytesConstRef cipherText = bytesConstRef(&(ret.m_data)).cropped(32 * ret.topic().size());
			cnote << "Test decrypting(" << i << "): " << _fullTopic[i] << tk << (_fullTopic[i] ^ tk) << toHex(cipherText);
			assert(decryptSym(_fullTopic[i] ^ tk, cipherText, b));
			cnote << "Got: " << toHex(b);
		}
	}

	ret.proveWork(_workToProve);
	return ret;
}

Envelope::Envelope(RLP const& _m)
{
	m_expiry = _m[0].toInt<unsigned>();
	m_ttl = _m[1].toInt<unsigned>();
	m_topic = _m[2].toVector<FixedHash<4>>();
	m_data = _m[3].toBytes();
	m_nonce = _m[4].toInt<u256>();
}

Message Envelope::open(FullTopic const& _ft, Secret const& _s) const
{
	return Message(*this, _ft, _s);
}

unsigned Envelope::workProved() const
{
	h256 d[2];
	d[0] = sha3(WithoutNonce);
	d[1] = m_nonce;
	return dev::sha3(bytesConstRef(d[0].data(), 64)).firstBitSet();
}

void Envelope::proveWork(unsigned _ms)
{
	// PoW
	h256 d[2];
	d[0] = sha3(WithoutNonce);
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
