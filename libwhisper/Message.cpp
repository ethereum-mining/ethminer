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
#include "BloomFilter.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

Message::Message(Envelope const& _e, Topics const& _t, Secret const& _s)
{
	try
	{
		bytes b;
		if (_s)
			if (!decrypt(_s, &(_e.data()), b))
				return;
			else{}
		else if (!openBroadcastEnvelope(_e, _t, b))
			return;

		if (populate(b))
			if (_s)
				m_to = KeyPair(_s).pub();
	}
	catch (...)	// Invalid secret? TODO: replace ... with InvalidSecret
	{
	}
}

bool Message::openBroadcastEnvelope(Envelope const& _e, Topics const& _fk, bytes& o_b)
{
	// retrieve the key using the known topic and topicIndex.
	unsigned topicIndex = 0;
	Secret topicSecret;

	// determine topicSecret/topicIndex from knowledge of the collapsed topics (which give the order) and our full-size filter topic.
	AbridgedTopics knownTopic = abridge(_fk);
	for (unsigned ti = 0; ti < _fk.size() && !topicSecret; ++ti)
		for (unsigned i = 0; i < _e.topic().size(); ++i)
			if (_e.topic()[i] == knownTopic[ti])
			{
				topicSecret = Secret(_fk[ti]);
				topicIndex = i;
				break;
			}

	if (_e.data().size() < _e.topic().size() * h256::size)
		return false;

	unsigned index = topicIndex * 2;
	Secret encryptedKey(bytesConstRef(&(_e.data())).cropped(h256::size * index, h256::size));
	h256 salt = h256(bytesConstRef(&(_e.data())).cropped(h256::size * ++index, h256::size));
	Secret key = Secret(generateGamma(topicSecret, salt).makeInsecure() ^ encryptedKey.makeInsecure());
	bytesConstRef cipherText = bytesConstRef(&(_e.data())).cropped(h256::size * 2 * _e.topic().size());
	return decryptSym(key, cipherText, o_b);
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

Envelope Message::seal(Secret const& _from, Topics const& _fullTopics, unsigned _ttl, unsigned _workToProve) const
{
	AbridgedTopics topics = abridge(_fullTopics);
	Envelope ret(time(0) + _ttl, _ttl, topics);

	bytes input(1 + m_payload.size());
	input[0] = 0;
	memcpy(input.data() + 1, m_payload.data(), m_payload.size());

	if (_from) // needs a signature
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
		// this message is for broadcast (could be read by anyone who knows at least one of the topics)
		// create the shared secret for encrypting the payload, then encrypt the shared secret with each topic
		Secret s = Secret::random();
		for (h256 const& t: _fullTopics)
		{
			h256 salt = h256::random();
			ret.m_data += (generateGamma(Secret(t), salt).makeInsecure() ^ s.makeInsecure()).ref().toBytes();
			ret.m_data += salt.asBytes();
		}

		bytes d;
		encryptSym(s, &input, d);
		ret.m_data += d;
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

Message Envelope::open(Topics const& _t, Secret const& _s) const
{
	return Message(*this, _t, _s);
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
	h256 d[2];
	d[0] = sha3(WithoutNonce);
	unsigned bestBitSet = 0;
	bytesConstRef chuck(d[0].data(), 64);

	chrono::high_resolution_clock::time_point then = chrono::high_resolution_clock::now() + chrono::milliseconds(_ms);
	while (chrono::high_resolution_clock::now() < then)
		// do it rounds of 1024 for efficiency
		for (unsigned i = 0; i < 1024; ++i, ++d[1])
		{
			auto fbs = dev::sha3(chuck).firstBitSet();
			if (fbs > bestBitSet)
			{
				bestBitSet = fbs;
				m_nonce = (h256::Arith)d[1];
			}
		}
}

bool Envelope::matchesBloomFilter(TopicBloomFilterHash const& f) const
{
	for (AbridgedTopic t: m_topic)
		if (f.contains(TopicBloomFilter::bloom(t)))
			return true;

	return false;
}
