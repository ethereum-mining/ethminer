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
/** @file BasicAuthority.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Exceptions.h"
#include "BasicAuthority.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace eth;

const Address BasicAuthority::Authority = Address("1234567890123456789012345678901234567890");

bool BasicAuthority::BlockHeaderRaw::verify() const
{
	return toAddress(recover(m_sig, hashWithout())) == Authority;
}

bool BasicAuthority::BlockHeaderRaw::preVerify() const
{
	return SignatureStruct(m_sig).isValid();
}

void BasicAuthority::BlockHeaderRaw::populateFromHeader(RLP const& _header, Strictness _s)
{
	m_sig = _header[BlockInfo::BasicFields].toHash<Signature>();

	// check it hashes according to proof of work or that it's the genesis block.
	if (_s == CheckEverything && parentHash && !verify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(difficulty);
		ex << errinfo_target(boundary());
		BOOST_THROW_EXCEPTION(ex);
	}
	else if (_s == QuickNonce && parentHash && !preVerify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(difficulty);
		BOOST_THROW_EXCEPTION(ex);
	}
}



class BasicAuthoritySeal: public SealFace
{
public:
	BasicAuthoritySeal(Signature const& _sig): m_sig(_sig) {}

	virtual bytes sealedHeader(BlockInfo const& _bi) const
	{
		BasicAuthority::BlockHeader h(_bi);
		h.m_sig = m_sig;
		RLPStream ret;
		h.streamRLP(ret);
		return ret.out();
	}

private:
	Signature m_sig;
};

class BasicAuthoritySealEngine: public SealEngineFace
{
public:
	void setSecret(Secret const& _s) { m_secret = _s; }
	void generateSeal(BlockInfo const& _bi)
	{
		BasicAuthoritySeal s(sign(m_secret, _bi.hashWithout()));
		m_onSealGenerated(&s);
	}
	void onSealGenerated(std::function<void(SealFace const* s)> const& _f) { m_onSealGenerated = _f; }
	bool isMining() const { return false; }
	MiningProgress miningProgress() const { return MiningProgress(); }

private:
	Secret m_secret;
	std::function<void(SealFace const* s)> m_onSealGenerated;
};

SealEngineFace* createSealEngine()
{
	return new BasicAuthoritySealEngine;
}
