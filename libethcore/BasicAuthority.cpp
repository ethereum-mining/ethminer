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

#include <libdevcore/CommonJS.h>
#include "Exceptions.h"
#include "BasicAuthority.h"
#include "BlockInfo.h"
using namespace std;
using namespace dev;
using namespace eth;

AddressHash BasicAuthority::s_authorities;

bool BasicAuthority::BlockHeaderRaw::verify() const
{
	return s_authorities.count(toAddress(recover(m_sig, hashWithout())));
}

bool BasicAuthority::BlockHeaderRaw::preVerify() const
{
	return SignatureStruct(m_sig).isValid();
}

void BasicAuthority::BlockHeaderRaw::populateFromHeader(RLP const& _header, Strictness _s)
{
	m_sig = _header[BlockInfo::BasicFields].toHash<Signature>();

	// check it hashes according to proof of work or that it's the genesis block.
	if (_s == CheckEverything && m_parentHash && !verify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(m_difficulty);
		ex << errinfo_target(boundary());
		BOOST_THROW_EXCEPTION(ex);
	}
	else if (_s == QuickNonce && m_parentHash && !preVerify())
	{
		InvalidBlockNonce ex;
		ex << errinfo_hash256(hashWithout());
		ex << errinfo_difficulty(m_difficulty);
		BOOST_THROW_EXCEPTION(ex);
	}
}

void BasicAuthority::BlockHeaderRaw::verifyParent(BlockHeaderRaw const& _parent)
{
	(void)_parent;
}

void BasicAuthority::BlockHeaderRaw::populateFromParent(BlockHeaderRaw const& _parent)
{
	(void)_parent;
}

StringHashMap BasicAuthority::BlockHeaderRaw::jsInfo() const
{
	return { { "sig", toJS(m_sig) } };
}



class BasicAuthoritySealEngine: public SealEngineBase<BasicAuthority>
{
public:
	void setSecret(Secret const& _s) { m_secret = _s; }
	void generateSeal(BlockInfo const& _bi)
	{
		BasicAuthority::BlockHeader h(_bi);
		h.m_sig = sign(m_secret, _bi.hashWithout());
		RLPStream ret;
		h.streamRLP(ret);
		m_onSealGenerated(ret.out());
	}
	void onSealGenerated(std::function<void(bytes const&)> const& _f) { m_onSealGenerated = _f; }
	bool isWorking() const { return false; }
	WorkingProgress workingProgress() const { return WorkingProgress(); }

private:
	virtual bool onOptionChanging(std::string const& _name, bytes const& _value)
	{
		RLP rlp(_value);
		if (_name == "authorities")
			BasicAuthority::s_authorities = rlp.toUnorderedSet<Address>();
		else if (_name == "authority")
			m_secret = Secret(rlp.toHash<h256>());
		else
			return false;
		return true;
	}

	Secret m_secret;
	std::function<void(bytes const& s)> m_onSealGenerated;
};

SealEngineFace* BasicAuthority::createSealEngine()
{
	return new BasicAuthoritySealEngine;
}
