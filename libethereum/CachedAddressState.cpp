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
/** @file CachedAddressState.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CachedAddressState.h"

#include <libdevcore/TrieDB.h>
#include <libdevcrypto/Common.h>
#include <libdevcrypto/OverlayDB.h>
#include "Account.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

bool CachedAddressState::exists() const
{
	return (m_r && (!m_s || m_s->isAlive())) || (m_s && m_s->isAlive());
}

u256 CachedAddressState::balance() const
{
	return m_r ? m_s ? m_s->balance() : m_r[1].toInt<u256>() : 0;
}

u256 CachedAddressState::nonce() const
{
	return m_r ? m_s ? m_s->nonce() : m_r[0].toInt<u256>() : 0;
}

bytes CachedAddressState::code() const
{
	if (m_s && m_s->codeCacheValid())
		return m_s->code();
	h256 h = m_r ? m_s ? m_s->codeHash() : m_r[3].toHash<h256>() : EmptySHA3;
	return h == EmptySHA3 ? bytes() : asBytes(m_o->lookup(h));
}

std::map<u256, u256> CachedAddressState::storage() const
{
	std::map<u256, u256> ret;
	if (m_r)
	{
		SecureTrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(m_o), m_r[2].toHash<h256>());		// promise we won't alter the overlay! :)
		for (auto const& j: memdb)
			ret[j.first] = RLP(j.second).toInt<u256>();
	}
	if (m_s)
		for (auto const& j: m_s->storageOverlay())
			if ((!ret.count(j.first) && j.second) || (ret.count(j.first) && ret.at(j.first) != j.second))
				ret[j.first] = j.second;
	return ret;
}

AccountDiff CachedAddressState::diff(CachedAddressState const& _c)
{
	AccountDiff ret;
	ret.exist = Diff<bool>(exists(), _c.exists());
	ret.balance = Diff<u256>(balance(), _c.balance());
	ret.nonce = Diff<u256>(nonce(), _c.nonce());
	ret.code = Diff<bytes>(code(), _c.code());
	auto st = storage();
	auto cst = _c.storage();
	auto it = st.begin();
	auto cit = cst.begin();
	while (it != st.end() || cit != cst.end())
	{
		if (it != st.end() && cit != cst.end() && it->first == cit->first && (it->second || cit->second) && (it->second != cit->second))
			ret.storage[it->first] = Diff<u256>(it->second, cit->second);
		else if (it != st.end() && (cit == cst.end() || it->first < cit->first) && it->second)
			ret.storage[it->first] = Diff<u256>(it->second, 0);
		else if (cit != cst.end() && (it == st.end() || it->first > cit->first) && cit->second)
			ret.storage[cit->first] = Diff<u256>(0, cit->second);
		if (it == st.end())
			++cit;
		else if (cit == cst.end())
			++it;
		else if (it->first < cit->first)
			++it;
		else if (it->first > cit->first)
			++cit;
		else
			++it, ++cit;
	}
	return ret;
}
