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
/** @file DownloadMan.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "DownloadMan.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

DownloadSub::DownloadSub(DownloadMan& _man): m_man(&_man)
{
	WriteGuard l(m_man->x_subs);
	m_man->m_subs.insert(this);
}

DownloadSub::~DownloadSub()
{
	if (m_man)
	{
		WriteGuard l(m_man->x_subs);
		m_man->m_subs.erase(this);
	}
}

h256Hash DownloadSub::nextFetch(unsigned _n)
{
	Guard l(m_fetch);

	if (m_remaining.size())
		return m_remaining;

	m_asked.clear();
	m_indices.clear();
	m_remaining.clear();

	if (!m_man || m_man->chainEmpty())
		return h256Hash();

	m_asked = (~(m_man->taken() + m_attempted)).lowest(_n);
	if (m_asked.empty())
		m_asked = (~(m_man->taken(true) + m_attempted)).lowest(_n);
	m_attempted += m_asked;
	for (auto i: m_asked)
	{
		auto x = m_man->m_chain[i];
		m_remaining.insert(x);
		m_indices[x] = i;
	}
	return m_remaining;
}

bool DownloadSub::noteBlock(h256 _hash)
{
	Guard l(m_fetch);
	if (m_man && m_indices.count(_hash))
		m_man->m_blocksGot += m_indices[_hash];
	bool ret = !!m_remaining.count(_hash);
	m_remaining.erase(_hash);
	return ret;
}

HashDownloadSub::HashDownloadSub(HashDownloadMan& _man): m_man(&_man)
{
	WriteGuard l(m_man->x_subs);
	m_asked = RangeMask<unsigned>(m_man->m_chainStart, m_man->m_chainStart + m_man->m_chainCount);
	m_attempted = RangeMask<unsigned>(m_man->m_chainStart, m_man->m_chainStart + m_man->m_chainCount);
	m_man->m_subs.insert(this);
}

HashDownloadSub::~HashDownloadSub()
{
	if (m_man)
	{
		WriteGuard l(m_man->x_subs);
		m_man->m_subs.erase(this);
	}
}

void HashDownloadSub::resetFetch()
{
	Guard l(m_fetch);
	m_remaining = 0;
	m_asked = RangeMask<unsigned>(m_man->m_chainStart, m_man->m_chainStart + m_man->m_chainCount);
	m_attempted = RangeMask<unsigned>(m_man->m_chainStart, m_man->m_chainStart + m_man->m_chainCount);
}

unsigned HashDownloadSub::nextFetch(unsigned _n)
{
	Guard l(m_fetch);

	m_asked = RangeMask<unsigned>(m_man->m_chainStart, m_man->m_chainStart + m_man->m_chainCount);

	if (!m_man || m_man->chainEmpty())
		return 0;

	m_asked = (~(m_man->taken() + m_attempted)).lowest(_n);
	if (m_asked.empty())
		m_asked = (~(m_man->taken(true) + m_attempted)).lowest(_n);
	m_attempted += m_asked;
	return *m_asked.begin();
}

void HashDownloadSub::noteHash(unsigned _index, unsigned _size)
{
	Guard l(m_fetch);
	if (m_man)
		for(unsigned i = _index; i < _index + _size; ++i)
			m_man->m_got += i;
}
