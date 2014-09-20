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
/** @file DownloadMan.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <map>
#include <vector>
#include <set>
#include <libdevcore/Guards.h>
#include <libdevcore/Worker.h>
#include <libdevcore/RangeMask.h>
#include <libdevcore/FixedHash.h>

namespace dev
{

namespace eth
{

class DownloadMan;

class DownloadSub
{
	friend class DownloadMan;

public:
	DownloadSub(DownloadMan& _man);
	~DownloadSub();

	/// Finished last fetch - grab the next bunch of block hashes to download.
	h256Set nextFetch(unsigned _n);

	/// Note that we've received a particular block.
	void noteBlock(h256 _hash);

	/// Nothing doing here.
	void doneFetch() { resetFetch(); }

private:
	void resetFetch()		// Called by DownloadMan when we need to reset the download.
	{
		Guard l(m_fetch);
		m_remaining.clear();
		m_indices.clear();
		m_asked.clear();
		m_attempted.clear();
	}

	DownloadMan* m_man = nullptr;

	Mutex m_fetch;
	h256Set m_remaining;
	std::map<h256, unsigned> m_indices;
	RangeMask<unsigned> m_asked;
	RangeMask<unsigned> m_attempted;
};

class DownloadMan
{
	friend class DownloadSub;

public:
	~DownloadMan()
	{
		for (auto i: m_subs)
			i->m_man = nullptr;
	}

	void resetToChain(h256s const& _chain)
	{
		{
			ReadGuard l(x_subs);
			for (auto i: m_subs)
				i->resetFetch();
		}
		m_chain.clear();
		m_chain.reserve(_chain.size());
		for (auto i = _chain.rbegin(); i != _chain.rend(); ++i)
			m_chain.push_back(*i);
		m_blocksGot = RangeMask<unsigned>(0, m_chain.size());
	}

	RangeMask<unsigned> taken(bool _desperate = false) const
	{
		auto ret = m_blocksGot;
		if (!_desperate)
		{
			ReadGuard l(x_subs);
			for (auto i: m_subs)
				ret += i->m_asked;
		}
		return ret;
	}

	bool isComplete() const
	{
		return m_blocksGot.full();
	}

private:
	h256s m_chain;
	RangeMask<unsigned> m_blocksGot;

	mutable SharedMutex x_subs;
	std::set<DownloadSub*> m_subs;
};

}

}
