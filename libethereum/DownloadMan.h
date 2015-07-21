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

#include <vector>
#include <unordered_set>
#include <unordered_map>
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
	h256Hash nextFetch(unsigned _n);

	/// Note that we've received a particular block. @returns true if we had asked for it but haven't received it yet.
	bool noteBlock(h256 _hash);

	/// Nothing doing here.
	void doneFetch() { resetFetch(); }

	bool askedContains(unsigned _i) const { Guard l(m_fetch); return m_asked.contains(_i); }
	RangeMask<unsigned> const& asked() const { return m_asked; }
	RangeMask<unsigned> const& attemped() const { return m_attempted; }

private:
	void resetFetch()		// Called by DownloadMan when we need to reset the download.
	{
		Guard l(m_fetch);
		m_remaining.clear();
		m_indices.clear();
		m_asked.reset();
		m_attempted.reset();
	}

	DownloadMan* m_man = nullptr;

	mutable Mutex m_fetch;
	h256Hash m_remaining;
	std::unordered_map<h256, unsigned> m_indices;
	RangeMask<unsigned> m_asked;
	RangeMask<unsigned> m_attempted;
};

class DownloadMan
{
	friend class DownloadSub;

public:
	struct Overview
	{
		size_t total;
		size_t firstIncomplete;
		size_t lastComplete;
		size_t lastStarted;
	};

	~DownloadMan()
	{
		for (auto i: m_subs)
			i->m_man = nullptr;
	}

	void appendToChain(h256s const& _hashes)
	{
		WriteGuard l(m_lock);
		m_chain.insert(m_chain.end(), _hashes.cbegin(), _hashes.cend());
		m_blocksGot = RangeMask<unsigned>(0, m_chain.size());
	}

	void resetToChain(h256s const& _chain)
	{
		DEV_READ_GUARDED(x_subs)
			for (auto i: m_subs)
				i->resetFetch();
		WriteGuard l(m_lock);
		m_chain.clear();
		m_chain.reserve(_chain.size());
		for (auto i = _chain.rbegin(); i != _chain.rend(); ++i)
			m_chain.push_back(*i);
		m_blocksGot = RangeMask<unsigned>(0, m_chain.size());
	}

	void reset()
	{
		DEV_READ_GUARDED(x_subs)
			for (auto i: m_subs)
				i->resetFetch();
		WriteGuard l(m_lock);
		m_chain.clear();
		m_blocksGot.reset();
	}

	RangeMask<unsigned> taken(bool _desperate = false) const
	{
		ReadGuard l(m_lock);
		auto ret = m_blocksGot;
		if (!_desperate)
			DEV_READ_GUARDED(x_subs)
				for (auto i: m_subs)
					ret += i->m_asked;
		return ret;
	}

	bool isComplete() const
	{
		ReadGuard l(m_lock);
		return m_blocksGot.full();
	}

	h256s remaining() const
	{
		h256s ret;
		DEV_READ_GUARDED(m_lock)
			for (auto i: m_blocksGot.inverted())
				ret.push_back(m_chain[i]);
		return ret;
	}

	h256 firstBlock() const { return m_chain.empty() ? h256() : m_chain[0]; }
	Overview overview() const;

	size_t chainSize() const { ReadGuard l(m_lock); return m_chain.size(); }
	size_t chainEmpty() const { ReadGuard l(m_lock); return m_chain.empty(); }
	void foreachSub(std::function<void(DownloadSub const&)> const& _f) const { ReadGuard l(x_subs); for(auto i: m_subs) _f(*i); }
	unsigned subCount() const { ReadGuard l(x_subs); return m_subs.size(); }
	RangeMask<unsigned> blocksGot() const { ReadGuard l(m_lock); return m_blocksGot; }

private:
	mutable SharedMutex m_lock;
	h256s m_chain;
	RangeMask<unsigned> m_blocksGot;

	mutable SharedMutex x_subs;
	std::unordered_set<DownloadSub*> m_subs;
};

}

}

