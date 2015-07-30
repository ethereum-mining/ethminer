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
/** @file BlockQueue.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "BlockQueue.h"
#include <thread>
#include <sstream>
#include <libdevcore/Log.h>
#include <libethcore/Exceptions.h>
#include <libethcore/BlockInfo.h>
#include "BlockChain.h"
#include "VerifiedBlock.h"
#include "State.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

#ifdef _WIN32
const char* BlockQueueChannel::name() { return EthOrange "[]>"; }
#else
const char* BlockQueueChannel::name() { return EthOrange "▣┅▶"; }
#endif
const char* BlockQueueTraceChannel::name() { return EthOrange "▣ ▶"; }

size_t const c_maxKnownCount = 100000;
size_t const c_maxKnownSize = 128 * 1024 * 1024;
size_t const c_maxUnknownCount = 100000;
size_t const c_maxUnknownSize = 512 * 1024 * 1024; // Block size can be ~50kb

BlockQueue::BlockQueue():
	m_unknownSize(0),
	m_knownSize(0),
	m_unknownCount(0),
	m_knownCount(0)
{
	// Allow some room for other activity
	unsigned verifierThreads = std::max(thread::hardware_concurrency(), 3U) - 2U;
	for (unsigned i = 0; i < verifierThreads; ++i)
		m_verifiers.emplace_back([=](){
			setThreadName("verifier" + toString(i));
			this->verifierBody();
		});
}

BlockQueue::~BlockQueue()
{
	m_deleting = true;
	m_moreToVerify.notify_all();
	for (auto& i: m_verifiers)
		i.join();
}

void BlockQueue::clear()
{
	WriteGuard l(m_lock);
	DEV_INVARIANT_CHECK;
	Guard l2(m_verification);
	m_readySet.clear();
	m_drainingSet.clear();
	m_verified.clear();
	m_unverified.clear();
	m_verifying.clear();
	m_unknownSet.clear();
	m_unknown.clear();
	m_future.clear();
	m_unknownSize = 0;
	m_unknownCount = 0;
	m_knownSize = 0;
	m_knownCount = 0;
	m_difficulty = 0;
	m_drainingDifficulty = 0;
}

void BlockQueue::verifierBody()
{
	while (!m_deleting)
	{
		UnverifiedBlock work;

		{
			unique_lock<Mutex> l(m_verification);
			m_moreToVerify.wait(l, [&](){ return !m_unverified.empty() || m_deleting; });
			if (m_deleting)
				return;
			swap(work, m_unverified.front());
			m_unverified.pop_front();
			BlockInfo bi;
			bi.setSha3Uncles(work.hash);
			bi.setParentHash(work.parentHash);
			m_verifying.emplace_back(move(bi));
		}

		VerifiedBlock res;
		swap(work.block, res.blockData);
		try
		{
			res.verified = m_bc->verifyBlock(&res.blockData, m_onBad, ImportRequirements::OutOfOrderChecks);
		}
		catch (...)
		{
			// bad block.
			// has to be this order as that's how invariants() assumes.
			WriteGuard l2(m_lock);
			unique_lock<Mutex> l(m_verification);
			m_readySet.erase(work.hash);
			m_knownBad.insert(work.hash);
			for (auto it = m_verifying.begin(); it != m_verifying.end(); ++it)
				if (it->verified.info.sha3Uncles() == work.hash)
				{
					m_verifying.erase(it);
					goto OK1;
				}
			cwarn << "BlockQueue missing our job: was there a GM?";
			OK1:;
			drainVerified_WITH_BOTH_LOCKS();
			continue;
		}

		bool ready = false;
		{
			WriteGuard l2(m_lock);
			unique_lock<Mutex> l(m_verification);
			if (!m_verifying.empty() && m_verifying.front().verified.info.sha3Uncles() == work.hash)
			{
				// we're next!
				m_verifying.pop_front();
				if (m_knownBad.count(res.verified.info.parentHash()))
				{
					m_readySet.erase(res.verified.info.hash());
					m_knownBad.insert(res.verified.info.hash());
				}
				else
					m_verified.emplace_back(move(res));

				drainVerified_WITH_BOTH_LOCKS();
				ready = true;
			}
			else
			{
				for (auto& i: m_verifying)
					if (i.verified.info.sha3Uncles() == work.hash)
					{
						i = move(res);
						goto OK;
					}
				cwarn << "BlockQueue missing our job: was there a GM?";
				OK:;
			}
		}
		if (ready)
			m_onReady();
	}
}

void BlockQueue::drainVerified_WITH_BOTH_LOCKS()
{
	while (!m_verifying.empty() && !m_verifying.front().blockData.empty())
	{
		if (m_knownBad.count(m_verifying.front().verified.info.parentHash()))
		{
			m_readySet.erase(m_verifying.front().verified.info.hash());
			m_knownBad.insert(m_verifying.front().verified.info.hash());
		}
		else
			m_verified.emplace_back(move(m_verifying.front()));
		m_verifying.pop_front();
	}
}

ImportResult BlockQueue::import(bytesConstRef _block, bool _isOurs)
{
	clog(BlockQueueTraceChannel) << std::this_thread::get_id();
	// Check if we already know this block.
	h256 h = BlockInfo::headerHashFromBlock(_block);

	clog(BlockQueueTraceChannel) << "Queuing block" << h << "for import...";

	UpgradableGuard l(m_lock);

	if (m_readySet.count(h) || m_drainingSet.count(h) || m_unknownSet.count(h) || m_knownBad.count(h))
	{
		// Already know about this one.
		clog(BlockQueueTraceChannel) << "Already known.";
		return ImportResult::AlreadyKnown;
	}

	BlockInfo bi;
	try
	{
		// TODO: quick verification of seal - will require BlockQueue to be templated on Sealer
		// VERIFY: populates from the block and checks the block is internally coherent.
		bi = m_bc->verifyBlock(_block, m_onBad, ImportRequirements::None).info;
	}
	catch (Exception const& _e)
	{
		cwarn << "Ignoring malformed block: " << diagnostic_information(_e);
		return ImportResult::Malformed;
	}

	clog(BlockQueueTraceChannel) << "Block" << h << "is" << bi.number() << "parent is" << bi.parentHash();

	// Check block doesn't already exist first!
	if (m_bc->isKnown(h))
	{
		cblockq << "Already known in chain.";
		return ImportResult::AlreadyInChain;
	}

	UpgradeGuard ul(l);
	DEV_INVARIANT_CHECK;

	// Check it's not in the future
	(void)_isOurs;
	if (bi.timestamp() > (u256)time(0)/* && !_isOurs*/)
	{
		m_future.insert(make_pair((unsigned)bi.timestamp(), make_pair(h, _block.toBytes())));
		char buf[24];
		time_t bit = (unsigned)bi.timestamp();
		if (strftime(buf, 24, "%X", localtime(&bit)) == 0)
			buf[0] = '\0'; // empty if case strftime fails
		clog(BlockQueueTraceChannel) << "OK - queued for future [" << bi.timestamp() << "vs" << time(0) << "] - will wait until" << buf;
		m_unknownSize += _block.size();
		m_unknownCount++;
		m_difficulty += bi.difficulty();
		bool unknown =  !m_readySet.count(bi.parentHash()) && !m_drainingSet.count(bi.parentHash()) && !m_bc->isKnown(bi.parentHash());
		return unknown ? ImportResult::FutureTimeUnknown : ImportResult::FutureTimeKnown;
	}
	else
	{
		// We now know it.
		if (m_knownBad.count(bi.parentHash()))
		{
			m_knownBad.insert(bi.hash());
			updateBad_WITH_LOCK(bi.hash());
			// bad parent; this is bad too, note it as such
			return ImportResult::BadChain;
		}
		else if (!m_readySet.count(bi.parentHash()) && !m_drainingSet.count(bi.parentHash()) && !m_bc->isKnown(bi.parentHash()))
		{
			// We don't know the parent (yet) - queue it up for later. It'll get resent to us if we find out about its ancestry later on.
			clog(BlockQueueTraceChannel) << "OK - queued as unknown parent:" << bi.parentHash();
			m_unknown.insert(make_pair(bi.parentHash(), make_pair(h, _block.toBytes())));
			m_unknownSet.insert(h);
			m_unknownSize += _block.size();
			m_difficulty += bi.difficulty();
			m_unknownCount++;

			return ImportResult::UnknownParent;
		}
		else
		{
			// If valid, append to blocks.
			clog(BlockQueueTraceChannel) << "OK - ready for chain insertion.";
			DEV_GUARDED(m_verification)
				m_unverified.push_back(UnverifiedBlock { h, bi.parentHash(), _block.toBytes() });
			m_moreToVerify.notify_one();
			m_readySet.insert(h);
			m_knownSize += _block.size();
			m_difficulty += bi.difficulty();
			m_knownCount++;

			noteReady_WITH_LOCK(h);

			return ImportResult::Success;
		}
	}
}

void BlockQueue::updateBad_WITH_LOCK(h256 const& _bad)
{
	DEV_INVARIANT_CHECK;
	DEV_GUARDED(m_verification)
	{
		collectUnknownBad_WITH_BOTH_LOCKS(_bad);
		bool moreBad = true;
		while (moreBad)
		{
			moreBad = false;
			std::deque<VerifiedBlock> oldVerified;
			swap(m_verified, oldVerified);
			for (auto& b: oldVerified)
				if (m_knownBad.count(b.verified.info.parentHash()) || m_knownBad.count(b.verified.info.hash()))
				{
					m_knownBad.insert(b.verified.info.hash());
					m_readySet.erase(b.verified.info.hash());
					collectUnknownBad_WITH_BOTH_LOCKS(b.verified.info.hash());
					moreBad = true;
				}
				else
					m_verified.push_back(std::move(b));

			std::deque<UnverifiedBlock> oldUnverified;
			swap(m_unverified, oldUnverified);
			for (auto& b: oldUnverified)
				if (m_knownBad.count(b.parentHash) || m_knownBad.count(b.hash))
				{
					m_knownBad.insert(b.hash);
					m_readySet.erase(b.hash);
					collectUnknownBad_WITH_BOTH_LOCKS(b.hash);
					moreBad = true;
				}
				else
					m_unverified.push_back(std::move(b));

			std::deque<VerifiedBlock> oldVerifying;
			swap(m_verifying, oldVerifying);
			for (auto& b: oldVerifying)
				if (m_knownBad.count(b.verified.info.parentHash()) || m_knownBad.count(b.verified.info.sha3Uncles()))
				{
					h256 const& h = b.blockData.size() != 0 ? b.verified.info.hash() : b.verified.info.sha3Uncles();
					m_knownBad.insert(h);
					m_readySet.erase(h);
					collectUnknownBad_WITH_BOTH_LOCKS(h);
					moreBad = true;
				}
				else
					m_verifying.push_back(std::move(b));
		}
	}
}

void BlockQueue::collectUnknownBad_WITH_BOTH_LOCKS(h256 const& _bad)
{
	list<h256> badQueue(1, _bad);
	while (!badQueue.empty())
	{
		auto r = m_unknown.equal_range(badQueue.front());
		badQueue.pop_front();
		for (auto it = r.first; it != r.second; ++it)
		{
			m_unknownSize -= it->second.second.size();
			m_unknownCount--;
			auto newBad = it->second.first;
			m_unknownSet.erase(newBad);
			m_knownBad.insert(newBad);
			badQueue.push_back(newBad);
		}
		m_unknown.erase(r.first, r.second);
	}
}

bool BlockQueue::doneDrain(h256s const& _bad)
{
	WriteGuard l(m_lock);
	DEV_INVARIANT_CHECK;
	m_drainingSet.clear();
	m_difficulty -= m_drainingDifficulty;
	m_drainingDifficulty = 0;
	if (_bad.size())
	{
		// at least one of them was bad.
		m_knownBad += _bad;
		for (h256 const& b : _bad)
			updateBad_WITH_LOCK(b);
	}
	return !m_readySet.empty();
}

void BlockQueue::tick()
{
	vector<pair<h256, bytes>> todo;
	{
		UpgradableGuard l(m_lock);
		if (m_future.empty())
			return;

		cblockq << "Checking past-future blocks...";

		unsigned t = time(0);
		if (t <= m_future.begin()->first)
			return;

		cblockq << "Past-future blocks ready.";

		{
			UpgradeGuard l2(l);
			DEV_INVARIANT_CHECK;
			auto end = m_future.lower_bound(t);
			for (auto i = m_future.begin(); i != end; ++i)
			{
				m_unknownSize -= i->second.second.size();
				m_unknownCount--;
				todo.push_back(move(i->second));
			}
			m_future.erase(m_future.begin(), end);
		}
	}
	cblockq << "Importing" << todo.size() << "past-future blocks.";

	for (auto const& b: todo)
		import(&b.second);
}

template <class T> T advanced(T _t, unsigned _n)
{
	std::advance(_t, _n);
	return _t;
}

QueueStatus BlockQueue::blockStatus(h256 const& _h) const
{
	ReadGuard l(m_lock);
	return
		m_readySet.count(_h) ?
			QueueStatus::Ready :
		m_drainingSet.count(_h) ?
			QueueStatus::Importing :
		m_unknownSet.count(_h) ?
			QueueStatus::UnknownParent :
		m_knownBad.count(_h) ?
			QueueStatus::Bad :
			QueueStatus::Unknown;
}

bool BlockQueue::knownFull() const
{
	return m_knownSize > c_maxKnownSize || m_knownCount > c_maxKnownCount;
}

bool BlockQueue::unknownFull() const
{
	return m_unknownSize > c_maxUnknownSize || m_unknownCount > c_maxUnknownCount;
}

void BlockQueue::drain(VerifiedBlocks& o_out, unsigned _max)
{
	bool wasFull = false;
	DEV_WRITE_GUARDED(m_lock)
	{
		DEV_INVARIANT_CHECK;
		wasFull = knownFull();
		if (m_drainingSet.empty())
		{
			m_drainingDifficulty = 0;
			DEV_GUARDED(m_verification)
			{
				o_out.resize(min<unsigned>(_max, m_verified.size()));
				for (unsigned i = 0; i < o_out.size(); ++i)
					swap(o_out[i], m_verified[i]);
				m_verified.erase(m_verified.begin(), advanced(m_verified.begin(), o_out.size()));
			}
			for (auto const& bs: o_out)
			{
				// TODO: @optimise use map<h256, bytes> rather than vector<bytes> & set<h256>.
				auto h = bs.verified.info.hash();
				m_drainingSet.insert(h);
				m_drainingDifficulty += bs.verified.info.difficulty();
				m_readySet.erase(h);
				m_knownSize -= bs.verified.block.size();
				m_knownCount--;
			}
		}
	}
	if (wasFull && !knownFull())
		m_onRoomAvailable();
}

bool BlockQueue::invariants() const
{
	Guard l(m_verification);
	if (!(m_readySet.size() == m_verified.size() + m_unverified.size() + m_verifying.size()))
	{
		std::stringstream s;
		s << "Failed BlockQueue invariant: m_readySet: " << m_readySet.size() << " m_verified: " << m_verified.size() << " m_unverified: " << m_unverified.size() << " m_verifying" << m_verifying.size();
		BOOST_THROW_EXCEPTION(FailedInvariant() << errinfo_comment(s.str()));
	}
	return true;
}

void BlockQueue::noteReady_WITH_LOCK(h256 const& _good)
{
	DEV_INVARIANT_CHECK;
	list<h256> goodQueue(1, _good);
	bool notify = false;
	while (!goodQueue.empty())
	{
		auto r = m_unknown.equal_range(goodQueue.front());
		goodQueue.pop_front();
		for (auto it = r.first; it != r.second; ++it)
		{
			DEV_GUARDED(m_verification)
				m_unverified.push_back(UnverifiedBlock { it->second.first, it->first, it->second.second });
			m_knownSize += it->second.second.size();
			m_knownCount++;
			m_unknownSize -= it->second.second.size();
			m_unknownCount--;
			auto newReady = it->second.first;
			m_unknownSet.erase(newReady);
			m_readySet.insert(newReady);
			goodQueue.push_back(newReady);
			notify = true;
		}
		m_unknown.erase(r.first, r.second);
	}
	if (notify)
		m_moreToVerify.notify_all();
}

void BlockQueue::retryAllUnknown()
{
	WriteGuard l(m_lock);
	DEV_INVARIANT_CHECK;
	for (auto it = m_unknown.begin(); it != m_unknown.end(); ++it)
	{
		DEV_GUARDED(m_verification)
			m_unverified.push_back(UnverifiedBlock { it->second.first, it->first, it->second.second });
		auto newReady = it->second.first;
		m_unknownSet.erase(newReady);
		m_readySet.insert(newReady);
		m_knownCount++;
		m_moreToVerify.notify_one();
	}
	m_unknown.clear();
	m_knownSize += m_unknownSize;
	m_unknownSize = 0;
	m_unknownCount = 0;
	m_moreToVerify.notify_all();
}

std::ostream& dev::eth::operator<<(std::ostream& _out, BlockQueueStatus const& _bqs)
{
	_out << "importing: " << _bqs.importing << endl;
	_out << "verified: " << _bqs.verified << endl;
	_out << "verifying: " << _bqs.verifying << endl;
	_out << "unverified: " << _bqs.unverified << endl;
	_out << "future: " << _bqs.future << endl;
	_out << "unknown: " << _bqs.unknown << endl;
	_out << "bad: " << _bqs.bad << endl;

	return _out;
}

u256 BlockQueue::difficulty() const
{
	UpgradableGuard l(m_lock);
	return m_difficulty;
}

bool BlockQueue::isActive() const
{
	UpgradableGuard l(m_lock);
	if (m_readySet.empty() && m_drainingSet.empty())
		DEV_GUARDED(m_verification)
			if (m_verified.empty() && m_verifying.empty() && m_unverified.empty())
				return false;
	return true;
}
