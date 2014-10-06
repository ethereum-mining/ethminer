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

#include <libdevcore/Log.h>
#include <libethcore/Exceptions.h>
#include <libethcore/BlockInfo.h>
#include "BlockChain.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

bool BlockQueue::import(bytesConstRef _block, BlockChain const& _bc)
{
	// Check if we already know this block.
	h256 h = BlockInfo::headerHash(_block);

	cblockq << "Queuing block" << h.abridged() << "for import...";

	UpgradableGuard l(m_lock);

	if (m_readySet.count(h) || m_drainingSet.count(h) || m_unknownSet.count(h))
	{
		// Already know about this one.
		cblockq << "Already known.";
		return false;
	}

	// VERIFY: populates from the block and checks the block is internally coherent.
	BlockInfo bi;

#if ETH_CATCH
	try
#endif
	{
		bi.populate(_block);
		bi.verifyInternals(_block);
	}
#if ETH_CATCH
	catch (Exception const& _e)
	{
		cwarn << "Ignoring malformed block: " << diagnostic_information(_e);
		return false;
	}
#endif

	// Check block doesn't already exist first!
	if (_bc.details(h))
	{
		cblockq << "Already known in chain.";
		return false;
	}

	UpgradeGuard ul(l);

	// Check it's not in the future
	if (bi.timestamp > (u256)time(0))
	{
		m_future.insert(make_pair((unsigned)bi.timestamp, _block.toBytes()));
		cblockq << "OK - queued for future.";
	}
	else
	{
		// We now know it.
		if (!m_readySet.count(bi.parentHash) && !m_drainingSet.count(bi.parentHash) && !_bc.isKnown(bi.parentHash))
		{
			// We don't know the parent (yet) - queue it up for later. It'll get resent to us if we find out about its ancestry later on.
			cblockq << "OK - queued as unknown parent:" << bi.parentHash.abridged();
			m_unknown.insert(make_pair(bi.parentHash, make_pair(h, _block.toBytes())));
			m_unknownSet.insert(h);
		}
		else
		{
			// If valid, append to blocks.
			cblockq << "OK - ready for chain insertion.";
			m_ready.push_back(_block.toBytes());
			m_readySet.insert(h);

			noteReadyWithoutWriteGuard(h);
		}
	}

	return true;
}

void BlockQueue::tick(BlockChain const& _bc)
{
	unsigned t = time(0);
	for (auto i = m_future.begin(); i != m_future.end() && i->first < time(0); ++i)
		import(&(i->second), _bc);

	WriteGuard l(m_lock);
	m_future.erase(m_future.begin(), m_future.upper_bound(t));
}

void BlockQueue::drain(std::vector<bytes>& o_out)
{
	WriteGuard l(m_lock);
	if (m_drainingSet.empty())
	{
		swap(o_out, m_ready);
		swap(m_drainingSet, m_readySet);
	}
}

void BlockQueue::noteReadyWithoutWriteGuard(h256 _good)
{
	list<h256> goodQueue(1, _good);
	while (goodQueue.size())
	{
		auto r = m_unknown.equal_range(goodQueue.front());
		goodQueue.pop_front();
		for (auto it = r.first; it != r.second; ++it)
		{
			m_ready.push_back(it->second.second);
			auto newReady = it->second.first;
			m_unknownSet.erase(newReady);
			m_readySet.insert(newReady);
			goodQueue.push_back(newReady);
		}
		m_unknown.erase(r.first, r.second);
	}
}
