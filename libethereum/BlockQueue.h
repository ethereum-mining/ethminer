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
/** @file BlockQueue.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <boost/thread.hpp>
#include <libethential/Common.h>
#include "libethcore/CommonEth.h"
#include "Guards.h"

namespace eth
{

class BlockChain;

/**
 * @brief A queue of blocks. Sits between network or other I/O and the BlockChain.
 * Sorts them ready for blockchain insertion (with the BlockChain::sync() method).
 * @threadsafe
 */
class BlockQueue
{
public:
	/// Import a block into the queue.
	bool import(bytesConstRef _tx, BlockChain const& _bc);

	/// Grabs the blocks that are ready, giving them in the correct order for insertion into the chain.
	void drain(std::vector<bytes>& o_out) { WriteGuard l(m_lock); swap(o_out, m_ready); m_readySet.clear(); }

	/// Notify the queue that the chain has changed and a new block has attained 'ready' status (i.e. is in the chain).
	void noteReady(h256 _b) { WriteGuard l(m_lock); noteReadyWithoutWriteGuard(_b); }

	/// Get information on the items queued.
	std::pair<unsigned, unsigned> items() const { ReadGuard l(m_lock); return std::make_pair(m_ready.size(), m_future.size()); }

private:
	void noteReadyWithoutWriteGuard(h256 _b);

	mutable boost::shared_mutex m_lock;						///< General lock.
	std::set<h256> m_readySet;								///< All blocks ready for chain-import.
	std::vector<bytes> m_ready;								///< List of blocks, in correct order, ready for chain-import.
	std::set<h256> m_futureSet;								///< Set of all blocks whose parents are not ready/in-chain.
	std::multimap<h256, std::pair<h256, bytes>> m_future;	///< For transactions that have an unknown parent; we map their parent hash to the block stuff, and insert once the block appears.
};

}


