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
#include <libdevcore/Common.h>
#include <libdevcore/Log.h>
#include <libethcore/CommonEth.h>
#include <libdevcore/Guards.h>

namespace dev
{
namespace eth
{

class BlockChain;

struct BlockQueueChannel: public LogChannel { static const char* name() { return "[]Q"; } static const int verbosity = 7; };
#define cblockq dev::LogOutputStream<dev::eth::BlockQueueChannel, true>()

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

	/// Notes that time has moved on and some blocks that used to be "in the future" may no be valid.
	void tick(BlockChain const& _bc);

	/// Grabs the blocks that are ready, giving them in the correct order for insertion into the chain.
	/// Don't forget to call doneDrain() once you're done importing.
	void drain(std::vector<bytes>& o_out);

	/// Must be called after a drain() call. Notes that the drained blocks have been imported into the blockchain, so we can forget about them.
	void doneDrain() { WriteGuard l(m_lock); m_drainingSet.clear(); }

	/// Notify the queue that the chain has changed and a new block has attained 'ready' status (i.e. is in the chain).
	void noteReady(h256 _b) { WriteGuard l(m_lock); noteReadyWithoutWriteGuard(_b); }

	/// Get information on the items queued.
	std::pair<unsigned, unsigned> items() const { ReadGuard l(m_lock); return std::make_pair(m_ready.size(), m_unknown.size()); }

private:
	void noteReadyWithoutWriteGuard(h256 _b);
	void notePresentWithoutWriteGuard(bytesConstRef _block);

	mutable boost::shared_mutex m_lock;						///< General lock.
	std::set<h256> m_readySet;								///< All blocks ready for chain-import.
	std::set<h256> m_drainingSet;							///< All blocks being imported.
	std::vector<bytes> m_ready;								///< List of blocks, in correct order, ready for chain-import.
	std::set<h256> m_unknownSet;							///< Set of all blocks whose parents are not ready/in-chain.
	std::multimap<h256, std::pair<h256, bytes>> m_unknown;	///< For transactions that have an unknown parent; we map their parent hash to the block stuff, and insert once the block appears.
	std::multimap<unsigned, bytes> m_future;	///< Set of blocks that are not yet valid.
};

}
}
