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
/** @file BlockChain.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)

#include <mutex>
#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
#include <libethcore/CommonEth.h>
#include <libethcore/BlockInfo.h>
#include <libdevcore/Guards.h>
#include "BlockDetails.h"
#include "Account.h"
#include "BlockQueue.h"
namespace ldb = leveldb;

namespace dev
{

class OverlayDB;

namespace eth
{

static const h256s NullH256s;

class State;

struct AlreadyHaveBlock: virtual Exception {};
struct UnknownParent: virtual Exception {};
struct FutureTime: virtual Exception {};

struct BlockChainChat: public LogChannel { static const char* name() { return "-B-"; } static const int verbosity = 7; };
struct BlockChainNote: public LogChannel { static const char* name() { return "=B="; } static const int verbosity = 4; };

// TODO: Move all this Genesis stuff into Genesis.h/.cpp
std::map<Address, Account> const& genesisState();

ldb::Slice toSlice(h256 _h, unsigned _sub = 0);

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
 * @threadsafe
 * @todo Make not memory hog (should actually act as a cache and deallocate old entries).
 */
class BlockChain
{
public:
	BlockChain(bytes const& _genesisBlock, std::string _path, bool _killExisting);
	~BlockChain();

	void reopen(std::string _path, bool _killExisting = false) { close(); open(_path, _killExisting); }

	/// (Potentially) renders invalid existing bytesConstRef returned by lastBlock.
	/// To be called from main loop every 100ms or so.
	void process();

	/// Sync the chain with any incoming blocks. All blocks should, if processed in order
	h256s sync(BlockQueue& _bq, OverlayDB const& _stateDB, unsigned _max);

	/// Attempt to import the given block directly into the CanonBlockChain and sync with the state DB.
	/// @returns the block hashes of any blocks that came into/went out of the canonical block chain.
	h256s attemptImport(bytes const& _block, OverlayDB const& _stateDB) noexcept;

	/// Import block into disk-backed DB
	/// @returns the block hashes of any blocks that came into/went out of the canonical block chain.
	h256s import(bytes const& _block, OverlayDB const& _stateDB);

	/// Returns true if the given block is known (though not necessarily a part of the canon chain).
	bool isKnown(h256 _hash) const;

	/// Get the familial details concerning a block (or the most recent mined if none given). Thread-safe.
	BlockInfo info(h256 _hash) const { return BlockInfo(block(_hash)); }
	BlockInfo info() const { return BlockInfo(block()); }

	/// Get the familial details concerning a block (or the most recent mined if none given). Thread-safe.
	BlockDetails details(h256 _hash) const { return queryExtras<BlockDetails, 0>(_hash, m_details, x_details, NullBlockDetails); }
	BlockDetails details() const { return details(currentHash()); }

	/// Get the transactions' log blooms of a block (or the most recent mined if none given). Thread-safe.
	BlockLogBlooms logBlooms(h256 _hash) const { return queryExtras<BlockLogBlooms, 3>(_hash, m_logBlooms, x_logBlooms, NullBlockLogBlooms); }
	BlockLogBlooms logBlooms() const { return logBlooms(currentHash()); }

	/// Get the transactions' receipts of a block (or the most recent mined if none given). Thread-safe.
	BlockReceipts receipts(h256 _hash) const { return queryExtras<BlockReceipts, 4>(_hash, m_receipts, x_receipts, NullBlockReceipts); }
	BlockReceipts receipts() const { return receipts(currentHash()); }

	/// Get a block (RLP format) for the given hash (or the most recent mined if none given). Thread-safe.
	bytes block(h256 _hash) const;
	bytes block() const { return block(currentHash()); }

	/// Get a number for the given hash (or the most recent mined if none given). Thread-safe.
	unsigned number(h256 _hash) const { return details(_hash).number; }
	unsigned number() const { return number(currentHash()); }

	/// Get a given block (RLP format). Thread-safe.
	h256 currentHash() const { ReadGuard l(x_lastBlockHash); return m_lastBlockHash; }

	/// Get the hash of the genesis block. Thread-safe.
	h256 genesisHash() const { return m_genesisHash; }

	/// Get the hash of a block of a given number. Slow; try not to use it too much.
	h256 numberHash(unsigned _n) const;

	/// Get all blocks not allowed as uncles given a parent (i.e. featured as uncles/main in parent, parent + 1, ... parent + 5).
	/// @returns set including the header-hash of every parent (including @a _parent) up to and including generation +5
	/// togther with all their quoted uncles.
	h256Set allUnclesFrom(h256 _parent) const;

	/** @returns the hash of all blocks between @a _from and @a _to, all blocks are ordered first by a number of
	 * blocks that are parent-to-child, then two sibling blocks, then a number of blocks that are child-to-parent.
	 *
	 * If non-null, the h256 at @a o_common is set to the latest common ancestor of both blocks.
	 *
	 * e.g. if the block tree is 3a -> 2a -> 1a -> g and 2b -> 1b -> g (g is genesis, *a, *b are competing chains),
	 * then:
	 * @code
	 * treeRoute(3a, 2b) == { 3a, 2a, 1a, 1b, 2b }; // *o_common == g
	 * treeRoute(2a, 1a) == { 2a, 1a }; // *o_common == 1a
	 * treeRoute(1a, 2a) == { 1a, 2a }; // *o_common == 1a
	 * treeRoute(1b, 2a) == { 1b, 1a, 2a }; // *o_common == g
	 * @endcode
	 */
	h256s treeRoute(h256 _from, h256 _to, h256* o_common = nullptr, bool _pre = true, bool _post = true) const;

private:
	void open(std::string _path, bool _killExisting = false);
	void close();

	template<class T, unsigned N> T queryExtras(h256 _h, std::map<h256, T>& _m, boost::shared_mutex& _x, T const& _n) const
	{
		{
			ReadGuard l(_x);
			auto it = _m.find(_h);
			if (it != _m.end())
				return it->second;
		}

		std::string s;
		m_extrasDB->Get(m_readOptions, toSlice(_h, N), &s);
		if (s.empty())
		{
//			cout << "Not found in DB: " << _h << endl;
			return _n;
		}

		WriteGuard l(_x);
		auto ret = _m.insert(std::make_pair(_h, T(RLP(s))));
		return ret.first->second;
	}

	void checkConsistency();

	/// The caches of the disk DB and their locks.
	mutable boost::shared_mutex x_details;
	mutable BlockDetailsHash m_details;
	mutable boost::shared_mutex x_logBlooms;
	mutable BlockLogBloomsHash m_logBlooms;
	mutable boost::shared_mutex x_receipts;
	mutable BlockReceiptsHash m_receipts;
	mutable boost::shared_mutex x_cache;
	mutable std::map<h256, bytes> m_cache;

	/// The disk DBs. Thread-safe, so no need for locks.
	ldb::DB* m_db;
	ldb::DB* m_extrasDB;

	/// Hash of the last (valid) block on the longest chain.
	mutable boost::shared_mutex x_lastBlockHash;
	h256 m_lastBlockHash;

	/// Genesis block info.
	h256 m_genesisHash;
	bytes m_genesisBlock;

	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;

	friend std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc);
};

std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc);

}
}
