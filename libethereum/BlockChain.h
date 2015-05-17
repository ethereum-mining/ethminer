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

#include <deque>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <libdevcore/Log.h>
#include <libdevcore/Exceptions.h>
#include <libdevcore/Guards.h>
#include <libethcore/Common.h>
#include <libethcore/BlockInfo.h>
#include <libevm/ExtVMFace.h>
#include "BlockDetails.h"
#include "Account.h"
#include "Transaction.h"
#include "BlockQueue.h"
namespace ldb = leveldb;

namespace std
{
template <> struct hash<pair<dev::h256, unsigned>>
{
	size_t operator()(pair<dev::h256, unsigned> const& _x) const { return hash<dev::h256>()(_x.first) ^ hash<unsigned>()(_x.second); }
};
}

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

struct BlockChainChat: public LogChannel { static const char* name(); static const int verbosity = 5; };
struct BlockChainNote: public LogChannel { static const char* name(); static const int verbosity = 3; };
struct BlockChainWarn: public LogChannel { static const char* name(); static const int verbosity = 1; };
struct BlockChainDebug: public LogChannel { static const char* name(); static const int verbosity = 0; };

// TODO: Move all this Genesis stuff into Genesis.h/.cpp
std::unordered_map<Address, Account> const& genesisState();

ldb::Slice toSlice(h256 const& _h, unsigned _sub = 0);

using BlocksHash = std::unordered_map<h256, bytes>;
using TransactionHashes = h256s;
using UncleHashes = h256s;
using ImportRoute = std::pair<h256s, h256s>;

enum {
	ExtraDetails = 0,
	ExtraBlockHash,
	ExtraTransactionAddress,
	ExtraLogBlooms,
	ExtraReceipts,
	ExtraBlocksBlooms
};

using ProgressCallback = std::function<void(unsigned, unsigned)>;

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
 * @threadsafe
 */
class BlockChain
{
public:
	BlockChain(bytes const& _genesisBlock, std::string _path, WithExisting _we, ProgressCallback const& _p = ProgressCallback());
	~BlockChain();

	/// Attempt a database re-open.
	void reopen(std::string const& _path, WithExisting _we = WithExisting::Trust) { close(); open(_path, _we); }

	/// (Potentially) renders invalid existing bytesConstRef returned by lastBlock.
	/// To be called from main loop every 100ms or so.
	void process();

	/// Sync the chain with any incoming blocks. All blocks should, if processed in order.
	/// @returns fresh blocks, dead blocks and true iff there are additional blocks to be processed waiting.
	std::tuple<h256s, h256s, bool> sync(BlockQueue& _bq, OverlayDB const& _stateDB, unsigned _max);

	/// Attempt to import the given block directly into the CanonBlockChain and sync with the state DB.
	/// @returns the block hashes of any blocks that came into/went out of the canonical block chain.
	std::pair<ImportResult, ImportRoute> attemptImport(bytes const& _block, OverlayDB const& _stateDB, ImportRequirements::value _ir = ImportRequirements::Default) noexcept;

	/// Import block into disk-backed DB
	/// @returns the block hashes of any blocks that came into/went out of the canonical block chain.
	ImportRoute import(bytes const& _block, OverlayDB const& _stateDB, ImportRequirements::value _ir = ImportRequirements::Default);
	ImportRoute import(BlockInfo const& _bi, bytes const& _block, OverlayDB const& _stateDB, ImportRequirements::value _ir = ImportRequirements::Default);

	/// Returns true if the given block is known (though not necessarily a part of the canon chain).
	bool isKnown(h256 const& _hash) const;

	/// Get the familial details concerning a block (or the most recent mined if none given). Thread-safe.
	BlockInfo info(h256 const& _hash) const { return BlockInfo(block(_hash), IgnoreNonce, _hash); }
	BlockInfo info() const { return info(currentHash()); }

	/// Get a block (RLP format) for the given hash (or the most recent mined if none given). Thread-safe.
	bytes block(h256 const& _hash) const;
	bytes block() const { return block(currentHash()); }
	bytes oldBlock(h256 const& _hash) const;

	/// Get the familial details concerning a block (or the most recent mined if none given). Thread-safe.
	BlockDetails details(h256 const& _hash) const { return queryExtras<BlockDetails, ExtraDetails>(_hash, m_details, x_details, NullBlockDetails); }
	BlockDetails details() const { return details(currentHash()); }

	/// Get the transactions' log blooms of a block (or the most recent mined if none given). Thread-safe.
	BlockLogBlooms logBlooms(h256 const& _hash) const { return queryExtras<BlockLogBlooms, ExtraLogBlooms>(_hash, m_logBlooms, x_logBlooms, NullBlockLogBlooms); }
	BlockLogBlooms logBlooms() const { return logBlooms(currentHash()); }

	/// Get the transactions' receipts of a block (or the most recent mined if none given). Thread-safe.
	BlockReceipts receipts(h256 const& _hash) const { return queryExtras<BlockReceipts, ExtraReceipts>(_hash, m_receipts, x_receipts, NullBlockReceipts); }
	BlockReceipts receipts() const { return receipts(currentHash()); }

	/// Get a list of transaction hashes for a given block. Thread-safe.
	TransactionHashes transactionHashes(h256 const& _hash) const { auto b = block(_hash); RLP rlp(b); h256s ret; for (auto t: rlp[1]) ret.push_back(sha3(t.data())); return ret; }
	TransactionHashes transactionHashes() const { return transactionHashes(currentHash()); }

	/// Get a list of uncle hashes for a given block. Thread-safe.
	UncleHashes uncleHashes(h256 const& _hash) const { auto b = block(_hash); RLP rlp(b); h256s ret; for (auto t: rlp[2]) ret.push_back(sha3(t.data())); return ret; }
	UncleHashes uncleHashes() const { return uncleHashes(currentHash()); }
	
	/// Get the hash for a given block's number.
	h256 numberHash(unsigned _i) const { if (!_i) return genesisHash(); return queryExtras<BlockHash, ExtraBlockHash>(h256(_i), m_blockHashes, x_blockHashes, NullBlockHash).value; }

	/// Get the last N hashes for a given block. (N is determined by the LastHashes type.)
	LastHashes lastHashes() const { return lastHashes(number()); }
	LastHashes lastHashes(unsigned _i) const;

	/** Get the block blooms for a number of blocks. Thread-safe.
	 * @returns the object pertaining to the blocks:
	 * level 0:
	 * 0x, 0x + 1, .. (1x - 1)
	 * 1x, 1x + 1, .. (2x - 1)
	 * ...
	 * (255x .. (256x - 1))
	 * level 1:
	 * 0x .. (1x - 1), 1x .. (2x - 1), ..., (255x .. (256x - 1))
	 * 256x .. (257x - 1), 257x .. (258x - 1), ..., (511x .. (512x - 1))
	 * ...
	 * level n, index i, offset o:
	 * i * (x ^ n) + o * x ^ (n - 1)
	 */
	BlocksBlooms blocksBlooms(unsigned _level, unsigned _index) const { return blocksBlooms(chunkId(_level, _index)); }
	BlocksBlooms blocksBlooms(h256 const& _chunkId) const { return queryExtras<BlocksBlooms, ExtraBlocksBlooms>(_chunkId, m_blocksBlooms, x_blocksBlooms, NullBlocksBlooms); }
	void clearBlockBlooms(unsigned _begin, unsigned _end);
	LogBloom blockBloom(unsigned _number) const { return blocksBlooms(chunkId(0, _number / c_bloomIndexSize)).blooms[_number % c_bloomIndexSize]; }
	std::vector<unsigned> withBlockBloom(LogBloom const& _b, unsigned _earliest, unsigned _latest) const;
	std::vector<unsigned> withBlockBloom(LogBloom const& _b, unsigned _earliest, unsigned _latest, unsigned _topLevel, unsigned _index) const;

	/// Get a transaction from its hash. Thread-safe.
	bytes transaction(h256 const& _transactionHash) const { TransactionAddress ta = queryExtras<TransactionAddress, ExtraTransactionAddress>(_transactionHash, m_transactionAddresses, x_transactionAddresses, NullTransactionAddress); if (!ta) return bytes(); return transaction(ta.blockHash, ta.index); }
	std::pair<h256, unsigned> transactionLocation(h256 const& _transactionHash) const { TransactionAddress ta = queryExtras<TransactionAddress, ExtraTransactionAddress>(_transactionHash, m_transactionAddresses, x_transactionAddresses, NullTransactionAddress); if (!ta) return std::pair<h256, unsigned>(h256(), 0); return std::make_pair(ta.blockHash, ta.index); }

	/// Get a block's transaction (RLP format) for the given block hash (or the most recent mined if none given) & index. Thread-safe.
	bytes transaction(h256 const& _blockHash, unsigned _i) const { bytes b = block(_blockHash); return RLP(b)[1][_i].data().toBytes(); }
	bytes transaction(unsigned _i) const { return transaction(currentHash(), _i); }

	/// Get all transactions from a block.
	std::vector<bytes> transactions(h256 const& _blockHash) const { bytes b = block(_blockHash); std::vector<bytes> ret; for (auto const& i: RLP(b)[1]) ret.push_back(i.data().toBytes()); return ret; }
	std::vector<bytes> transactions() const { return transactions(currentHash()); }

	/// Get a number for the given hash (or the most recent mined if none given). Thread-safe.
	unsigned number(h256 const& _hash) const { return details(_hash).number; }
	unsigned number() const { return m_lastBlockNumber; }

	/// Get a given block (RLP format). Thread-safe.
	h256 currentHash() const { ReadGuard l(x_lastBlockHash); return m_lastBlockHash; }

	/// Get the hash of the genesis block. Thread-safe.
	h256 genesisHash() const { return m_genesisHash; }

	/// Get all blocks not allowed as uncles given a parent (i.e. featured as uncles/main in parent, parent + 1, ... parent + @a _generations).
	/// @returns set including the header-hash of every parent (including @a _parent) up to and including generation + @a _generations
	/// togther with all their quoted uncles.
	h256Hash allKinFrom(h256 const& _parent, unsigned _generations) const;

	/// Run through database and verify all blocks by reevaluating.
	/// Will call _progress with the progress in this operation first param done, second total.
	void rebuild(std::string const& _path, ProgressCallback const& _progress = std::function<void(unsigned, unsigned)>(), bool _prepPoW = false);

	/** @returns a tuple of:
	 * - an vector of hashes of all blocks between @a _from and @a _to, all blocks are ordered first by a number of
	 * blocks that are parent-to-child, then two sibling blocks, then a number of blocks that are child-to-parent;
	 * - the block hash of the latest common ancestor of both blocks;
	 * - the index where the latest common ancestor of both blocks would either be found or inserted, depending
	 * on whether it is included.
	 *
	 * @param _common if true, include the common ancestor in the returned vector.
	 * @param _pre if true, include all block hashes running from @a _from until the common ancestor in the returned vector.
	 * @param _post if true, include all block hashes running from the common ancestor until @a _to in the returned vector.
	 *
	 * e.g. if the block tree is 3a -> 2a -> 1a -> g and 2b -> 1b -> g (g is genesis, *a, *b are competing chains),
	 * then:
	 * @code
	 * treeRoute(3a, 2b, false) == make_tuple({ 3a, 2a, 1a, 1b, 2b }, g, 3);
	 * treeRoute(2a, 1a, false) == make_tuple({ 2a, 1a }, 1a, 1)
	 * treeRoute(1a, 2a, false) == make_tuple({ 1a, 2a }, 1a, 0)
	 * treeRoute(1b, 2a, false) == make_tuple({ 1b, 1a, 2a }, g, 1)
	 * treeRoute(3a, 2b, true) == make_tuple({ 3a, 2a, 1a, g, 1b, 2b }, g, 3);
	 * treeRoute(2a, 1a, true) == make_tuple({ 2a, 1a }, 1a, 1)
	 * treeRoute(1a, 2a, true) == make_tuple({ 1a, 2a }, 1a, 0)
	 * treeRoute(1b, 2a, true) == make_tuple({ 1b, g, 1a, 2a }, g, 1)
	 * @endcode
	 */
	std::tuple<h256s, h256, unsigned> treeRoute(h256 const& _from, h256 const& _to, bool _common = true, bool _pre = true, bool _post = true) const;

	struct Statistics
	{
		unsigned memBlocks;
		unsigned memDetails;
		unsigned memLogBlooms;
		unsigned memReceipts;
		unsigned memTransactionAddresses;
		unsigned memBlockHashes;
		unsigned memTotal() const { return memBlocks + memDetails + memLogBlooms + memReceipts + memTransactionAddresses + memBlockHashes; }
	};

	/// @returns statistics about memory usage.
	Statistics usage(bool _freshen = false) const { if (_freshen) updateStats(); return m_lastStats; }

	/// Deallocate unused data.
	void garbageCollect(bool _force = false);

private:
	static h256 chunkId(unsigned _level, unsigned _index) { return h256(_index * 0xff + _level); }

	void open(std::string const& _path, WithExisting _we = WithExisting::Trust);
	void close();

	template<class T, unsigned N> T queryExtras(h256 const& _h, std::unordered_map<h256, T>& _m, boost::shared_mutex& _x, T const& _n, ldb::DB* _extrasDB = nullptr) const
	{
		{
			ReadGuard l(_x);
			auto it = _m.find(_h);
			if (it != _m.end())
				return it->second;
		}

		std::string s;
		(_extrasDB ? _extrasDB : m_extrasDB)->Get(m_readOptions, toSlice(_h, N), &s);
		if (s.empty())
		{
//			cout << "Not found in DB: " << _h << endl;
			return _n;
		}

		noteUsed(_h, N);

		WriteGuard l(_x);
		auto ret = _m.insert(std::make_pair(_h, T(RLP(s))));
		return ret.first->second;
	}

	void checkConsistency();

	/// The caches of the disk DB and their locks.
	mutable SharedMutex x_blocks;
	mutable BlocksHash m_blocks;
	mutable SharedMutex x_details;
	mutable BlockDetailsHash m_details;
	mutable SharedMutex x_logBlooms;
	mutable BlockLogBloomsHash m_logBlooms;
	mutable SharedMutex x_receipts;
	mutable BlockReceiptsHash m_receipts;
	mutable SharedMutex x_transactionAddresses;
	mutable TransactionAddressHash m_transactionAddresses;
	mutable SharedMutex x_blockHashes;
	mutable BlockHashHash m_blockHashes;
	mutable SharedMutex x_blocksBlooms;
	mutable BlocksBloomsHash m_blocksBlooms;

	using CacheID = std::pair<h256, unsigned>;
	mutable Mutex x_cacheUsage;
	mutable std::deque<std::unordered_set<CacheID>> m_cacheUsage;
	mutable std::unordered_set<CacheID> m_inUse;
	void noteUsed(h256 const& _h, unsigned _extra = (unsigned)-1) const;
	std::chrono::system_clock::time_point m_lastCollection;

	void noteCanonChanged() const { Guard l(x_lastLastHashes); m_lastLastHashes.clear(); }
	mutable Mutex x_lastLastHashes;
	mutable LastHashes m_lastLastHashes;
	mutable unsigned m_lastLastHashesNumber = (unsigned)-1;

	void updateStats() const;
	mutable Statistics m_lastStats;

	/// The disk DBs. Thread-safe, so no need for locks.
	ldb::DB* m_blocksDB;
	ldb::DB* m_extrasDB;

	/// Hash of the last (valid) block on the longest chain.
	mutable boost::shared_mutex x_lastBlockHash;
	h256 m_lastBlockHash;
	unsigned m_lastBlockNumber = 0;

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
