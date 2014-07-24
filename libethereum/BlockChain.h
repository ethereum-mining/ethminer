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

#include <mutex>
#include <libethential/Log.h>
#include <libethcore/CommonEth.h>
#include <libethcore/BlockInfo.h>
#include "Manifest.h"
#include "AddressState.h"
namespace ldb = leveldb;

namespace eth
{

class RLP;
class RLPStream;

struct BlockDetails
{
	BlockDetails(): number(0), totalDifficulty(0) {}
	BlockDetails(uint _n, u256 _tD, h256 _p, h256s _c, h256 _bloom): number(_n), totalDifficulty(_tD), parent(_p), children(_c), bloom(_bloom) {}
	BlockDetails(RLP const& _r);
	bytes rlp() const;

	bool isNull() const { return !totalDifficulty; }
	explicit operator bool() const { return !isNull(); }

	uint number;			// TODO: remove?
	u256 totalDifficulty;
	h256 parent;
	h256s children;
	h256 bloom;
};

struct BlockBlooms
{
	BlockBlooms() {}
	BlockBlooms(RLP const& _r) { blooms = _r.toVector<h256>(); }
	bytes rlp() const { RLPStream s; s << blooms; return s.out(); }

	h256s blooms;
};

struct BlockTraces
{
	BlockTraces() {}
	BlockTraces(RLP const& _r) { for (auto const& i: _r) traces.emplace_back(i.data()); }
	bytes rlp() const { RLPStream s(traces.size()); for (auto const& i: traces) i.streamOut(s); return s.out(); }

	Manifests traces;
};


typedef std::map<h256, BlockDetails> BlockDetailsHash;
typedef std::map<h256, BlockBlooms> BlockBloomsHash;
typedef std::map<h256, BlockTraces> BlockTracesHash;

static const BlockDetails NullBlockDetails;
static const BlockBlooms NullBlockBlooms;
static const BlockTraces NullBlockTraces;

static const h256s NullH256s;

class State;
class OverlayDB;

class AlreadyHaveBlock: public std::exception {};
class UnknownParent: public std::exception {};
class FutureTime: public std::exception {};

struct BlockChainChat: public LogChannel { static const char* name() { return "-B-"; } static const int verbosity = 7; };
struct BlockChainNote: public LogChannel { static const char* name() { return "=B="; } static const int verbosity = 4; };

// TODO: Move all this Genesis stuff into Genesis.h/.cpp
std::map<Address, AddressState> const& genesisState();

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
 * @todo Make thread-safe.
 * @todo Make not memory hog (should actually act as a cache and deallocate old entries).
 */
class BlockChain
{
public:
	BlockChain(bool _killExisting = false): BlockChain(std::string(), _killExisting) {}
	BlockChain(std::string _path, bool _killExisting = false);
	~BlockChain();

	/// (Potentially) renders invalid existing bytesConstRef returned by lastBlock.
	/// To be called from main loop every 100ms or so.
	void process();

	/// Attempt to import the given block.
	h256s attemptImport(bytes const& _block, OverlayDB const& _stateDB);

	/// Import block into disk-backed DB
	/// @returns the block hashes of any blocks that came into/went out of the canonical block chain.
	h256s import(bytes const& _block, OverlayDB const& _stateDB);

	/// Get the familial details concerning a block (or the most recent mined if none given). Thread-safe.
	BlockDetails details(h256 _hash) const;
	BlockDetails details() const { return details(currentHash()); }

	/// Get the transactions' bloom filters of a block (or the most recent mined if none given). Thread-safe.
	BlockBlooms blooms(h256 _hash) const;
	BlockBlooms blooms() const { return blooms(currentHash()); }

	/// Get the transactions' trace manifests of a block (or the most recent mined if none given). Thread-safe.
	BlockTraces traces(h256 _hash) const;
	BlockTraces traces() const { return traces(currentHash()); }

	/// Get a given block (RLP format). Thread-safe.
	bytes block(h256 _hash) const;
	bytes block() const { return block(currentHash()); }

	uint number(h256 _hash) const;
	uint number() const { return number(currentHash()); }

	/// Get a given block (RLP format). Thread-safe.
	h256 currentHash() const { return m_lastBlockHash; }

	/// Get the hash of the genesis block.
	h256 genesisHash() const { return m_genesisHash; }

	/// Get the hash of a block of a given number.
	h256 numberHash(unsigned _n) const;

	/// @returns the genesis block header.
	static BlockInfo const& genesis() { if (!s_genesis) { auto gb = createGenesisBlock(); (s_genesis = new BlockInfo)->populate(&gb); } return *s_genesis; }

	/// @returns the genesis block as its RLP-encoded byte array.
	/// @note This is slow as it's constructed anew each call. Consider genesis() instead.
	static bytes createGenesisBlock();

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
	h256s treeRoute(h256 _from, h256 _to, h256* o_common = nullptr) const;

private:
	void checkConsistency();

	/// Get fully populated from disk DB.
	mutable BlockDetailsHash m_details;
	mutable BlockBloomsHash m_blooms;
	mutable BlockTracesHash m_traces;

	mutable std::map<h256, bytes> m_cache;
	mutable std::recursive_mutex m_lock;

	ldb::DB* m_db;
	ldb::DB* m_extrasDB;

	/// Hash of the last (valid) block on the longest chain.
	h256 m_lastBlockHash;
	h256 m_genesisHash;
	bytes m_genesisBlock;

	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;

	friend std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc);

	static BlockInfo* s_genesis;
};

std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc);

}
