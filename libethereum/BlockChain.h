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
// TODO: DB for full traces.

typedef std::map<h256, BlockDetails> BlockDetailsHash;

static const BlockDetails NullBlockDetails;
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
	bool attemptImport(bytes const& _block, OverlayDB const& _stateDB);

	/// Import block into disk-backed DB
	void import(bytes const& _block, OverlayDB const& _stateDB);

	/// Get the number of the last block of the longest chain.
	BlockDetails details(h256 _hash) const;
	BlockDetails details() const { return details(currentHash()); }

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

	std::vector<std::pair<Address, AddressState>> interestQueue() { std::vector<std::pair<Address, AddressState>> ret; swap(ret, m_interestQueue); return ret; }
	void pushInterest(Address _a) { m_interest[_a]++; }
	void popInterest(Address _a) { if (m_interest[_a] > 1) m_interest[_a]--; else if (m_interest[_a]) m_interest.erase(_a); }

	static BlockInfo const& genesis() { if (!s_genesis) { auto gb = createGenesisBlock(); (s_genesis = new BlockInfo)->populate(&gb); } return *s_genesis; }
	static bytes createGenesisBlock();

private:
	void checkConsistency();

	/// Get fully populated from disk DB.
	mutable BlockDetailsHash m_details;
	mutable std::map<h256, bytes> m_cache;
	mutable std::recursive_mutex m_lock;

	/// The queue of transactions that have happened that we're interested in.
	std::map<Address, int> m_interest;
	std::vector<std::pair<Address, AddressState>> m_interestQueue;

	ldb::DB* m_db;
	ldb::DB* m_detailsDB;

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
