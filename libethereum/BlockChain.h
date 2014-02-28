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
#include "Common.h"
#include "AddressState.h"
namespace ldb = leveldb;

namespace eth
{

class RLP;
class RLPStream;

struct BlockDetails
{
	BlockDetails(): number(0), totalDifficulty(0) {}
	BlockDetails(uint _n, u256 _tD, h256 _p, h256s _c): number(_n), totalDifficulty(_tD), parent(_p), children(_c) {}
	BlockDetails(RLP const& _r);
	bytes rlp() const;

	bool isNull() const { return !totalDifficulty; }
	explicit operator bool() const { return !isNull(); }

	uint number;
	u256 totalDifficulty;
	h256 parent;
	h256s children;
};

typedef std::map<h256, BlockDetails> BlockDetailsHash;

static const BlockDetails NullBlockDetails;
static const h256s NullH256s;

class Overlay;

class AlreadyHaveBlock: public std::exception {};
class UnknownParent: public std::exception {};

struct BlockChainChat: public LogChannel { static const char* name() { return "-B-"; } static const int verbosity = 7; };
struct BlockChainNote: public LogChannel { static const char* name() { return "=B="; } static const int verbosity = 4; };

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
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
	bool attemptImport(bytes const& _block, Overlay const& _stateDB);

	/// Import block into disk-backed DB
	void import(bytes const& _block, Overlay const& _stateDB);

	/// Get the number of the last block of the longest chain.
	BlockDetails const& details(h256 _hash) const;
	BlockDetails const& details() const { return details(currentHash()); }

	/// Get a given block (RLP format). Thread-safe.
	bytesConstRef block(h256 _hash) const;
	bytesConstRef block() const { return block(currentHash()); }

	/// Get a given block (RLP format). Thread-safe.
	h256 currentHash() const { return m_lastBlockHash; }

	/// Get the hash of the genesis block.
	h256 genesisHash() const { return m_genesisHash; }

	std::vector<std::pair<Address, AddressState>> interestQueue() { std::vector<std::pair<Address, AddressState>> ret; swap(ret, m_interestQueue); return ret; }
	void pushInterest(Address _a) { m_interest[_a]++; }
	void popInterest(Address _a) { if (m_interest[_a] > 1) m_interest[_a]--; else if (m_interest[_a]) m_interest.erase(_a); }

private:
	void checkConsistency();

	/// Get fully populated from disk DB.
	mutable BlockDetailsHash m_details;
	mutable std::map<h256, std::string> m_cache;
	mutable std::mutex m_lock;

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
};

std::ostream& operator<<(std::ostream& _out, BlockChain const& _bc);

}
