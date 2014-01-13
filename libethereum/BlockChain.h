/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file BlockChain.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
#include <leveldb/db.h>
namespace ldb = leveldb;

namespace eth
{

struct BlockDetails
{
	uint number;
	u256 totalDifficulty;
	h256 parent;
	h256s children;
};

static const BlockDetails NullBlockDetails({0, 0, h256()});
static const h256s NullH256s;

/**
 * @brief Implements the blockchain database. All data this gives is disk-backed.
 */
class BlockChain
{
public:
	BlockChain();
	~BlockChain();

	/// (Potentially) renders invalid existing bytesConstRef returned by lastBlock.
	/// To be called from main loop every 100ms or so.
	void process();
	
	/// Attempt to import the given block.
	bool attemptImport(bytes const& _block) { try { import(_block); return true; } catch (...) { return false; } }

	/// Import block into disk-backed DB
	void import(bytes const& _block);

	/// Get the full block chain, according to the GHOST algo and the blocks available in the db.
	h256s blockChain(h256Set const& _earlyExit) const;

	/// Get the number of the last block of the longest chain.
	BlockDetails const& details(h256 _hash) const;

	/// Get a given block (RLP format).
	bytesConstRef block(h256 _hash) const;

	/// Get a given block (RLP format).
	h256 currentHash() const { return m_lastBlockHash; }

	/// Get the coinbase address of a given block.
	Address coinbaseAddress(h256 _hash) const;

private:
	/// Get fully populated from disk DB.
	mutable std::map<h256, BlockDetails> m_details;

	mutable std::map<h256, std::string> m_cache;

	ldb::DB* m_db;

	/// Hash of the last (valid) block on the longest chain.
	h256 m_lastBlockHash;
	h256 m_genesisHash;
	bytes m_genesisBlock;

	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;
};

}
