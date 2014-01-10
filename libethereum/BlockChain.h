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

	/// Get the last block of the longest chain.
	bytesConstRef lastBlock() const { return block(m_lastBlockHash); }

	/// Get the full block chain, according to the GHOST algo and the blocks available in the db.
	u256s blockChain(u256Set const& _earlyExit) const;

	/// Get the number of the last block of the longest chain.
	u256 lastBlockNumber() const;

	bytesConstRef block(u256 _hash) const;

private:
	/// Get fully populated from disk DB.
	mutable std::map<u256, std::pair<uint, u256>> m_numberAndParent;
	mutable std::multimap<u256, u256> m_children;

	ldb::DB* m_db;

	/// Hash of the last (valid) block on the longest chain.
	u256 m_lastBlockHash;
	u256 m_genesisHash;
	bytes m_genesisBlock;
};

}
