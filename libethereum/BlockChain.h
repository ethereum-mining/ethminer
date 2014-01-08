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

namespace eth
{

class MappedBlock
{
public:
	MappedBlock() {}
	MappedBlock(u256 _hash) {}	// TODO: map memory from disk.
	~MappedBlock() {}			// TODO: unmap memory from disk

	bytesConstRef data() const { return bytesConstRef(); }

private:
	// TODO: memory mapping.
};

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
	bool attemptImport(bytes const& _block) { try { import(_bytes); return true; } catch (...) { return false; } }

	/// Import block into disk-backed DB
	void import(bytes const& _block);

	/// Get the last block of the longest chain.
	bytesConstRef lastBlock() const;	// TODO: switch to return MappedBlock or add the lock into vector_ref

	std::vector<u256> blockChain()

	/// Get the number of the last block of the longest chain.
	u256 lastBlockNumber() const;

	bytesConstRef block(u256 _hash) const;

private:
	/// Get fully populated from disk DB.
	mutable std::map<u256, std::pair<u256, u256>> m_numberAndParent;
	mutable std::multimap<u256, u256> m_children;

	/// Gets populated on demand. Inactive nodes are pruned after a while.
	mutable std::map<u256, std::shared_ptr<MappedBlock>> m_cache;

	/// Hash of the last (valid) block on the longest chain.
	u256 m_lastBlockHash;
	u256 m_genesisHash;
	bytes m_genesisBlock;
};

}
