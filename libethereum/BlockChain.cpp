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
/** @file BlockChain.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Common.h"
#include "RLP.h"
#include "Exceptions.h"
#include "Dagger.h"
#include "BlockInfo.h"
#include "State.h"
#include "BlockChain.h"
using namespace std;
using namespace eth;

BlockChain::BlockChain()
{
	ldb::Options o;
	auto s = ldb::DB::Open(o, "blockchain", &m_db);

	// Initialise with the genesis as the last block on the longest chain.
	m_lastBlockHash = m_genesisHash = BlockInfo::genesis().hash;
	m_genesisBlock = BlockInfo::createGenesisBlock();

	// TODO: Insert details of genesis block.
}

BlockChain::~BlockChain()
{
}

h256s BlockChain::blockChain(h256Set const& _earlyExit) const
{
	// Return the current valid block chain from most recent to genesis.
	// Arguments for specifying a set of early-ends
	h256s ret;
	ret.reserve(m_details[m_lastBlockHash].number + 1);
	auto i = m_lastBlockHash;
	for (; i != m_genesisHash && !_earlyExit.count(i); i = m_details[i].parent)
		ret.push_back(i);
	ret.push_back(i);
	return ret;
}
// _bc.details(m_previousBlock.parentHash).children
// _bc->coinbaseAddress(i)
void BlockChain::import(bytes const& _block)
{
	try
	{
		// VERIFY: populates from the block and checks the block is internally coherent.
		BlockInfo bi(&_block);
		bi.verifyInternals(&_block);

		auto newHash = eth::sha3(_block);

		// Check block doesn't already exist first!
		if (m_details.count(newHash))
			return;

		// Work out its number as the parent's number + 1
		auto it = m_details.find(bi.parentHash);
		if (it == m_details.end())
			// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
			return;

		// Check family:
		BlockInfo biParent(block(bi.parentHash));
		bi.verifyParent(biParent);

		// Check transactions are valid and that they result in a state equivalent to our state_root.
		State s(bi.coinbaseAddress);
		s.sync(*this, bi.parentHash);

		// Get total difficulty increase and update state, checking it.
		BlockInfo biGrandParent;
		if (it->second.number)
			biGrandParent.populate(block(it->second.parent));
		u256 td = it->second.totalDifficulty + s.playback(&_block, bi, biParent, biGrandParent);

		// All ok - insert into DB
		m_details[newHash] = BlockDetails{(uint)it->second.number + 1, bi.parentHash, td};
		m_details[bi.parentHash].children.push_back(newHash);
		m_db->Put(m_writeOptions, ldb::Slice(toBigEndianString(newHash)), (ldb::Slice)ref(_block));

		// This might be the new last block...
		if (td > m_details[m_lastBlockHash].totalDifficulty)
			m_lastBlockHash = newHash;
	}
	catch (...)
	{
		// Exit silently on exception(?)
		return;
	}
}

bytesConstRef BlockChain::block(h256 _hash) const
{
	if (_hash == m_genesisHash)
		return &m_genesisBlock;

	m_db->Get(m_readOptions, ldb::Slice(toBigEndianString(_hash)), &m_cache[_hash]);
	return bytesConstRef(&m_cache[_hash]);
}

BlockDetails const& BlockChain::details(h256 _h) const
{
	auto it = m_details.find(_h);
	if (it == m_details.end())
		return NullBlockDetails;
	return it->second;
}
