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
#include "BlockInfo.h"
#include "BlockChain.h"
using namespace std;
using namespace eth;

BlockChain::BlockChain()
{
	// Initialise with the genesis as the last block on the longest chain.
	m_lastBlockHash = m_genesisHash = BlockInfo::genesis().hash;
	m_genesisBlock = BlockInfo::createGenesisBlock();
}

BlockChain::~BlockChain()
{
}

void BlockChain::import(bytes const& _block)
{
}

bytesConstRef BlockChain::block(u256 _hash) const
{
	auto it = m_cache.find(_hash);
	if (it == m_cache.end())
	{
		// Load block from disk.
		pair<u256, std::shared_ptr<MappedBlock>> loaded;
		it = m_cache.insert(loaded).first;
	}
	return it->second->data();
}

bytesConstRef BlockChain::lastBlock() const
{
	if (m_lastBlockHash == m_genesisHash)
		return bytesConstRef((bytes*)&m_genesisBlock);

	return block(m_lastBlockHash);
}

u256 BlockChain::lastBlockNumber() const
{
	if (m_lastBlockHash == m_genesisHash)
		return 0;

	return m_numberAndParent[m_lastBlockHash].first;
}
