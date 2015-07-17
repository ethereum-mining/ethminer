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
/** @file BlockChainLoader.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include <libethereum/CanonBlockChain.h>
#include "BlockChainLoader.h"
#include "StateLoader.h"
#include "Common.h"

using namespace std;
using namespace dev;
using namespace dev::test;
using namespace dev::eth;

BlockChainLoader::BlockChainLoader(Json::Value const& _json)
{
	// load pre state
	StateLoader sl(_json["pre"], m_dir.path());
	m_state = sl.state();

	// load genesisBlock
	m_bc.reset(new FullBlockChain<Ethash>(fromHex(_json["genesisRLP"].asString()), sl.stateDefinition(), m_dir.path(), WithExisting::Kill));
	assert(m_state.rootHash() == m_bc->info().stateRoot());

	// load blocks
	for (auto const& block: _json["blocks"])
	{
		bytes rlp = fromHex(block["rlp"].asString());
		m_bc->import(rlp, m_state.db());
	}

	// sync state
	m_state.sync(*m_bc);
}
