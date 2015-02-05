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
/** @file CanonBlockChain.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "CanonBlockChain.h"

#include <boost/filesystem.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcrypto/FileSystem.h>
#include <libethcore/Exceptions.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/BlockInfo.h>
#include <liblll/Compiler.h>
#include "State.h"
#include "Defaults.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

#define ETH_CATCH 1

std::map<Address, Account> const& dev::eth::genesisState()
{
	static std::map<Address, Account> s_ret;
	if (s_ret.empty())
	{
		// Initialise.
		for (auto i: vector<string>({
			"dbdbdb2cbd23b783741e8d7fcf51e459b497e4a6",
			"e6716f9544a56c530d868e4bfbacb172315bdead",
			"b9c015918bdaba24b4ff057a92a3873d6eb201be",
			"1a26338f0d905e295fccb71fa9ea849ffa12aaf4",
			"2ef47100e0787b915105fd5e3f4ff6752079d5cb",
			"cd2a3d9f938e13cd947ec05abc7fe734df8dd826",
			"6c386a4b26f73c802f34673f7248bb118f97424a",
			"e4157b34ea9615cfbde6b4fda419828124b70c78"
		}))
			s_ret[Address(fromHex(i))] = Account(u256(1) << 200, Account::NormalCreation);
	}
	return s_ret;
}

std::unique_ptr<BlockInfo> CanonBlockChain::s_genesis;
boost::shared_mutex CanonBlockChain::x_genesis;

bytes CanonBlockChain::createGenesisBlock()
{
	RLPStream block(3);

	h256 stateRoot;
	{
		MemoryDB db;
		TrieDB<Address, MemoryDB> state(&db);
		state.init();
		dev::eth::commit(genesisState(), db, state);
		stateRoot = state.root();
	}

	block.appendList(14)
			<< h256() << EmptyListSHA3 << h160() << stateRoot << EmptyTrie << EmptyTrie << LogBloom() << c_genesisDifficulty << 0 << 1000000 << 0 << (unsigned)0 << string() << sha3(bytes(1, 42));
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

CanonBlockChain::CanonBlockChain(std::string _path, bool _killExisting): BlockChain(CanonBlockChain::createGenesisBlock(), _path, _killExisting)
{
}
