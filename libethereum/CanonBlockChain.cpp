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

#include <test/JsonSpiritHeaders.h>
#include <boost/filesystem.hpp>
#include <libdevcore/Common.h>
#include <libdevcore/RLP.h>
#include <libdevcore/FileSystem.h>
#include <libethcore/Exceptions.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/Params.h>
#include <liblll/Compiler.h>
#include "GenesisInfo.h"
#include "State.h"
#include "Defaults.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
namespace js = json_spirit;

#define ETH_CATCH 1

std::unordered_map<Address, Account> const& dev::eth::genesisState()
{
	static std::unordered_map<Address, Account> s_ret;

	if (s_ret.empty())
	{
		js::mValue val;
		json_spirit::read_string(c_genesisInfo, val);
		for (auto account: val.get_obj())
		{
			u256 balance;
			if (account.second.get_obj().count("wei"))
				balance = u256(account.second.get_obj()["wei"].get_str());
			else
				balance = u256(account.second.get_obj()["finney"].get_str()) * finney;
			if (account.second.get_obj().count("code"))
			{
				s_ret[Address(fromHex(account.first))] = Account(balance, Account::ContractConception);
				s_ret[Address(fromHex(account.first))].setCode(fromHex(account.second.get_obj()["code"].get_str()));
			}
			else
				s_ret[Address(fromHex(account.first))] = Account(balance, Account::NormalCreation);
		}
	}
	return s_ret;
}

// TODO: place Registry in here.

std::unique_ptr<BlockInfo> CanonBlockChain::s_genesis;
boost::shared_mutex CanonBlockChain::x_genesis;

bytes CanonBlockChain::createGenesisBlock()
{
	RLPStream block(3);

	h256 stateRoot;
	{
		MemoryDB db;
		SecureTrieDB<Address, MemoryDB> state(&db);
		state.init();
		dev::eth::commit(genesisState(), db, state);
		stateRoot = state.root();
	}

	block.appendList(15)
			<< h256() << EmptyListSHA3 << h160() << stateRoot << EmptyTrie << EmptyTrie << LogBloom() << c_genesisDifficulty << 0 << c_genesisGasLimit << 0 << (unsigned)0 << string() << h256() << Nonce(u64(42));
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

CanonBlockChain::CanonBlockChain(std::string const& _path, WithExisting _we, ProgressCallback const& _pc): BlockChain(CanonBlockChain::createGenesisBlock(), _path, _we, _pc)
{
}
