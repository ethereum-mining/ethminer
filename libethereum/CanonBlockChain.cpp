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

unique_ptr<Ethash::BlockHeader> CanonBlockChain<Ethash>::s_genesis;
boost::shared_mutex CanonBlockChain<Ethash>::x_genesis;
Nonce CanonBlockChain<Ethash>::s_nonce(u64(42));
string CanonBlockChain<Ethash>::s_genesisStateJSON;
bytes CanonBlockChain<Ethash>::s_genesisExtraData;

CanonBlockChain<Ethash>::CanonBlockChain(std::string const& _path, WithExisting _we, ProgressCallback const& _pc):
	FullBlockChain<Ethash>(createGenesisBlock(), createGenesisState(), _path)
{
	BlockChain::openDatabase(_path, _we, _pc);
}

void CanonBlockChain<Ethash>::reopen(WithExisting _we, ProgressCallback const& _pc)
{
	close();
	open(createGenesisBlock(), createGenesisState(), m_dbPath);
	openDatabase(m_dbPath, _we, _pc);
}

bytes CanonBlockChain<Ethash>::createGenesisBlock()
{
	RLPStream block(3);

	h256 stateRoot;
	{
		MemoryDB db;
		SecureTrieDB<Address, MemoryDB> state(&db);
		state.init();
		dev::eth::commit(createGenesisState(), state);
		stateRoot = state.root();
	}

	js::mValue val;
	json_spirit::read_string(s_genesisStateJSON.empty() ? c_network == Network::Frontier ? c_genesisInfoFrontier : c_genesisInfoOlympic : s_genesisStateJSON, val);
	js::mObject genesis = val.get_obj();

	h256 mixHash(genesis["mixhash"].get_str());
	h256 parentHash(genesis["parentHash"].get_str());
	h160 beneficiary(genesis["coinbase"].get_str());
	u256 difficulty = fromBigEndian<u256>(fromHex(genesis["difficulty"].get_str()));
	u256 gasLimit = fromBigEndian<u256>(fromHex(genesis["gasLimit"].get_str()));
	u256 timestamp = fromBigEndian<u256>(fromHex(genesis["timestamp"].get_str()));
	bytes extraData = fromHex(genesis["extraData"].get_str());
	h64 nonce(genesis["nonce"].get_str());

	block.appendList(15)
			<< parentHash
			<< EmptyListSHA3	// sha3(uncles)
			<< beneficiary
			<< stateRoot
			<< EmptyTrie	// transactions
			<< EmptyTrie	// receipts
			<< LogBloom()
			<< difficulty
			<< 0	// number
			<< gasLimit
			<< 0	// gasUsed
			<< timestamp
			<< (s_genesisExtraData.empty() ? extraData : s_genesisExtraData)
			<< mixHash
			<< nonce;
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

unordered_map<Address, Account> CanonBlockChain<Ethash>::createGenesisState()
{
	static std::unordered_map<Address, Account> s_ret;

	if (s_ret.empty())
	{
		js::mValue val;
		js::read_string(s_genesisStateJSON.empty() ? c_network == Network::Frontier ? c_genesisInfoFrontier : c_genesisInfoOlympic : s_genesisStateJSON, val);
		for (auto account: val.get_obj()["alloc"].get_obj())
		{
			u256 balance;
			if (account.second.get_obj().count("wei"))
				balance = u256(account.second.get_obj()["wei"].get_str());
			else if (account.second.get_obj().count("balance"))
				balance = u256(account.second.get_obj()["balance"].get_str());
			else if (account.second.get_obj().count("finney"))
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

void CanonBlockChain<Ethash>::setGenesis(std::string const& _json)
{
	WriteGuard l(x_genesis);
	s_genesisStateJSON = _json;
	s_genesis.reset();
}

void CanonBlockChain<Ethash>::forceGenesisExtraData(bytes const& _genesisExtraData)
{
	WriteGuard l(x_genesis);
	s_genesisExtraData = _genesisExtraData;
	s_genesis.reset();
}

Ethash::BlockHeader const& CanonBlockChain<Ethash>::genesis()
{
	UpgradableGuard l(x_genesis);
	if (!s_genesis)
	{
		auto gb = createGenesisBlock();
		UpgradeGuard ul(l);
		s_genesis.reset(new Ethash::BlockHeader);
		s_genesis->populate(&gb, CheckEverything);
	}
	return *s_genesis;
}
