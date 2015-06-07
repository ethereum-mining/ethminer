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
/** @file WebThreeStubServer.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#pragma once

#include <memory>
#include <iostream>
#include <jsonrpccpp/server.h>
#include <libdevcrypto/Common.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "abstractwebthreestubserver.h"
#pragma GCC diagnostic pop


namespace dev
{
class WebThreeNetworkFace;
class KeyPair;
namespace eth
{
class AccountHolder;
struct TransactionSkeleton;
class Interface;
}
namespace shh
{
class Interface;
}

extern const unsigned SensibleHttpThreads;
extern const unsigned SensibleHttpPort;

class WebThreeStubDatabaseFace
{
public:
	virtual std::string get(std::string const& _name, std::string const& _key) = 0;
	virtual void put(std::string const& _name, std::string const& _key, std::string const& _value) = 0;
};

/**
 * @brief JSON-RPC api implementation
 * @todo filters should work on unsigned instead of int
 * unsigned are not supported in json-rpc-cpp and there are bugs with double in json-rpc-cpp version 0.2.1
 * @todo split these up according to subprotocol (eth, shh, db, p2p, web3) and make it /very/ clear about how to add other subprotocols.
 * @todo modularise everything so additional subprotocols don't need to change this file.
 */
class WebThreeStubServerBase: public AbstractWebThreeStubServer
{
public:
	WebThreeStubServerBase(jsonrpc::AbstractServerConnector& _conn, std::shared_ptr<dev::eth::AccountHolder> const& _ethAccounts, std::vector<dev::KeyPair> const& _sshAccounts);

	virtual std::string web3_sha3(std::string const& _param1);
	virtual std::string web3_clientVersion() { return "C++ (ethereum-cpp)"; }

	virtual std::string net_version() { return ""; }
	virtual std::string net_peerCount();
	virtual bool net_listening();

	virtual std::string eth_protocolVersion();
	virtual std::string eth_hashrate();
	virtual std::string eth_coinbase();
	virtual bool eth_mining();
	virtual std::string eth_gasPrice();
	virtual Json::Value eth_accounts();
	virtual std::string eth_blockNumber();
	virtual std::string eth_getBalance(std::string const& _address, std::string const& _blockNumber);
	virtual std::string eth_getStorageAt(std::string const& _address, std::string const& _position, std::string const& _blockNumber);
	virtual std::string eth_getTransactionCount(std::string const& _address, std::string const& _blockNumber);
	virtual std::string eth_getBlockTransactionCountByHash(std::string const& _blockHash);
	virtual std::string eth_getBlockTransactionCountByNumber(std::string const& _blockNumber);
	virtual std::string eth_getUncleCountByBlockHash(std::string const& _blockHash);
	virtual std::string eth_getUncleCountByBlockNumber(std::string const& _blockNumber);
	virtual std::string eth_getCode(std::string const& _address, std::string const& _blockNumber);
	virtual std::string eth_sendTransaction(Json::Value const& _json);
	virtual std::string eth_call(Json::Value const& _json, std::string const& _blockNumber);
	virtual bool eth_flush();
	virtual Json::Value eth_getBlockByHash(std::string const& _blockHash, bool _includeTransactions);
	virtual Json::Value eth_getBlockByNumber(std::string const& _blockNumber, bool _includeTransactions);
	virtual Json::Value eth_getTransactionByHash(std::string const& _transactionHash);
	virtual Json::Value eth_getTransactionByBlockHashAndIndex(std::string const& _blockHash, std::string const& _transactionIndex);
	virtual Json::Value eth_getTransactionByBlockNumberAndIndex(std::string const& _blockNumber, std::string const& _transactionIndex);
	virtual Json::Value eth_getUncleByBlockHashAndIndex(std::string const& _blockHash, std::string const& _uncleIndex);
	virtual Json::Value eth_getUncleByBlockNumberAndIndex(std::string const& _blockNumber, std::string const& _uncleIndex);
	virtual Json::Value eth_getCompilers();
	virtual std::string eth_compileLLL(std::string const& _s);
	virtual std::string eth_compileSerpent(std::string const& _s);
	virtual std::string eth_compileSolidity(std::string const& _code);
	virtual std::string eth_newFilter(Json::Value const& _json);
	virtual std::string eth_newBlockFilter();
	virtual std::string eth_newPendingTransactionFilter();
	virtual bool eth_uninstallFilter(std::string const& _filterId);
	virtual Json::Value eth_getFilterChanges(std::string const& _filterId);
	virtual Json::Value eth_getFilterLogs(std::string const& _filterId);
	virtual Json::Value eth_getLogs(Json::Value const& _json);
	virtual Json::Value eth_getWork();
	virtual bool eth_submitWork(std::string const& _nonce, std::string const&, std::string const& _mixHash);
	virtual std::string eth_register(std::string const& _address);
	virtual bool eth_unregister(std::string const& _accountId);
	virtual Json::Value eth_fetchQueuedTransactions(std::string const& _accountId);
	virtual std::string eth_signTransaction(Json::Value const& _transaction);
	virtual Json::Value eth_inspectTransaction(std::string const& _rlp);
	virtual bool eth_injectTransaction(std::string const& _rlp);

	virtual bool db_put(std::string const& _name, std::string const& _key, std::string const& _value);
	virtual std::string db_get(std::string const& _name, std::string const& _key);

	virtual bool shh_post(Json::Value const& _json);
	virtual std::string shh_newIdentity();
	virtual bool shh_hasIdentity(std::string const& _identity);
	virtual std::string shh_newGroup(std::string const& _id, std::string const& _who);
	virtual std::string shh_addToGroup(std::string const& _group, std::string const& _who);
	virtual std::string shh_newFilter(Json::Value const& _json);
	virtual bool shh_uninstallFilter(std::string const& _filterId);
	virtual Json::Value shh_getFilterChanges(std::string const& _filterId);
	virtual Json::Value shh_getMessages(std::string const& _filterId);
	
	void setIdentities(std::vector<dev::KeyPair> const& _ids);
	std::map<dev::Public, dev::Secret> const& ids() const { return m_shhIds; }

protected:
	virtual dev::eth::Interface* client() = 0;
	virtual std::shared_ptr<dev::shh::Interface> face() = 0;
	virtual dev::WebThreeNetworkFace* network() = 0;
	virtual dev::WebThreeStubDatabaseFace* db() = 0;

	std::shared_ptr<dev::eth::AccountHolder> m_ethAccounts;

	std::map<dev::Public, dev::Secret> m_shhIds;
	std::map<unsigned, dev::Public> m_shhWatches;
};

} //namespace dev
