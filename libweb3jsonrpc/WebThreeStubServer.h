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

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)

#include <iostream>
#include <jsonrpccpp/server.h>
#include <libdevcrypto/Common.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "abstractwebthreestubserver.h"
#pragma GCC diagnostic pop

namespace ldb = leveldb;

namespace dev
{
class WebThreeDirect;
class KeyPair;
namespace eth
{
class Interface;
}
namespace shh
{
class Interface;
}
}

/**
 * @brief JSON-RPC api implementation
 * @todo filters should work on unsigned instead of int
 * unsigned are not supported in json-rpc-cpp and there are bugs with double in json-rpc-cpp version 0.2.1
 */
class WebThreeStubServer: public AbstractWebThreeStubServer
{
public:
	WebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, dev::WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts);
	
	virtual std::string account();
	virtual Json::Value accounts();
	virtual std::string addToGroup(std::string const& _group, std::string const& _who);
	virtual std::string balanceAt(std::string const& _address);
	virtual Json::Value blockByHash(std::string const& _hash);
	virtual Json::Value blockByNumber(int const& _number);
	virtual std::string call(Json::Value const& _json);
	virtual bool changed(int const& _id);
	virtual std::string codeAt(std::string const& _address);
	virtual std::string coinbase();
	virtual std::string compile(std::string const& _s);
	virtual double countAt(std::string const& _address);
	virtual int defaultBlock();
	virtual std::string gasPrice();
	virtual std::string get(std::string const& _name, std::string const& _key);
	virtual Json::Value getMessages(int const& _id);
	virtual std::string getString(std::string const& _name, std::string const& _key);
	virtual bool haveIdentity(std::string const& _id);
	virtual bool listening();
	virtual bool mining();
	virtual int newFilter(Json::Value const& _json);
	virtual int newFilterString(std::string const& _filter);
	virtual std::string newGroup(std::string const& _id, std::string const& _who);
	virtual std::string newIdentity();
	virtual int number();
	virtual int peerCount();
	virtual bool post(Json::Value const& _json);
	virtual bool put(std::string const& _name, std::string const& _key, std::string const& _value);
	virtual bool putString(std::string const& _name, std::string const& _key, std::string const& _value);
	virtual bool setCoinbase(std::string const& _address);
	virtual bool setDefaultBlock(int const& _block);
	virtual bool setListening(bool const& _listening);
	virtual bool setMining(bool const& _mining);
	virtual Json::Value shhChanged(int const& _id);
	virtual int shhNewFilter(Json::Value const& _json);
	virtual bool shhUninstallFilter(int const& _id);
	virtual std::string stateAt(std::string const& _address, std::string const& _storage);
	virtual std::string transact(Json::Value const& _json);
	virtual Json::Value transactionByHash(std::string const& _hash, int const& _i);
	virtual Json::Value transactionByNumber(int const& _number, int const& _i);
	virtual Json::Value uncleByHash(std::string const& _hash, int const& _i);
	virtual Json::Value uncleByNumber(int const& _number, int const& _i);
	virtual bool uninstallFilter(int const& _id);
	
	void setAccounts(std::vector<dev::KeyPair> const& _accounts);
	void setIdentities(std::vector<dev::KeyPair> const& _ids);
	std::map<dev::Public, dev::Secret> const& ids() const { return m_ids; }
private:
	dev::eth::Interface* client() const;
	std::shared_ptr<dev::shh::Interface> face() const;
	dev::WebThreeDirect& m_web3;
	std::map<dev::Address, dev::KeyPair> m_accounts;
	
	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;
	ldb::DB* m_db;
	
	std::map<dev::Public, dev::Secret> m_ids;
	std::map<unsigned, dev::Public> m_shhWatches;
};
