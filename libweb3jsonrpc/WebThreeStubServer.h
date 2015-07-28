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

#include <libdevcore/db.h>
#include "WebThreeStubServerBase.h"

namespace dev
{

class WebThreeDirect;
namespace eth
{
class KeyManager;
class TrivialGasPricer;
class BlockChain;
class BlockQueue;
}

struct SessionPermissions
{
	std::unordered_set<Privilege> privileges;
};

/**
 * @brief JSON-RPC api implementation for WebThreeDirect
 */
class WebThreeStubServer: public dev::WebThreeStubServerBase, public dev::WebThreeStubDatabaseFace
{
public:
	WebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, dev::WebThreeDirect& _web3, std::shared_ptr<dev::eth::AccountHolder> const& _ethAccounts, std::vector<dev::KeyPair> const& _shhAccounts, dev::eth::KeyManager& _keyMan, dev::eth::TrivialGasPricer& _gp);

	virtual std::string web3_clientVersion() override;

	std::string newSession(SessionPermissions const& _p);
	void addSession(std::string const& _session, SessionPermissions const& _p) { m_sessions[_session] = _p; }

	virtual void setMiningBenefactorChanger(std::function<void(Address const&)> const& _f) { m_setMiningBenefactor = _f; }

private:
	virtual bool hasPrivilegeLevel(std::string const& _session, Privilege _l) const override { auto it = m_sessions.find(_session); return it != m_sessions.end() && it->second.privileges.count(_l); }

	virtual dev::eth::Interface* client() override;
	virtual std::shared_ptr<dev::shh::Interface> face() override;
	virtual dev::WebThreeNetworkFace* network() override;
	virtual dev::WebThreeStubDatabaseFace* db() override;

	virtual std::string get(std::string const& _name, std::string const& _key) override;
	virtual void put(std::string const& _name, std::string const& _key, std::string const& _value) override;

	virtual bool eth_notePassword(std::string const& _password) override;
	virtual Json::Value admin_eth_blockQueueStatus(std::string const& _session) override;
	virtual bool admin_eth_setAskPrice(std::string const& _wei, std::string const& _session) override;
	virtual bool admin_eth_setBidPrice(std::string const& _wei, std::string const& _session) override;

	virtual Json::Value admin_eth_findBlock(std::string const& _blockHash, std::string const& _session) override;
	virtual std::string admin_eth_blockQueueFirstUnknown(std::string const& _session) override;
	virtual bool admin_eth_blockQueueRetryUnknown(std::string const& _session) override;

	virtual bool admin_eth_setMiningBenefactor(std::string const& _uuidOrAddress, std::string const& _session) override;
	virtual Json::Value admin_eth_allAccounts(std::string const& _session) override;
	virtual Json::Value admin_eth_newAccount(const Json::Value& _info, std::string const& _session) override;
	virtual Json::Value admin_eth_inspect(std::string const& _address, std::string const& _session) override;
	virtual Json::Value admin_eth_reprocess(std::string const& _blockNumberOrHash, std::string const& _session) override;
	virtual Json::Value admin_eth_vmTrace(std::string const& _blockNumberOrHash, int _txIndex, std::string const& _session) override;
	virtual Json::Value admin_eth_getReceiptByHashAndIndex(std::string const& _blockNumberOrHash, int _txIndex, std::string const& _session) override;

private:
	h256 blockHash(std::string const& _blockNumberOrHash) const;

	dev::eth::BlockChain const& bc() const;
	dev::eth::BlockQueue const& bq() const;

	dev::WebThreeDirect& m_web3;
	dev::eth::KeyManager& m_keyMan;
	dev::eth::TrivialGasPricer& m_gp;
	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;
	ldb::DB* m_db;

	std::function<void(Address const&)> m_setMiningBenefactor;
	std::unordered_map<std::string, SessionPermissions> m_sessions;
};

}
