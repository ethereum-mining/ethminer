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

#include "WebThreeStubServerBase.h"

namespace dev
{
class WebThreeDirect;
}

/**
 * @brief JSON-RPC api implementation for WebThreeDirect
 */
class WebThreeStubServer: public dev::WebThreeStubServerBase, public dev::WebThreeStubDatabaseFace
{
public:
	WebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, dev::WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts);
	
private:
	virtual dev::eth::Interface* client() override;
	virtual std::shared_ptr<dev::shh::Interface> face() override;
	virtual dev::WebThreeNetworkFace* network() override;
	virtual dev::WebThreeStubDatabaseFace* db() override;

	virtual std::string get(std::string const& _name, std::string const& _key) override;
	virtual void put(std::string const& _name, std::string const& _key, std::string const& _value) override;

private:
	dev::WebThreeDirect& m_web3;
	leveldb::ReadOptions m_readOptions;
	leveldb::WriteOptions m_writeOptions;
	leveldb::DB* m_db;
};
