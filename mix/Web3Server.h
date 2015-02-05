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
/** @file Web3Server.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <map>
#include <string>
#include <QObject>
#include <libweb3jsonrpc/WebThreeStubServerBase.h>

namespace dev
{

namespace mix
{

class Web3Server: public QObject, public dev::WebThreeStubServerBase, public dev::WebThreeStubDatabaseFace
{
	Q_OBJECT

public:
	Web3Server(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _accounts, dev::eth::Interface* _client);

signals:
	void newTransaction();

protected:
	virtual Json::Value eth_changed(int const& _id) override;
	virtual std::string eth_transact(Json::Value const& _json) override;
	virtual std::string eth_call(Json::Value const& _json) override;

private:
	dev::eth::Interface* client() override { return m_client; }
	std::shared_ptr<dev::shh::Interface> face() override;
	dev::WebThreeNetworkFace* network() override;
	dev::WebThreeStubDatabaseFace* db() override { return this; }

	std::string get(std::string const& _name, std::string const& _key) override;
	void put(std::string const& _name, std::string const& _key, std::string const& _value) override;

private:
	dev::eth::Interface* m_client;
	std::map<std::string, std::string> m_db;
};

}

}
