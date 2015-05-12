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
/** @file FixedWebThreeStubServer.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#pragma once

#include <libdevcore/Exceptions.h>
#include <libweb3jsonrpc/WebThreeStubServerBase.h>
#include <libweb3jsonrpc/AccountHolder.h>

/**
 * @brief dummy JSON-RPC api implementation
 * Should be used for test purposes only
 * Supports eth && db interfaces
 * Doesn't support shh && net interfaces
 */
class FixedWebThreeServer: public dev::WebThreeStubServerBase, public dev::WebThreeStubDatabaseFace
{
public:
	FixedWebThreeServer(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _allAccounts, dev::eth::Interface* _client):
		WebThreeStubServerBase(_conn, std::make_shared<dev::eth::FixedAccountHolder>([=](){return _client;}, _allAccounts), _allAccounts),
		m_client(_client)
	{}

private:
	dev::eth::Interface* client() override { return m_client; }
	std::shared_ptr<dev::shh::Interface> face() override {	BOOST_THROW_EXCEPTION(dev::InterfaceNotSupported("dev::shh::Interface")); }
	dev::WebThreeNetworkFace* network() override { BOOST_THROW_EXCEPTION(dev::InterfaceNotSupported("dev::WebThreeNetworkFace")); }
	dev::WebThreeStubDatabaseFace* db() override { return this; }
	std::string get(std::string const& _name, std::string const& _key) override
	{
		std::string k(_name + "/" + _key);
		return m_db[k];
	}
	void put(std::string const& _name, std::string const& _key, std::string const& _value) override
	{
		std::string k(_name + "/" + _key);
		m_db[k] = _value;
	}

private:
	dev::eth::Interface* m_client;
	std::map<std::string, std::string> m_db;
};
