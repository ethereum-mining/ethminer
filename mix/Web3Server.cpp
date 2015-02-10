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
/** @file Web3Server.h.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <libdevcore/Exceptions.h>
#include <libdevcore/Log.h>
#include <libethereum/Interface.h>
#include "Web3Server.h"

using namespace dev::mix;

Web3Server::Web3Server(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _accounts, dev::eth::Interface* _client):
	WebThreeStubServerBase(_conn, _accounts),
	m_client(_client)
{
}

std::shared_ptr<dev::shh::Interface> Web3Server::face()
{
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::shh::Interface"));
}

dev::WebThreeNetworkFace* Web3Server::network()
{
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::WebThreeNetworkFace"));
}

std::string Web3Server::get(std::string const& _name, std::string const& _key)
{
	std::string k(_name + "/" + _key);
	return m_db[k];
}

void Web3Server::put(std::string const& _name, std::string const& _key, std::string const& _value)
{
	std::string k(_name + "/" + _key);
	m_db[k] = _value;
}

Json::Value Web3Server::eth_changed(int const& _id)
{
	return WebThreeStubServerBase::eth_changed(_id);
}

std::string Web3Server::eth_transact(Json::Value const& _json)
{
	std::string ret = WebThreeStubServerBase::eth_transact(_json);
	emit newTransaction();
	return ret;
}

std::string Web3Server::eth_call(Json::Value const& _json)
{
	std::string ret = WebThreeStubServerBase::eth_call(_json);
	emit newTransaction();
	return ret;
}
