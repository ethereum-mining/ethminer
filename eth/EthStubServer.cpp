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
/** @file EthStubServer.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "EthStubServer.h"
#include <libethereum/Client.h>
#include "CommonJS.h"
using namespace std;
using namespace eth;

EthStubServer::EthStubServer(jsonrpc::AbstractServerConnector* _conn, Client& _client):
	AbstractEthStubServer(_conn),
	m_client(_client)
{
}

std::string EthStubServer::coinbase()
{
	return toJS(m_client.address());
}

std::string EthStubServer::balanceAt(std::string const& _a)
{
	return toJS(m_client.postState().balance(jsToAddress(_a)));
}

Json::Value EthStubServer::check(Json::Value const& _as)
{
	if (m_client.changed())
		return _as;
	else
	{
		Json::Value ret;
		ret.resize(0);
		return ret;
	}
}




