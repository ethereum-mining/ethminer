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
/** @file JSV8Connector.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#pragma once

#include <string>
#include <jsonrpccpp/server/abstractserverconnector.h>
#include <libjsengine/JSV8RPC.h>

namespace dev
{
namespace eth
{

class JSV8Connector: public jsonrpc::AbstractServerConnector, private JSV8RPC
{
public:
	JSV8Connector(JSV8Engine const& _engine): JSV8RPC(_engine) {}
	virtual ~JSV8Connector();

	// implement AbstractServerConnector interface
	bool StartListening() override;
	bool StopListening() override;
	bool SendResponse(std::string const& _response, void* _addInfo = nullptr) override;

private:
	// implement JSV8RPC interface
	void onSend(char const* _payload) override;

	char const* lastResponse() const override { return m_lastResponse.c_str(); }

	std::string m_lastResponse = R"({"id": 1, "jsonrpc": "2.0", "error": "Uninitalized JSV8RPC!"})";
};

}
}
