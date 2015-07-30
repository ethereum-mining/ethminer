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
/** @file JSV8RemoteConnector.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Connector from the standalone javascript console to a remote RPC node.
 */

#pragma once

#include <libjsengine/JSV8RPC.h>
#include "CURLRequest.h"

namespace dev
{
namespace eth
{

class JSV8RemoteConnector: private JSV8RPC
{

public:
	JSV8RemoteConnector(JSV8Engine const& _engine, std::string _url): JSV8RPC(_engine), m_url(_url) {}
	virtual ~JSV8RemoteConnector() {}

private:
	// implement JSV8RPC interface
	void onSend(char const* _payload) override;
	const char* lastResponse() const override { return m_lastResponse.c_str(); }

private:
	std::string m_url;
	std::string m_lastResponse = R"({"id": 1, "jsonrpc": "2.0", "error": "Uninitalized JSV8RPC!"})";
	CURLRequest m_request;
};

}
}
