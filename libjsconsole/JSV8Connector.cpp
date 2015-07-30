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
/** @file JSV8Connector.cpp
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 * Ethereum client.
 */

#include "JSV8Connector.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

bool JSV8Connector::StartListening()
{
	return true;
}

bool JSV8Connector::StopListening()
{
	return true;
}

bool JSV8Connector::SendResponse(std::string const& _response, void* _addInfo)
{
	(void)_addInfo;
	m_lastResponse = _response;
	return true;
}

void JSV8Connector::onSend(char const* payload)
{
	OnRequest(payload, NULL);
}

JSV8Connector::~JSV8Connector()
{
	StopListening();
}
