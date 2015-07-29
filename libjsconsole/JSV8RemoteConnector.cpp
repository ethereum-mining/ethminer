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

#include "JSV8RemoteConnector.h"

using namespace dev;
using namespace dev::eth;

void JSV8RemoteConnector::onSend(char const* _payload)
{
	m_request.setUrl(m_url);
	m_request.setBody(_payload);
	long code;
	tie(code, m_lastResponse) = m_request.post();
	(void)code;
}
