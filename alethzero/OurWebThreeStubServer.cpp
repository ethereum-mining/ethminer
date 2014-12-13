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
/** @file OurWebThreeStubServer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "OurWebThreeStubServer.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

OurWebThreeStubServer::OurWebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, dev::WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts):
	WebThreeStubServer(_conn, _web3, _accounts)
{}

std::string OurWebThreeStubServer::shh_newIdentity()
{
	dev::KeyPair kp = dev::KeyPair::create();
	emit onNewId(QString::fromStdString(toJS(kp.sec())));
	return toJS(kp.pub());
}
