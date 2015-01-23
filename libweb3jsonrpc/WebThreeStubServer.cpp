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
/** @file WebThreeStubServer.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#include <boost/filesystem.hpp>
#include <libwebthree/WebThree.h>
#include <libdevcrypto/FileSystem.h>
#include "WebThreeStubServer.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

WebThreeStubServer::WebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, WebThreeDirect& _web3, std::vector<dev::KeyPair> const& _accounts):
	WebThreeStubServerBase(_conn, _accounts),
	m_web3(_web3)
{
	auto path = getDataDir() + "/.web3";
	boost::filesystem::create_directories(path);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path, &m_db);
}

dev::eth::Interface* WebThreeStubServer::client()
{
	return m_web3.ethereum();
}

std::shared_ptr<dev::shh::Interface> WebThreeStubServer::face()
{
	return m_web3.whisper();
}

dev::WebThreeNetworkFace* WebThreeStubServer::network()
{
	return &m_web3;
}

dev::WebThreeStubDatabaseFace* WebThreeStubServer::db()
{
	return this;
}

std::string WebThreeStubServer::get(std::string const& _name, std::string const& _key)
{
	bytes k = sha3(_name).asBytes() + sha3(_key).asBytes();
	string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return ret;
}

void WebThreeStubServer::put(std::string const& _name, std::string const& _key, std::string const& _value)
{
	bytes k = sha3(_name).asBytes() + sha3(_key).asBytes();
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)_value.data(), _value.size()));
}

