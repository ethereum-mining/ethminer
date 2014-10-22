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
/** @file WebThreeStubServer.h
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

#pragma once

#include <iostream>
#include <jsonrpc/rpc.h>
#include <libdevcrypto/Common.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "abstractwebthreestubserver.h"
#pragma GCC diagnostic pop

namespace dev { class WebThreeDirect; namespace eth { class Interface; } class KeyPair; }

class WebThreeStubServer: public AbstractWebThreeStubServer
{
public:
	WebThreeStubServer(jsonrpc::AbstractServerConnector* _conn, dev::WebThreeDirect& _web3, std::vector<dev::KeyPair> _accounts);
	
	virtual Json::Value accounts();
	virtual std::string balanceAt(std::string const& _address);
	virtual Json::Value block(int const& _block, std::string const& _hash);
	virtual std::string call(Json::Value const& _json);
	virtual bool check(int const& _id);
	virtual std::string codeAt(std::string const& _address);
	virtual std::string coinbase();
	virtual std::string compile(std::string const& _s);
	virtual double countAt(std::string const& _address);
	virtual int defaultBlock();
	virtual std::string gasPrice();
	virtual bool killWatch(int const& _id);
	virtual bool listening();
	virtual Json::Value messages(Json::Value const& _json);
	virtual bool mining();
	virtual int number();
	virtual int peerCount();
	virtual bool setCoinbase(std::string const& _address);
	virtual bool setListening(bool const& _listening);
	virtual bool setMining(bool const& _mining);
	virtual std::string stateAt(std::string const& _address, std::string const& _storage);
	virtual Json::Value transact(Json::Value const& _json);
	virtual Json::Value transaction(int const& _block, std::string const& _hash, int const& _i);
	virtual Json::Value uncle(int const& _block, std::string const& _hash, int const& _i);
	virtual int watch(std::string const& _json);
	
	void setAccounts(std::vector<dev::KeyPair> const& _accounts);
private:
	dev::eth::Interface* client() const;
	dev::WebThreeDirect& m_web3;
	std::map<dev::Address, dev::KeyPair> m_accounts;
};
