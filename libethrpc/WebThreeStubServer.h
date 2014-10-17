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
	WebThreeStubServer(jsonrpc::AbstractServerConnector* _conn, dev::WebThreeDirect& _web3);
	
	virtual std::string balanceAt(std::string const& address, int const& block);
	virtual Json::Value block(Json::Value const& params);
	virtual std::string call(Json::Value const& json);
	virtual std::string codeAt(std::string const& address, int const& block);
	virtual std::string coinbase();
	virtual double countAt(std::string const& address, int const& block);
	virtual int defaultBlock();
	virtual std::string fromAscii(int const& padding, std::string const& s);
	virtual double fromFixed(std::string const& s);
	virtual std::string gasPrice();
	virtual bool listening();
	virtual bool mining();
	virtual std::string key();
	virtual Json::Value keys();
	virtual std::string lll(std::string const& s);
	virtual Json::Value messages(Json::Value const& json);
	virtual int number();
	virtual std::string offset(int const& o, std::string const& s);
	virtual int peerCount();
	virtual std::string secretToAddress(std::string const& s);
	virtual bool setCoinbase(std::string const& address);
	virtual bool setListening(bool const& listening);
	virtual bool setMining(bool const& mining);
	virtual std::string sha3(std::string const& s);
	virtual std::string stateAt(std::string const& address, int const& block, std::string const& storage);
	virtual std::string toAscii(std::string const& s);
	virtual std::string toDecimal(std::string const& s);
	virtual std::string toFixed(double const& s);
	virtual std::string transact(Json::Value const & json);
	virtual Json::Value transaction(int const& i, Json::Value const& params);
	virtual Json::Value uncle(int const& i, Json::Value const& params);
	virtual int watch(std::string const& json);
	virtual bool check(int const& id);
	virtual bool killWatch(int const& id);
	
	void setKeys(std::vector<dev::KeyPair> _keys) { m_keys = _keys; }
private:
	dev::eth::Interface* client() const;
	dev::WebThreeDirect& m_web3;
	std::vector<dev::KeyPair> m_keys;
	dev::FixedHash<32> numberOrHash(Json::Value const& _json) const;
};
