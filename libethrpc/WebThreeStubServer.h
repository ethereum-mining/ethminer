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
	
	virtual std::string balanceAt(const std::string& address, const int& block);
	virtual Json::Value block(const Json::Value& params);
	virtual std::string call(const Json::Value& json);
	virtual std::string codeAt(const std::string& address, const int& block);
	virtual std::string coinbase();
	virtual double countAt(const std::string& address, const int& block);
	virtual int defaultBlock();
	virtual std::string fromAscii(const int& padding, const std::string& s);
	virtual double fromFixed(const std::string& s);
	virtual std::string gasPrice();
	virtual bool listening();
	virtual bool mining();
	virtual std::string key();
	virtual Json::Value keys();
	virtual std::string lll(const std::string& s);
	virtual Json::Value messages(const Json::Value& json);
	virtual int number();
	virtual std::string offset(const int& o, const std::string& s);
	virtual int peerCount();
	virtual std::string secretToAddress(const std::string& s);
	virtual bool setCoinbase(const std::string& address);
	virtual bool setListening(const bool& listening);
	virtual bool setMining(const bool& mining);
	virtual std::string sha3(const std::string& s);
	virtual std::string stateAt(const std::string& address, const int& block, const std::string& storage);
	virtual std::string toAscii(const std::string& s);
	virtual std::string toDecimal(const std::string& s);
	virtual std::string toFixed(const double& s);
	virtual std::string transact(const Json::Value& json);
	virtual Json::Value transaction(const int& i, const Json::Value& params);
	virtual Json::Value uncle(const int& i, const Json::Value &params);
	virtual int watch(const std::string& json);
	virtual bool check(const int& id);
	virtual bool killWatch(const int& id);
	
	void setKeys(std::vector<dev::KeyPair> _keys) { m_keys = _keys; }
private:
	dev::eth::Interface* client() const;
	dev::WebThreeDirect& m_web3;
	std::vector<dev::KeyPair> m_keys;
	dev::FixedHash<32> numberOrHash(Json::Value const &_json) const;
};
