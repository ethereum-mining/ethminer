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
/** @file EthStubServer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <iostream>
#include <jsonrpc/rpc.h>
#include <libdevcrypto/Common.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "abstractethstubserver.h"
#pragma GCC diagnostic pop

namespace dev { class WebThreeDirect; namespace eth { class Client; } class KeyPair; }

class EthStubServer: public AbstractEthStubServer
{
public:
	EthStubServer(jsonrpc::AbstractServerConnector* _conn, dev::WebThreeDirect& _web3);

	virtual Json::Value procedures();
	virtual std::string balanceAt(std::string const& _a);
	virtual Json::Value check(Json::Value const& _as);
	virtual std::string coinbase();
	virtual std::string create(const std::string& bCode, const std::string& sec, const std::string& xEndowment, const std::string& xGas, const std::string& xGasPrice);
	virtual std::string gasPrice();
	virtual bool isContractAt(const std::string& a);
	virtual bool isListening();
	virtual bool isMining();
	virtual std::string key();
	virtual Json::Value keys();
	virtual int peerCount();
	virtual std::string storageAt(const std::string& a, const std::string& x);
	virtual Json::Value transact(const std::string& aDest, const std::string& bData, const std::string& sec, const std::string& xGas, const std::string& xGasPrice, const std::string& xValue);
	virtual std::string txCountAt(const std::string& a);
	virtual std::string secretToAddress(const std::string& a);
	virtual Json::Value lastBlock();
	virtual std::string lll(const std::string& s);
	virtual Json::Value block(const std::string&);
	void setKeys(std::vector<dev::KeyPair> _keys) { m_keys = _keys; }
private:
	dev::eth::Client& ethereum() const;
	dev::WebThreeDirect& m_web3;
	std::vector<dev::KeyPair> m_keys;
	Json::Value jsontypeToValue(int);
	Json::Value blockJson(const std::string&);
};
