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
/** @file EthStubServer.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 * @date 2014
 */

#if ETH_JSONRPC
#include "EthStubServer.h"
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include "CommonJS.h"
using namespace std;
using namespace eth;

EthStubServer::EthStubServer(jsonrpc::AbstractServerConnector* _conn, Client& _client):
	AbstractEthStubServer(_conn),
	m_client(_client)
{
}

//only works with a json spec that doesn't have notifications for now
Json::Value EthStubServer::procedures()
{
	Json::Value ret;
	
	for (auto proc: this->GetProtocolHanlder()->GetProcedures())
	{
		Json::Value proc_j;

		proc_j[proc.second->GetProcedureType() == 0 ? "method" : "notification"] = proc.first;

		Json::Value params_j;
		for (auto params: proc.second->GetParameters())
			params_j[params.first] = jsontypeToValue(params.second);
		proc_j["params"] = params_j;
		
		proc_j["returns"] = jsontypeToValue(proc.second->GetReturnType());

		ret.append(proc_j);
	}
	return ret;
}

std::string EthStubServer::coinbase()
{
	ClientGuard g(&m_client);
	return toJS(m_client.address());
}

std::string EthStubServer::balanceAt(std::string const& _a)
{
	ClientGuard g(&m_client);
	return toJS(m_client.postState().balance(jsToAddress(_a)));
}

Json::Value EthStubServer::check(Json::Value const& _as)
{
	if (m_client.changed())
		return _as;
	else
	{
		Json::Value ret;
		ret.resize(0);
		return ret;
	}
}

std::string EthStubServer::create(const std::string& _bCode, const std::string& _sec, const std::string& _xEndowment, const std::string& _xGas, const std::string& _xGasPrice)
{
	ClientGuard g(&m_client);
	Address ret = m_client.transact(jsToSecret(_sec), jsToU256(_xEndowment), jsToBytes(_bCode), jsToU256(_xGas), jsToU256(_xGasPrice));
	return toJS(ret);
}

std::string EthStubServer::lll(const std::string& _s)
{
	return "0x" + toHex(eth::compileLLL(_s));
}

std::string EthStubServer::gasPrice()
{
	return "100000000000000";
}

bool EthStubServer::isContractAt(const std::string& _a)
{
	ClientGuard g(&m_client);
	return m_client.postState().addressHasCode(jsToAddress(_a));
}

bool EthStubServer::isListening()
{
	return m_client.haveNetwork();
}

bool EthStubServer::isMining()
{
	return m_client.isMining();
}

std::string EthStubServer::key()
{
	if (!m_keys.size())
		return std::string();
	return toJS(m_keys[0].sec());
}

Json::Value EthStubServer::keys()
{
	Json::Value ret;
	for (auto i: m_keys)
		ret.append(toJS(i.secret()));
	return ret;
}

int EthStubServer::peerCount()
{
	ClientGuard g(&m_client);
	return m_client.peerCount();
}

std::string EthStubServer::storageAt(const std::string& _a, const std::string& x)
{
	ClientGuard g(&m_client);
	return toJS(m_client.postState().storage(jsToAddress(_a), jsToU256(x)));
}

Json::Value EthStubServer::transact(const std::string& _aDest, const std::string& _bData, const std::string& _sec, const std::string& _xGas, const std::string& _xGasPrice, const std::string& _xValue)
{
	ClientGuard g(&m_client);
	m_client.transact(jsToSecret(_sec), jsToU256(_xValue), jsToAddress(_aDest), jsToBytes(_bData), jsToU256(_xGas), jsToU256(_xGasPrice));
	return Json::Value();
}

std::string EthStubServer::txCountAt(const std::string& _a)
{
	ClientGuard g(&m_client);
	return toJS(m_client.postState().transactionsFrom(jsToAddress(_a)));
}

std::string EthStubServer::secretToAddress(const std::string& _a)
{
	return toJS(KeyPair(jsToSecret(_a)).address());
}

Json::Value EthStubServer::lastBlock()
{
	return blockJson("");
}

Json::Value EthStubServer::block(const std::string& _hash)
{
	return blockJson(_hash);
}

Json::Value EthStubServer::blockJson(const std::string& _hash)
{
	Json::Value res;
	auto const& bc = m_client.blockChain();
	
	auto b = _hash.length() ? bc.block(h256(_hash)) : bc.block();
	
	auto bi = BlockInfo(b);
	res["number"] = to_string(bc.details(bc.currentHash()).number);
	res["hash"] = boost::lexical_cast<string>(bi.hash);
	res["parentHash"] = boost::lexical_cast<string>(bi.parentHash);
	res["sha3Uncles"] = boost::lexical_cast<string>(bi.sha3Uncles);
	res["coinbaseAddress"] = boost::lexical_cast<string>(bi.coinbaseAddress);
	res["stateRoot"] = boost::lexical_cast<string>(bi.stateRoot);
	res["transactionsRoot"] = boost::lexical_cast<string>(bi.transactionsRoot);
	res["minGasPrice"] = boost::lexical_cast<string>(bi.minGasPrice);
	res["gasLimit"] = boost::lexical_cast<string>(bi.gasLimit);
	res["gasUsed"] = boost::lexical_cast<string>(bi.gasUsed);
	res["difficulty"] = boost::lexical_cast<string>(bi.difficulty);
	res["timestamp"] = boost::lexical_cast<string>(bi.timestamp);
	res["nonce"] = boost::lexical_cast<string>(bi.nonce);

	return res;
}

Json::Value EthStubServer::jsontypeToValue(int _jsontype)
{
	switch (_jsontype)
	{
		case jsonrpc::JSON_STRING: return ""; //Json::stringValue segfault, fuck knows why
		case jsonrpc::JSON_BOOLEAN: return Json::booleanValue;
		case jsonrpc::JSON_INTEGER: return Json::intValue;
		case jsonrpc::JSON_REAL: return Json::realValue;
		case jsonrpc::JSON_OBJECT: return Json::objectValue;
		case jsonrpc::JSON_ARRAY: return Json::arrayValue;
		default: return Json::nullValue;
	}
}

#endif
