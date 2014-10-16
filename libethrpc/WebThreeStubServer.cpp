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

#if ETH_JSONRPC
#include "WebThreeStubServer.h"
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libwebthree/WebThree.h>
#include <libdevcore/CommonJS.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

static Json::Value toJson(const dev::eth::BlockInfo& bi)
{
	Json::Value res;
	res["hash"] = boost::lexical_cast<string>(bi.hash);
	
	res["parentHash"] = toJS(bi.parentHash);
	res["sha3Uncles"] = toJS(bi.sha3Uncles);
	res["miner"] = toJS(bi.coinbaseAddress);
	res["stateRoot"] = toJS(bi.stateRoot);
	res["transactionsRoot"] = toJS(bi.transactionsRoot);
	res["difficulty"] = toJS(bi.difficulty);
	res["number"] = (int)bi.number;
	res["minGasPrice"] = toJS(bi.minGasPrice);
	res["gasLimit"] = (int)bi.gasLimit;
	res["timestamp"] = (int)bi.timestamp;
	res["extraData"] = jsFromBinary(bi.extraData);
	res["nonce"] = toJS(bi.nonce);
	return res;
}

static Json::Value toJson(const dev::eth::PastMessage& t)
{
	Json::Value res;
	res["input"] = jsFromBinary(t.input);
	res["output"] = jsFromBinary(t.output);
	res["to"] = toJS(t.to);
	res["from"] = toJS(t.from);
	res["value"] = jsToDecimal(toJS(t.value));
	res["origin"] = toJS(t.origin);
	res["timestamp"] = toJS(t.timestamp);
	res["coinbase"] = toJS(t.coinbase);
	res["block"] =  toJS(t.block);
	Json::Value path;
	for (int i: t.path)
		path.append(i);
	res["path"] = path;
	res["number"] = (int)t.number;
	return res;
}

static Json::Value toJson(const dev::eth::PastMessages& pms)
{
	Json::Value res;
	for (dev::eth::PastMessage const & t: pms)
		res.append(toJson(t));
	
	return res;
}

static Json::Value toJson(const dev::eth::Transaction& t)
{
	Json::Value res;
	res["hash"] = toJS(t.sha3());
	res["input"] = jsFromBinary(t.data);
	res["to"] = toJS(t.receiveAddress);
	res["from"] = toJS(t.sender());
	res["gas"] = (int)t.gas;
	res["gasPrice"] = toJS(t.gasPrice);
	res["nonce"] = toJS(t.nonce);
	res["value"] = toJS(t.value);
	return res;
}

WebThreeStubServer::WebThreeStubServer(jsonrpc::AbstractServerConnector* _conn, WebThreeDirect& _web3):
	AbstractWebThreeStubServer(_conn),
	m_web3(_web3)
{
}

dev::eth::Interface* WebThreeStubServer::client() const
{
	return m_web3.ethereum();
}

std::string WebThreeStubServer::balanceAt(const string &address, const int& block)
{
	return toJS(client()->balanceAt(jsToAddress(address), block));
}

dev::FixedHash<32> WebThreeStubServer::numberOrHash(Json::Value const &json) const
{
	dev::FixedHash<32> hash;
	if (!json["hash"].empty())
		hash = jsToFixed<32>(json["hash"].asString());
	else if (!json["number"].empty())
		hash = client()->hashFromNumber((unsigned)json["number"].asInt());
	return hash;
}

Json::Value WebThreeStubServer::block(const Json::Value &params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(params);
	return toJson(client()->blockInfo(hash));
}

static TransactionJS toTransaction(const Json::Value &json)
{
	TransactionJS ret;
	if (!json.isObject() || json.empty()){
		return ret;
	}
	
	if (!json["from"].empty())
		ret.from = jsToSecret(json["from"].asString());
	if (!json["to"].empty())
		ret.to = jsToAddress(json["to"].asString());
	if (!json["value"].empty())
		ret.value = jsToU256(json["value"].asString());
	if (!json["gas"].empty())
		ret.gas = jsToU256(json["gas"].asString());
	if (!json["gasPrice"].empty())
		ret.gasPrice = jsToU256(json["gasPrice"].asString());
	
	if (!json["data"].empty() || json["code"].empty() || json["dataclose"].empty())
	{
		if (json["data"].isString())
			ret.data = jsToBytes(json["data"].asString());
		else if (json["code"].isString())
			ret.data = jsToBytes(json["code"].asString());
		else if (json["data"].isArray())
			for (auto i: json["data"])
				dev::operator +=(ret.data, asBytes(jsPadded(i.asString(), 32)));
		else if (json["code"].isArray())
			for (auto i: json["code"])
				dev::operator +=(ret.data, asBytes(jsPadded(i.asString(), 32)));
		else if (json["dataclose"].isArray())
			for (auto i: json["dataclose"])
				dev::operator +=(ret.data, jsToBytes(i.asString()));
	}
	
	return ret;
}

std::string WebThreeStubServer::call(const Json::Value &json)
{
	std::string ret;
	if (!client())
		return ret;
	TransactionJS t = toTransaction(json);
	if (!t.to)
		return ret;
	if (!t.from && m_keys.size())
		t.from = m_keys[0].secret();
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = client()->balanceAt(KeyPair(t.from).address()) / t.gasPrice;
	ret = toJS(client()->call(t.from, t.value, t.to, t.data, t.gas, t.gasPrice));
	return ret;
}

std::string WebThreeStubServer::codeAt(const string &address, const int& block)
{
	return client() ? jsFromBinary(client()->codeAt(jsToAddress(address), block)) : "";
}

std::string WebThreeStubServer::coinbase()
{
	return client() ? toJS(client()->address()) : "";
}

double WebThreeStubServer::countAt(const string &address, const int& block)
{
	return client() ? (double)(uint64_t)client()->countAt(jsToAddress(address), block) : 0;
}

int WebThreeStubServer::defaultBlock()
{
	return client() ? client()->getDefault() : 0;
}

std::string WebThreeStubServer::fromAscii(const int& padding, const std::string& s)
{
	return jsFromBinary(s, padding);
}

double WebThreeStubServer::fromFixed(const string &s)
{
	return jsFromFixed(s);
}

std::string WebThreeStubServer::gasPrice()
{
	return toJS(10 * dev::eth::szabo);
}

bool WebThreeStubServer::listening()
{
	return m_web3.isNetworkStarted();
}

bool WebThreeStubServer::mining()
{
	return client() ? client()->isMining() : false;
}

std::string WebThreeStubServer::key()
{
	if (!m_keys.size())
		return std::string();
	return toJS(m_keys[0].sec());
}

Json::Value WebThreeStubServer::keys()
{
	Json::Value ret;
	for (auto i: m_keys)
		ret.append(toJS(i.secret()));
	return ret;
}

std::string WebThreeStubServer::lll(const string &s)
{
	return toJS(dev::eth::compileLLL(s));
}

static dev::eth::MessageFilter toMessageFilter(const Json::Value &json)
{
	dev::eth::MessageFilter filter;
	if (!json.isObject() || json.empty()){
		return filter;
	}
	
	if (!json["earliest"].empty())
		filter.withEarliest(json["earliest"].asInt());
	if (!json["latest"].empty())
		filter.withLatest(json["lastest"].asInt());
	if (!json["max"].empty())
		filter.withMax(json["max"].asInt());
	if (!json["skip"].empty())
		filter.withSkip(json["skip"].asInt());
	if (!json["from"].empty())
	{
		if (json["from"].isArray())
			for (auto i : json["from"])
				filter.from(jsToAddress(i.asString()));
		else
			filter.from(jsToAddress(json["from"].asString()));
	}
	if (!json["to"].empty())
	{
		if (json["to"].isArray())
			for (auto i : json["to"])
				filter.from(jsToAddress(i.asString()));
		else
			filter.from(jsToAddress(json["to"].asString()));
	}
	if (!json["altered"].empty())
	{
		if (json["altered"].isArray())
			for (auto i: json["altered"])
				if (i.isObject())
					filter.altered(jsToAddress(i["id"].asString()), jsToU256(i["at"].asString()));
				else
					filter.altered((jsToAddress(i.asString())));
		else if (json["altered"].isObject())
			filter.altered(jsToAddress(json["altered"]["id"].asString()), jsToU256(json["altered"]["at"].asString()));
		else
			filter.altered(jsToAddress(json["altered"].asString()));
    }

	return filter;
}

Json::Value WebThreeStubServer::messages(const Json::Value &json)
{
	Json::Value res;
	if (!client())
		return  res;
	return toJson(client()->messages(toMessageFilter(json)));
}

int WebThreeStubServer::number()
{
	return client() ? client()->number() + 1 : 0;
}

std::string WebThreeStubServer::offset(const int& o, const std::string& s)
{
	return toJS(jsToU256(s) + o);
}

int WebThreeStubServer::peerCount()
{
	return m_web3.peerCount();
}

std::string WebThreeStubServer::secretToAddress(const string &s)
{
	return toJS(KeyPair(jsToSecret(s)).address());
}

bool WebThreeStubServer::setCoinbase(const std::string &address)
{
	client()->setAddress(jsToAddress(address));
	return true;
}

bool WebThreeStubServer::setListening(const bool &listening)
{
	if (listening)
		m_web3.startNetwork();
	else
		m_web3.stopNetwork();
	return true;
}

bool WebThreeStubServer::setMining(const bool &mining)
{
	if (!client())
		return Json::nullValue;

	if (mining)
		client()->startMining();
	else
		client()->stopMining();
	return true;
}

std::string WebThreeStubServer::sha3(const string &s)
{
	return toJS(dev::eth::sha3(jsToBytes(s)));
}

std::string WebThreeStubServer::stateAt(const string &address, const int& block, const string &storage)
{
	return client() ? toJS(client()->stateAt(jsToAddress(address), jsToU256(storage), block)) : "";
}

std::string WebThreeStubServer::toAscii(const string &s)
{
	return jsToBinary(s);
}

std::string WebThreeStubServer::toDecimal(const string &s)
{
	return jsToDecimal(s);
}

std::string WebThreeStubServer::toFixed(const double &s)
{
	return jsToFixed(s);
}

std::string WebThreeStubServer::transact(const Json::Value &json)
{
	std::string ret;
	if (!client())
		return ret;
	TransactionJS t = toTransaction(json);
	if (!t.from && m_keys.size())
	{
		auto b = m_keys.front();
		for (auto a: m_keys)
			if (client()->balanceAt(KeyPair(a).address()) > client()->balanceAt(KeyPair(b).address()))
				b = a;
		t.from = b.secret();
	}
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(KeyPair(t.from).address()) / t.gasPrice);
	if (t.to)
		client()->transact(t.from, t.value, t.to, t.data, t.gas, t.gasPrice);
	else
		ret = toJS(client()->transact(t.from, t.value, t.data, t.gas, t.gasPrice));
	client()->flushTransactions();
	return ret;
}

Json::Value WebThreeStubServer::transaction(const int &i, const Json::Value &params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(params);
	return toJson(client()->transaction(hash, i));
}

Json::Value WebThreeStubServer::uncle(const int &i, const Json::Value &params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(params);
	return toJson(client()->uncle(hash, i));
}

int WebThreeStubServer::watch(const string &json)
{
	unsigned ret = -1;
	if (!client())
		return ret;
	if (json.compare("chain") == 0)
		ret = client()->installWatch(dev::eth::ChainChangedFilter);
	else if (json.compare("pending") == 0)
		ret = client()->installWatch(dev::eth::PendingChangedFilter);
	else
	{
		Json::Reader reader;
		Json::Value object;
		reader.parse(json, object);
		ret = client()->installWatch(toMessageFilter(object));
	}

	return ret;
}

bool WebThreeStubServer::check(const int& id)
{
	if (!client())
		return false;
	return client()->checkWatch(id);
}

bool WebThreeStubServer::killWatch(const int& id)
{
	if (!client())
		return false;
	client()->uninstallWatch(id);
	return true;
}

#endif
