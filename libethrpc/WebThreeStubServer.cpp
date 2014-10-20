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

static Json::Value toJson(dev::eth::BlockInfo const& _bi)
{
	Json::Value res;
	res["hash"] = boost::lexical_cast<string>(_bi.hash);
	
	res["parentHash"] = toJS(_bi.parentHash);
	res["sha3Uncles"] = toJS(_bi.sha3Uncles);
	res["miner"] = toJS(_bi.coinbaseAddress);
	res["stateRoot"] = toJS(_bi.stateRoot);
	res["transactionsRoot"] = toJS(_bi.transactionsRoot);
	res["difficulty"] = toJS(_bi.difficulty);
	res["number"] = (int)_bi.number;
	res["minGasPrice"] = toJS(_bi.minGasPrice);
	res["gasLimit"] = (int)_bi.gasLimit;
	res["timestamp"] = (int)_bi.timestamp;
	res["extraData"] = jsFromBinary(_bi.extraData);
	res["nonce"] = toJS(_bi.nonce);
	return res;
}

static Json::Value toJson(dev::eth::PastMessage const& _t)
{
	Json::Value res;
	res["input"] = jsFromBinary(_t.input);
	res["output"] = jsFromBinary(_t.output);
	res["to"] = toJS(_t.to);
	res["from"] = toJS(_t.from);
	res["value"] = jsToDecimal(toJS(_t.value));
	res["origin"] = toJS(_t.origin);
	res["timestamp"] = toJS(_t.timestamp);
	res["coinbase"] = toJS(_t.coinbase);
	res["block"] =  toJS(_t.block);
	Json::Value path;
	for (int i: _t.path)
		path.append(i);
	res["path"] = path;
	res["number"] = (int)_t.number;
	return res;
}

static Json::Value toJson(dev::eth::PastMessages const& _pms)
{
	Json::Value res;
	for (dev::eth::PastMessage const& t: _pms)
		res.append(toJson(t));
	
	return res;
}

static Json::Value toJson(dev::eth::Transaction const& _t)
{
	Json::Value res;
	res["hash"] = toJS(_t.sha3());
	res["input"] = jsFromBinary(_t.data);
	res["to"] = toJS(_t.receiveAddress);
	res["from"] = toJS(_t.sender());
	res["gas"] = (int)_t.gas;
	res["gasPrice"] = toJS(_t.gasPrice);
	res["nonce"] = toJS(_t.nonce);
	res["value"] = toJS(_t.value);
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

std::string WebThreeStubServer::balanceAt(string const& _address, int const& _block)
{
	return toJS(client()->balanceAt(jsToAddress(_address), _block));
}

dev::FixedHash<32> WebThreeStubServer::numberOrHash(Json::Value const& _json) const
{
	dev::FixedHash<32> hash;
	if (!_json["hash"].empty())
		hash = jsToFixed<32>(_json["hash"].asString());
	else if (!_json["number"].empty())
		hash = client()->hashFromNumber((unsigned)_json["number"].asInt());
	return hash;
}

Json::Value WebThreeStubServer::block(Json::Value const& _params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(_params);
	return toJson(client()->blockInfo(hash));
}

static TransactionJS toTransaction(Json::Value const& _json)
{
	TransactionJS ret;
	if (!_json.isObject() || _json.empty()){
		return ret;
	}
	
	if (!_json["from"].empty())
		ret.from = jsToSecret(_json["from"].asString());
	if (!_json["to"].empty())
		ret.to = jsToAddress(_json["to"].asString());
	if (!_json["value"].empty())
		ret.value = jsToU256(_json["value"].asString());
	if (!_json["gas"].empty())
		ret.gas = jsToU256(_json["gas"].asString());
	if (!_json["gasPrice"].empty())
		ret.gasPrice = jsToU256(_json["gasPrice"].asString());
	
	if (!_json["data"].empty() || _json["code"].empty() || _json["dataclose"].empty())
	{
		if (_json["data"].isString())
			ret.data = jsToBytes(_json["data"].asString());
		else if (_json["code"].isString())
			ret.data = jsToBytes(_json["code"].asString());
		else if (_json["data"].isArray())
			for (auto i: _json["data"])
				dev::operator +=(ret.data, asBytes(jsPadded(i.asString(), 32)));
		else if (_json["code"].isArray())
			for (auto i: _json["code"])
				dev::operator +=(ret.data, asBytes(jsPadded(i.asString(), 32)));
		else if (_json["dataclose"].isArray())
			for (auto i: _json["dataclose"])
				dev::operator +=(ret.data, jsToBytes(i.asString()));
	}
	
	return ret;
}

std::string WebThreeStubServer::call(Json::Value const& _json)
{
	std::string ret;
	if (!client())
		return ret;
	TransactionJS t = toTransaction(_json);
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

std::string WebThreeStubServer::codeAt(string const& _address, int const& _block)
{
	return client() ? jsFromBinary(client()->codeAt(jsToAddress(_address), _block)) : "";
}

std::string WebThreeStubServer::coinbase()
{
	return client() ? toJS(client()->address()) : "";
}

double WebThreeStubServer::countAt(string const& _address, int const& _block)
{
	return client() ? (double)(uint64_t)client()->countAt(jsToAddress(_address), _block) : 0;
}

int WebThreeStubServer::defaultBlock()
{
	return client() ? client()->getDefault() : 0;
}

std::string WebThreeStubServer::fromAscii(int const& _padding, std::string const& _s)
{
	return jsFromBinary(_s, _padding);
}

double WebThreeStubServer::fromFixed(string const& _s)
{
	return jsFromFixed(_s);
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

std::string WebThreeStubServer::lll(string const& _s)
{
	return toJS(dev::eth::compileLLL(_s));
}

static dev::eth::MessageFilter toMessageFilter(Json::Value const& _json)
{
	dev::eth::MessageFilter filter;
	if (!_json.isObject() || _json.empty()){
		return filter;
	}
	
	if (!_json["earliest"].empty())
		filter.withEarliest(_json["earliest"].asInt());
	if (!_json["latest"].empty())
		filter.withLatest(_json["lastest"].asInt());
	if (!_json["max"].empty())
		filter.withMax(_json["max"].asInt());
	if (!_json["skip"].empty())
		filter.withSkip(_json["skip"].asInt());
	if (!_json["from"].empty())
	{
		if (_json["from"].isArray())
			for (auto i : _json["from"])
				filter.from(jsToAddress(i.asString()));
		else
			filter.from(jsToAddress(_json["from"].asString()));
	}
	if (!_json["to"].empty())
	{
		if (_json["to"].isArray())
			for (auto i : _json["to"])
				filter.from(jsToAddress(i.asString()));
		else
			filter.from(jsToAddress(_json["to"].asString()));
	}
	if (!_json["altered"].empty())
	{
		if (_json["altered"].isArray())
			for (auto i: _json["altered"])
				if (i.isObject())
					filter.altered(jsToAddress(i["id"].asString()), jsToU256(i["at"].asString()));
				else
					filter.altered((jsToAddress(i.asString())));
		else if (_json["altered"].isObject())
			filter.altered(jsToAddress(_json["altered"]["id"].asString()), jsToU256(_json["altered"]["at"].asString()));
		else
			filter.altered(jsToAddress(_json["altered"].asString()));
	}

	return filter;
}

Json::Value WebThreeStubServer::messages(Json::Value const& _json)
{
	Json::Value res;
	if (!client())
		return  res;
	return toJson(client()->messages(toMessageFilter(_json)));
}

int WebThreeStubServer::number()
{
	return client() ? client()->number() + 1 : 0;
}

std::string WebThreeStubServer::offset(int const& _o, std::string const& _s)
{
	return toJS(jsToU256(_s) + _o);
}

int WebThreeStubServer::peerCount()
{
	return m_web3.peerCount();
}

std::string WebThreeStubServer::secretToAddress(string const& _s)
{
	return toJS(KeyPair(jsToSecret(_s)).address());
}

bool WebThreeStubServer::setCoinbase(std::string const& _address)
{
	client()->setAddress(jsToAddress(_address));
	return true;
}

bool WebThreeStubServer::setListening(bool const& _listening)
{
	if (_listening)
		m_web3.startNetwork();
	else
		m_web3.stopNetwork();
	return true;
}

bool WebThreeStubServer::setMining(bool const& _mining)
{
	if (!client())
		return Json::nullValue;

	if (_mining)
		client()->startMining();
	else
		client()->stopMining();
	return true;
}

std::string WebThreeStubServer::sha3(string const& _s)
{
	return toJS(dev::sha3(jsToBytes(_s)));
}

std::string WebThreeStubServer::stateAt(string const& _address, int const& _block, string const& _storage)
{
	return client() ? toJS(client()->stateAt(jsToAddress(_address), jsToU256(_storage), _block)) : "";
}

std::string WebThreeStubServer::toAscii(string const& _s)
{
	return jsToBinary(_s);
}

std::string WebThreeStubServer::toDecimal(string const& _s)
{
	return jsToDecimal(_s);
}

std::string WebThreeStubServer::toFixed(double const& _s)
{
	return jsToFixed(_s);
}

std::string WebThreeStubServer::transact(Json::Value const& _json)
{
	std::string ret;
	if (!client())
		return ret;
	TransactionJS t = toTransaction(_json);
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

Json::Value WebThreeStubServer::transaction(int const& _i, Json::Value const& _params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(_params);
	return toJson(client()->transaction(hash, _i));
}

Json::Value WebThreeStubServer::uncle(int const& _i, Json::Value const& _params)
{
	if (!client())
		return "";
	
	auto hash = numberOrHash(_params);
	return toJson(client()->uncle(hash, _i));
}

int WebThreeStubServer::watch(string const& _json)
{
	unsigned ret = -1;
	if (!client())
		return ret;
	if (_json.compare("chain") == 0)
		ret = client()->installWatch(dev::eth::ChainChangedFilter);
	else if (_json.compare("pending") == 0)
		ret = client()->installWatch(dev::eth::PendingChangedFilter);
	else
	{
		Json::Reader reader;
		Json::Value object;
		reader.parse(_json, object);
		ret = client()->installWatch(toMessageFilter(object));
	}

	return ret;
}

bool WebThreeStubServer::check(int const& _id)
{
	if (!client())
		return false;
	return client()->checkWatch(_id);
}

bool WebThreeStubServer::killWatch(int const& _id)
{
	if (!client())
		return false;
	client()->uninstallWatch(_id);
	return true;
}

#endif
