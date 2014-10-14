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

EthStubServer::EthStubServer(jsonrpc::AbstractServerConnector* _conn, WebThreeDirect& _web3):
	AbstractEthStubServer(_conn),
	m_web3(_web3)
{
}

dev::eth::Interface* EthStubServer::client() const
{
    return &(*m_web3.ethereum());
}

std::string EthStubServer::balanceAt(const string &a, const int& block)
{
    return jsToDecimal(toJS(client()->balanceAt(jsToAddress(a), block)));
}

//TODO BlockDetails?
Json::Value EthStubServer::block(const string &numberOrHash)
{
    auto n = jsToU256(numberOrHash);
    auto h = n < client()->number() ? client()->hashFromNumber((unsigned)n) : ::jsToFixed<32>(numberOrHash);
    return toJson(client()->blockInfo(h));
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

std::string EthStubServer::call(const Json::Value &json)
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

std::string EthStubServer::codeAt(const string &a, const int& block)
{
    return client() ? jsFromBinary(client()->codeAt(jsToAddress(a), block)) : "";
}

std::string EthStubServer::coinbase()
{
    return client() ? toJS(client()->address()) : "";
}

double EthStubServer::countAt(const string &a, const int& block)
{
    return client() ? (double)(uint64_t)client()->countAt(jsToAddress(a), block) : 0;
}

int EthStubServer::defaultBlock()
{
    return client() ? client()->getDefault() : 0;
}

std::string EthStubServer::fromAscii(const int& padding, const std::string& s)
{
    return jsFromBinary(s, padding);
}

double EthStubServer::fromFixed(const string &s)
{
    return jsFromFixed(s);
}

std::string EthStubServer::gasPrice()
{
    return toJS(10 * dev::eth::szabo);
}

//TODO
bool EthStubServer::isListening()
{
    return /*client() ? client()->haveNetwork() :*/ false;
}

bool EthStubServer::isMining()
{
    return client() ? client()->isMining() : false;
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

std::string EthStubServer::lll(const string &s)
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

Json::Value EthStubServer::messages(const Json::Value &json)
{
    Json::Value res;
    if (!client())
        return  res;
    return toJson(client()->messages(toMessageFilter(json)));
}

int EthStubServer::number()
{
    return client() ? client()->number() + 1 : 0;
}

//TODO!
int EthStubServer::peerCount()
{
    return /*client() ? (unsigned)client()->peerCount() :*/ 0;
    //return m_web3.peerCount();
}

std::string EthStubServer::secretToAddress(const string &s)
{
    return toJS(KeyPair(jsToSecret(s)).address());
}

Json::Value EthStubServer::setListening(const bool &l)
{
    if (!client())
        return Json::nullValue;

/*	if (l)
        client()->startNetwork();
    else
        client()->stopNetwork();*/
    return Json::nullValue;
}

Json::Value EthStubServer::setMining(const bool &l)
{
    if (!client())
        return Json::nullValue;

    if (l)
        client()->startMining();
    else
        client()->stopMining();
    return Json::nullValue;
}

std::string EthStubServer::sha3(const string &s)
{
    return toJS(dev::eth::sha3(jsToBytes(s)));
}

std::string EthStubServer::stateAt(const string &a, const int& block, const string &s)
{
    return client() ? toJS(client()->stateAt(jsToAddress(a), jsToU256(s), block)) : "";
}

std::string EthStubServer::toAscii(const string &s)
{
    return jsToBinary(s);
}

std::string EthStubServer::toDecimal(const string &s)
{
    return jsToDecimal(s);
}

std::string EthStubServer::toFixed(const double &s)
{
    return jsToFixed(s);
}

std::string EthStubServer::transact(const Json::Value &json)
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

Json::Value EthStubServer::transaction(const int &i, const string &numberOrHash)
{
    if (!client()){
        return Json::Value();
    }
    auto n = jsToU256(numberOrHash);
    auto h = n < client()->number() ? client()->hashFromNumber((unsigned)n) : jsToFixed<32>(numberOrHash);
    return toJson(client()->transaction(h, i));
}

Json::Value EthStubServer::uncle(const int &i, const string &numberOrHash)
{
    auto n = jsToU256(numberOrHash);
    auto h  = n < client()->number() ? client()->hashFromNumber((unsigned)n) : jsToFixed<32>(numberOrHash);
    return client() ? toJson(client()->uncle(h, i)) : Json::Value();
}

//TODO watch!
std::string EthStubServer::watch(const string &json)
{

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
