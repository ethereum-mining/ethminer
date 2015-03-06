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
/** @file WebThreeStubServerBase.cpp
 * @authors:
 *   Gav Wood <i@gavwood.com>
 *   Marek Kotewicz <marek@ethdev.com>
 * @date 2014
 */

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#include <jsonrpccpp/common/exception.h>
#include <libdevcore/CommonData.h>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libevmcore/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libwebthree/WebThree.h>
#include <libethcore/CommonJS.h>
#include <libwhisper/Message.h>
#include <libwhisper/WhisperHost.h>
#ifndef _MSC_VER
#include <libserpent/funcs.h>
#endif
#include "WebThreeStubServerBase.h"
#include "AccountHolder.h"

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
	res["number"] = toJS(_bi.number);
	res["gasLimit"] = toJS(_bi.gasLimit);
	res["timestamp"] = toJS(_bi.timestamp);
	res["extraData"] = jsFromBinary(_bi.extraData);
	res["nonce"] = toJS(_bi.nonce);
	return res;
}

static Json::Value toJson(dev::eth::Transaction const& _t)
{
	Json::Value res;
	res["hash"] = toJS(_t.sha3());
	res["input"] = jsFromBinary(_t.data());
	res["to"] = toJS(_t.receiveAddress());
	res["from"] = toJS(_t.safeSender());
	res["gas"] = toJS(_t.gas());
	res["gasPrice"] = toJS(_t.gasPrice());
	res["nonce"] = toJS(_t.nonce());
	res["value"] = toJS(_t.value());
	return res;
}

static Json::Value toJson(dev::eth::BlockInfo const& _bi, Transactions const& _ts)
{
	Json::Value res = toJson(_bi);
	res["transactions"] = Json::Value(Json::arrayValue);
	for (Transaction const& t: _ts)
		res["transactions"].append(toJson(t));
	return res;
}

static Json::Value toJson(dev::eth::TransactionSkeleton const& _t)
{
	Json::Value res;
	res["to"] = toJS(_t.to);
	res["from"] = toJS(_t.from);
	res["gas"] = toJS(_t.gas);
	res["gasPrice"] = toJS(_t.gasPrice);
	res["value"] = toJS(_t.value);
	res["data"] = jsFromBinary(_t.data);
	return res;
}

static Json::Value toJson(dev::eth::LocalisedLogEntry const& _e)
{
	Json::Value res;

	res["data"] = jsFromBinary(_e.data);
	res["address"] = toJS(_e.address);
	res["topics"] = Json::Value(Json::arrayValue);
	for (auto const& t: _e.topics)
		res["topics"].append(toJS(t));
	res["number"] = _e.number;
	res["hash"] = toJS(_e.sha3);
	return res;
}

static Json::Value toJson(dev::eth::LocalisedLogEntries const& _es)	// commented to avoid warning. Uncomment once in use @ poC-7.
{
	Json::Value res(Json::arrayValue);
	for (dev::eth::LocalisedLogEntry const& e: _es)
		res.append(toJson(e));
	return res;
}

static Json::Value toJson(std::map<u256, u256> const& _storage)
{
	Json::Value res(Json::objectValue);
	for (auto i: _storage)
		res[toJS(i.first)] = toJS(i.second);
	return res;
}

static dev::eth::LogFilter toLogFilter(Json::Value const& _json)	// commented to avoid warning. Uncomment once in use @ PoC-7.
{
	dev::eth::LogFilter filter;
	if (!_json.isObject() || _json.empty())
		return filter;

	// check only !empty. it should throw exceptions if input params are incorrect
	if (!_json["earliest"].empty())
		filter.withEarliest(jsToInt(_json["earliest"].asString()));
	if (!_json["latest"].empty())
		filter.withLatest(jsToInt(_json["latest"].asString()));
	if (!_json["max"].empty())
		filter.withMax(jsToInt(_json["max"].asString()));
	if (!_json["skip"].empty())
		filter.withSkip(jsToInt(_json["skip"].asString()));
	if (!_json["address"].empty())
	{
		if (_json["address"].isArray())
			for (auto i : _json["address"])
				filter.address(jsToAddress(i.asString()));
		else
			filter.address(jsToAddress(_json["address"].asString()));
	}
	if (!_json["topics"].empty())
	{
		unsigned i = 0;
		for (auto t: _json["topics"])
		{
			for (auto tt: t)
				filter.topic(i, jsToFixed<32>(tt.asString()));
			i++;
		}
	}
	return filter;
}

static shh::Message toMessage(Json::Value const& _json)
{
	shh::Message ret;
	if (!_json["from"].empty())
		ret.setFrom(jsToPublic(_json["from"].asString()));
	if (!_json["to"].empty())
		ret.setTo(jsToPublic(_json["to"].asString()));
	if (!_json["payload"].empty())
		ret.setPayload(jsToBytes(_json["payload"].asString()));
	return ret;
}

static shh::Envelope toSealed(Json::Value const& _json, shh::Message const& _m, Secret _from)
{
	unsigned ttl = 50;
	unsigned workToProve = 50;
	shh::BuildTopic bt;

	if (!_json["ttl"].empty())
		ttl = jsToInt(_json["ttl"].asString());
	
	if (!_json["workToProve"].empty())
		workToProve = jsToInt(_json["workToProve"].asString());
	
	if (!_json["topics"].empty())
		for (auto i: _json["topics"])
			bt.shift(jsToBytes(i.asString()));
	
	return _m.seal(_from, bt, ttl, workToProve);
}

static pair<shh::FullTopic, Public> toWatch(Json::Value const& _json)
{
	shh::BuildTopic bt;
	Public to;

	if (!_json["to"].empty())
		to = jsToPublic(_json["to"].asString());

	if (!_json["topics"].empty())
		for (auto i: _json["topics"])
			bt.shift(jsToBytes(i.asString()));
	
	return make_pair(bt, to);
}

static Json::Value toJson(h256 const& _h, shh::Envelope const& _e, shh::Message const& _m)
{
	Json::Value res;
	res["hash"] = toJS(_h);
	res["expiry"] = toJS(_e.expiry());
	res["sent"] = toJS(_e.sent());
	res["ttl"] = toJS(_e.ttl());
	res["workProved"] = toJS(_e.workProved());
	res["topics"] = Json::Value(Json::arrayValue);
	for (auto const& t: _e.topic())
		res["topics"].append(toJS(t));
	res["payload"] = toJS(_m.payload());
	res["from"] = toJS(_m.from());
	res["to"] = toJS(_m.to());
	return res;
}

static int toBlockNumber(string const& _string)
{
	if (_string.compare("latest") == 0)
		return -1;
	return jsToInt(_string);
}

WebThreeStubServerBase::WebThreeStubServerBase(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _accounts):
	AbstractWebThreeStubServer(_conn), m_accounts(make_shared<AccountHolder>(std::bind(&WebThreeStubServerBase::client, this)))
{
	m_accounts->setAccounts(_accounts);
}

void WebThreeStubServerBase::setIdentities(std::vector<dev::KeyPair> const& _ids)
{
	m_ids.clear();
	for (auto i: _ids)
		m_ids[i.pub()] = i.secret();
}

std::string WebThreeStubServerBase::web3_sha3(std::string const& _param1)
{
	return toJS(sha3(jsToBytes(_param1)));
}

string WebThreeStubServerBase::net_peerCount()
{
	return toJS(network()->peerCount());
}

bool WebThreeStubServerBase::net_listening()
{
	return network()->isNetworkStarted();
}

std::string WebThreeStubServerBase::eth_coinbase()
{
	return toJS(client()->address());
}

bool WebThreeStubServerBase::eth_mining()
{
	return client()->isMining();
}

std::string WebThreeStubServerBase::eth_gasPrice()
{
	return toJS(10 * dev::eth::szabo);
}

Json::Value WebThreeStubServerBase::eth_accounts()
{
	Json::Value ret(Json::arrayValue);
	for (auto const& i: m_accounts->getAllAccounts())
		ret.append(toJS(i));
	return ret;
}

string WebThreeStubServerBase::eth_blockNumber()
{
	return toJS(client()->number());
}


std::string WebThreeStubServerBase::eth_getBalance(string const& _address, string const& _blockNumber)
{
	Address address;
	int number;
	
	try
	{
		address = jsToAddress(_address);
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->balanceAt(address, number));
}


Json::Value WebThreeStubServerBase::eth_getStorage(string const& _address, string const& _blockNumber)
{
	Address address;
	int number;
	
	try
	{
		address = jsToAddress(_address);
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	//TODO: fix this naming !
	return toJson(client()->storageAt(address, number));
}

std::string WebThreeStubServerBase::eth_getStorageAt(string const& _address, string const& _position, string const& _blockNumber)
{
	Address address;
	u256 position;
	int number;
	
	try
	{
		address = jsToAddress(_address);
		position = jsToU256(_position);
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	//TODO: fix this naming !
	return toJS(client()->stateAt(address, position, number));
}

string WebThreeStubServerBase::eth_getTransactionCount(string const& _address, string const& _blockNumber)
{
	Address address;
	int number;
	
	try
	{
		address = jsToAddress(_address);
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->countAt(address, number));
}

string WebThreeStubServerBase::eth_getBlockTransactionCountByHash(std::string const& _blockHash)
{
	h256 hash;
	
	try
	{
		hash = jsToFixed<32>(_blockHash);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->transactionCount(hash));
}


string WebThreeStubServerBase::eth_getBlockTransactionCountByNumber(string const& _blockNumber)
{
	int number;
	
	try
	{
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->transactionCount(client()->hashFromNumber(number)));
}

string WebThreeStubServerBase::eth_getUncleCountByBlockHash(std::string const& _blockHash)
{
	h256 hash;
	
	try
	{
		hash = jsToFixed<32>(_blockHash);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->uncleCount(hash));
}

string WebThreeStubServerBase::eth_getUncleCountByBlockNumber(string const& _blockNumber)
{
	int number;
	
	try
	{
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->uncleCount(client()->hashFromNumber(number)));
}

std::string WebThreeStubServerBase::eth_getData(string const& _address, string const& _blockNumber)
{
	Address address;
	int number;
	
	try
	{
		address = jsToAddress(_address);
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return jsFromBinary(client()->codeAt(address, number));
}

static TransactionSkeleton toTransaction(Json::Value const& _json)
{
	TransactionSkeleton ret;
	if (!_json.isObject() || _json.empty())
		return ret;
	
	if (!_json["from"].empty())
		ret.from = jsToAddress(_json["from"].asString());
	if (!_json["to"].empty())
		ret.to = jsToAddress(_json["to"].asString());
	else
		ret.creation = true;
	
	if (!_json["value"].empty())
		ret.value = jsToU256(_json["value"].asString());

	if (!_json["gas"].empty())
		ret.gas = jsToU256(_json["gas"].asString());

	if (!_json["gasPrice"].empty())
		ret.gasPrice = jsToU256(_json["gasPrice"].asString());

	if (!_json["data"].empty())							// ethereum.js has preconstructed the data array
		ret.data = jsToBytes(_json["data"].asString());
	
	if (!_json["code"].empty())
		ret.data = jsToBytes(_json["code"].asString());
	return ret;
}

std::string WebThreeStubServerBase::eth_sendTransaction(Json::Value const& _json)
{
	TransactionSkeleton t;
	
	try
	{
		t = toTransaction(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	std::string ret;
	if (!t.from)
		t.from = m_accounts->getDefaultTransactAccount();
	if (t.creation)
		ret = toJS(right160(sha3(rlpList(t.from, client()->countAt(t.from)))));;
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);
	
	if (m_accounts->isRealAccount(t.from))
		authenticate(t, false);
	else if (m_accounts->isProxyAccount(t.from))
		authenticate(t, true);
	
	return ret;
}


std::string WebThreeStubServerBase::eth_call(Json::Value const& _json)
{
	TransactionSkeleton t;
	
	try
	{
		t = toTransaction(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	std::string ret;
	if (!t.from)
		t.from = m_accounts->getDefaultTransactAccount();
	if (!m_accounts->isRealAccount(t.from))
		return ret;
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);
	ret = toJS(client()->call(m_accounts->secretKey(t.from), t.value, t.to, t.data, t.gas, t.gasPrice));
	
	return ret;
}

bool WebThreeStubServerBase::eth_flush()
{
	client()->flushTransactions();
	return true;
}

Json::Value WebThreeStubServerBase::eth_getBlockByHash(string const& _blockHash, bool _includeTransactions)
{
	h256 hash;
	
	try
	{
		hash = jsToFixed<32>(_blockHash);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	if (_includeTransactions) {
		return toJson(client()->blockInfo(hash), client()->transactions(hash));
	}
	
	return toJson(client()->blockInfo(hash));
}

Json::Value WebThreeStubServerBase::eth_getBlockByNumber(string const& _blockNumber, bool _includeTransactions)
{
	int number;
	
	try
	{
		number = toBlockNumber(_blockNumber);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	h256 hash = client()->hashFromNumber(number);
	
	if (_includeTransactions) {
		return toJson(client()->blockInfo(hash), client()->transactions(hash));
	}
	
	return toJson(client()->blockInfo(hash));
}

Json::Value WebThreeStubServerBase::eth_getTransactionByHash(string const& _transactionHash)
{
	h256 hash;
	
	try
	{
		hash = jsToFixed<32>(_transactionHash);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
//	return toJson(client()->transaction(hash, index));
	// TODO:
	return "";
}

Json::Value WebThreeStubServerBase::eth_getTransactionByBlockHashAndIndex(string const& _blockHash, string const& _transactionIndex)
{
	h256 hash;
	unsigned index;
	
	try
	{
		hash = jsToFixed<32>(_blockHash);
		index = jsToInt(_transactionIndex);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJson(client()->transaction(hash, index));
}

Json::Value WebThreeStubServerBase::eth_getTransactionByBlockNumberAndIndex(string const& _blockNumber, string const& _transactionIndex)
{
	int number;
	unsigned index;
	
	try
	{
		number = jsToInt(_blockNumber);
		index = jsToInt(_transactionIndex);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJson(client()->transaction(client()->hashFromNumber(number), index));
}

Json::Value WebThreeStubServerBase::eth_getUncleByBlockHashAndIndex(string const& _blockHash, string const& _uncleIndex)
{
	h256 hash;
	unsigned index;
	
	try
	{
		hash = jsToFixed<32>(_blockHash);
		index = jsToInt(_uncleIndex);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJson(client()->uncle(hash, index));
}

Json::Value WebThreeStubServerBase::eth_getUncleByBlockNumberAndIndex(string const& _blockNumber, string const& _uncleIndex)
{
	int number;
	unsigned index;
	
	try
	{
		number = toBlockNumber(_blockNumber);
		index = jsToInt(_uncleIndex);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJson(client()->uncle(client()->hashFromNumber(number), index));
}

Json::Value WebThreeStubServerBase::eth_getCompilers()
{
	Json::Value ret(Json::arrayValue);
	ret.append("lll");
	ret.append("solidity");
#ifndef _MSC_VER
	ret.append("serpent");
#endif
	return ret;
}


string WebThreeStubServerBase::eth_compileLLL(string const& _code)
{
	// TODO throw here jsonrpc errors
	string res;
	vector<string> errors;
	res = toJS(dev::eth::compileLLL(_code, true, &errors));
	cwarn << "LLL compilation errors: " << errors;
	return res;
}

string WebThreeStubServerBase::eth_compileSerpent(string const& _code)
{
	// TODO throw here jsonrpc errors
	string res;
#ifndef _MSC_VER
	try
	{
		res = toJS(dev::asBytes(::compile(_code)));
	}
	catch (string err)
	{
		cwarn << "Solidity compilation error: " << err;
	}
	catch (...)
	{
		cwarn << "Uncought serpent compilation exception";
	}
#endif
	return res;
}

string WebThreeStubServerBase::eth_compileSolidity(string const& _code)
{
	// TOOD throw here jsonrpc errors
	string res;
	dev::solidity::CompilerStack compiler;
	try
	{
		res = toJS(compiler.compile(_code, true));
	}
	catch (dev::Exception const& exception)
	{
		ostringstream error;
		solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
		cwarn << "Solidity compilation error: " << error.str();
	}
	catch (...)
	{
		cwarn << "Uncought solidity compilation exception";
	}
	return res;
}

string WebThreeStubServerBase::eth_newFilter(Json::Value const& _json)
{
	LogFilter filter;
	
	try
	{
		filter = toLogFilter(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(client()->installWatch(filter));
}

string WebThreeStubServerBase::eth_newBlockFilter(string const& _filter)
{
	h256 filter;
	
	if (_filter.compare("chain") == 0)
		filter = dev::eth::ChainChangedFilter;
	else if (_filter.compare("pending") == 0)
		filter = dev::eth::PendingChangedFilter;
	else
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	
	return toJS(client()->installWatch(filter));
}

bool WebThreeStubServerBase::eth_uninstallFilter(string const& _filterId)
{
	unsigned id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	// TODO: throw an error if there is no watch with given id?
	client()->uninstallWatch(id);
	return true;
}

Json::Value WebThreeStubServerBase::eth_getFilterChanges(string const& _filterId)
{
	unsigned id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	// TODO: throw an error if there is no watch with given id?
	auto entries = client()->checkWatch(id);
	if (entries.size())
		cnote << "FIRING WATCH" << id << entries.size();
	return toJson(entries);
}

Json::Value WebThreeStubServerBase::eth_getFilterLogs(string const& _filterId)
{
	unsigned id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	// TODO: throw an error if there is no watch with given id?
	return toJson(client()->logs(id));
}

Json::Value WebThreeStubServerBase::eth_getLogs(Json::Value const& _json)
{
	LogFilter filter;
	
	try
	{
		filter = toLogFilter(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJson(client()->logs(filter));
}

Json::Value WebThreeStubServerBase::eth_getWork()
{
	Json::Value ret(Json::arrayValue);
	auto r = client()->getWork();
	ret.append(toJS(r.first));
	ret.append(toJS(r.second));
	return ret;
}

bool WebThreeStubServerBase::eth_submitWork(string const& _nonce, std::string const& _mixHash)
{
	
	Nonce nonce;
	h256 mixHash;
	
	try
	{
		nonce = jsToFixed<Nonce::size>(_nonce);
		mixHash = jsToFixed<32>(_mixHash);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return client()->submitWork(ProofOfWork::Proof{nonce, mixHash});
}

string WebThreeStubServerBase::eth_register(string const& _address)
{
	Address address;
	
	try
	{
		address = jsToAddress(_address);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return toJS(m_accounts->addProxyAccount(address));
}

bool WebThreeStubServerBase::eth_unregister(string const& _accountId)
{
	unsigned id;
	
	try
	{
		id = jsToInt(_accountId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}

	// TODO: throw an error on no account with given id
	return m_accounts->removeProxyAccount(id);
}

Json::Value WebThreeStubServerBase::eth_queuedTransactions(string const& _accountId)
{
	unsigned id;
	
	try
	{
		id = jsToInt(_accountId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}

	// TODO: throw an error on no account with given id
	Json::Value ret(Json::arrayValue);
	for (TransactionSkeleton const& t: m_accounts->getQueuedTransactions(id))
		ret.append(toJson(t));
	m_accounts->clearQueue(id);
	return ret;
}

bool WebThreeStubServerBase::db_put(string const& _name, string const& _key, string const& _value)
{
	db()->put(_name, _key,_value);
	return true;
}

string WebThreeStubServerBase::db_get(string const& _name, string const& _key)
{
	return db()->get(_name, _key);;
}

bool WebThreeStubServerBase::shh_post(Json::Value const& _json)
{
	shh::Message m;
	try
	{
		m = toMessage(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	Secret from;
	if (m.from() && m_ids.count(m.from()))
	{
		cwarn << "Silently signing message from identity" << m.from().abridged() << ": User validation hook goes here.";
		// TODO: insert validification hook here.
		from = m_ids[m.from()];
	}
	
	face()->inject(toSealed(_json, m, from));
	return true;
}

std::string WebThreeStubServerBase::shh_newIdentity()
{
	KeyPair kp = KeyPair::create();
	m_ids[kp.pub()] = kp.secret();
	return toJS(kp.pub());
}

bool WebThreeStubServerBase::shh_hasIdentity(string const& _identity)
{
	Public identity;
	
	try
	{
		identity = jsToPublic(_identity);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	return m_ids.count(identity) > 0;
}


string WebThreeStubServerBase::shh_newGroup(string const& _id, string const& _who)
{
	(void)_id;
	(void)_who;
	return "";
}

string WebThreeStubServerBase::shh_addToGroup(std::string const& _group, string const& _who)
{
	(void)_group;
	(void)_who;
	return "";
}

string WebThreeStubServerBase::shh_newFilter(Json::Value const& _json)
{
	pair<shh::FullTopic, Public> w;
	
	try
	{
		w = toWatch(_json);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	auto ret = face()->installWatch(w.first);
	m_shhWatches.insert(make_pair(ret, w.second));
	return toJS(ret);
}


bool WebThreeStubServerBase::shh_uninstallFilter(string const& _filterId)
{
	int id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	face()->uninstallWatch(id);
	return true;
}

Json::Value WebThreeStubServerBase::shh_getFilterChanges(string const& _filterId)
{
	int id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	Json::Value ret(Json::arrayValue);
	auto pub = m_shhWatches[id];
	if (!pub || m_ids.count(pub))
		for (h256 const& h: face()->checkWatch(id))
		{
			auto e = face()->envelope(h);
			shh::Message m;
			if (pub)
			{
				cwarn << "Silently decrypting message from identity" << pub.abridged() << ": User validation hook goes here.";
				m = e.open(face()->fullTopic(id), m_ids[pub]);
			}
			else
				m = e.open(face()->fullTopic(id));
			if (!m)
				continue;
			ret.append(toJson(h, e, m));
		}

	return ret;
}

Json::Value WebThreeStubServerBase::shh_getMessages(string const& _filterId)
{
	int id;
	
	try
	{
		id = jsToInt(_filterId);
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException(jsonrpc::Errors::ERROR_RPC_INVALID_PARAMS);
	}
	
	Json::Value ret(Json::arrayValue);
	auto pub = m_shhWatches[id];
	if (!pub || m_ids.count(pub))
		for (h256 const& h: face()->watchMessages(id))
		{
			auto e = face()->envelope(h);
			shh::Message m;
			if (pub)
			{
				cwarn << "Silently decrypting message from identity" << pub.abridged() << ": User validation hook goes here.";
				m = e.open(face()->fullTopic(id), m_ids[pub]);
			}
			else
				m = e.open(face()->fullTopic(id));
			if (!m)
				continue;
			ret.append(toJson(h, e, m));
		}
	return ret;
}

void WebThreeStubServerBase::authenticate(TransactionSkeleton const& _t, bool _toProxy)
{
	if (_toProxy)
		m_accounts->queueTransaction(_t);
	else if (_t.to)
		client()->transact(m_accounts->secretKey(_t.from), _t.value, _t.to, _t.data, _t.gas, _t.gasPrice);
	else
		client()->transact(m_accounts->secretKey(_t.from), _t.value, _t.data, _t.gas, _t.gasPrice);
}

void WebThreeStubServerBase::setAccounts(const std::vector<KeyPair>& _accounts)
{
	m_accounts->setAccounts(_accounts);
}
