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
#if ETH_SOLIDITY || !ETH_TRUE
#include <libsolidity/CompilerStack.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/SourceReferenceFormatter.h>
#endif
#include <libevmcore/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libwebthree/WebThree.h>
#include <libethcore/CommonJS.h>
#include <libwhisper/Message.h>
#include <libwhisper/WhisperHost.h>
#if ETH_SERPENT || !ETH_TRUE
#include <libserpent/funcs.h>
#endif
#include "WebThreeStubServerBase.h"
#include "AccountHolder.h"

using namespace std;
using namespace jsonrpc;
using namespace dev;
using namespace dev::eth;

#if ETH_DEBUG
const unsigned dev::SensibleHttpThreads = 1;
#else
const unsigned dev::SensibleHttpThreads = 4;
#endif
const unsigned dev::SensibleHttpPort = 8545;

static Json::Value toJson(dev::eth::BlockInfo const& _bi)
{
	Json::Value res;
	if (_bi)
	{
		res["hash"] = toJS(_bi.hash());
		res["parentHash"] = toJS(_bi.parentHash);
		res["sha3Uncles"] = toJS(_bi.sha3Uncles);
		res["miner"] = toJS(_bi.coinbaseAddress);
		res["stateRoot"] = toJS(_bi.stateRoot);
		res["transactionsRoot"] = toJS(_bi.transactionsRoot);
		res["difficulty"] = toJS(_bi.difficulty);
		res["number"] = toJS(_bi.number);
		res["gasUsed"] = toJS(_bi.gasUsed);
		res["gasLimit"] = toJS(_bi.gasLimit);
		res["timestamp"] = toJS(_bi.timestamp);
		res["extraData"] = toJS(_bi.extraData);
		res["nonce"] = toJS(_bi.nonce);
		res["logsBloom"] = toJS(_bi.logBloom);
	}
	return res;
}

static Json::Value toJson(dev::eth::Transaction const& _t, std::pair<h256, unsigned> _location, BlockNumber _blockNumber)
{
	Json::Value res;
	if (_t)
	{
		res["hash"] = toJS(_t.sha3());
		res["input"] = toJS(_t.data());
		res["to"] = _t.isCreation() ? Json::Value() : toJS(_t.receiveAddress());
		res["from"] = toJS(_t.safeSender());
		res["gas"] = toJS(_t.gas());
		res["gasPrice"] = toJS(_t.gasPrice());
		res["nonce"] = toJS(_t.nonce());
		res["value"] = toJS(_t.value());
		res["blockHash"] = toJS(_location.first);
		res["transactionIndex"] = toJS(_location.second);
		res["blockNumber"] = toJS(_blockNumber);
	}
	return res;
}

static Json::Value toJson(dev::eth::BlockInfo const& _bi, BlockDetails const& _bd, UncleHashes const& _us, Transactions const& _ts)
{
	Json::Value res = toJson(_bi);
	if (_bi)
	{
		res["totalDifficulty"] = toJS(_bd.totalDifficulty);
		res["uncles"] = Json::Value(Json::arrayValue);
		for (h256 h: _us)
			res["uncles"].append(toJS(h));
		res["transactions"] = Json::Value(Json::arrayValue);
		for (unsigned i = 0; i < _ts.size(); i++)
			res["transactions"].append(toJson(_ts[i], std::make_pair(_bi.hash(), i), (BlockNumber)_bi.number));
	}
	return res;
}

static Json::Value toJson(dev::eth::BlockInfo const& _bi, BlockDetails const& _bd, UncleHashes const& _us, TransactionHashes const& _ts)
{
	Json::Value res = toJson(_bi);
	if (_bi)
	{
		res["totalDifficulty"] = toJS(_bd.totalDifficulty);
		res["uncles"] = Json::Value(Json::arrayValue);
		for (h256 h: _us)
			res["uncles"].append(toJS(h));
		res["transactions"] = Json::Value(Json::arrayValue);
		for (h256 const& t: _ts)
			res["transactions"].append(toJS(t));
	}
	return res;
}

static Json::Value toJson(dev::eth::TransactionSkeleton const& _t)
{
	Json::Value res;
	res["to"] = _t.creation ? Json::Value() : toJS(_t.to);
	res["from"] = toJS(_t.from);
	res["gas"] = toJS(_t.gas);
	res["gasPrice"] = toJS(_t.gasPrice);
	res["value"] = toJS(_t.value);
	res["data"] = toJS(_t.data, 32);
	return res;
}

static Json::Value toJson(dev::eth::Transaction const& _t)
{
	Json::Value res;
	res["to"] = _t.isCreation() ? Json::Value() : toJS(_t.to());
	res["from"] = toJS(_t.from());
	res["gas"] = toJS(_t.gas());
	res["gasPrice"] = toJS(_t.gasPrice());
	res["value"] = toJS(_t.value());
	res["data"] = toJS(_t.data(), 32);
	res["nonce"] = toJS(_t.nonce());
	res["hash"] = toJS(_t.sha3(WithSignature));
	res["sighash"] = toJS(_t.sha3(WithoutSignature));
	res["r"] = toJS(_t.signature().r);
	res["s"] = toJS(_t.signature().s);
	res["v"] = toJS(_t.signature().v);
	return res;
}

static Json::Value toJson(dev::eth::LocalisedLogEntry const& _e)
{
	Json::Value res;
	if (_e.transactionHash)
	{
		res["data"] = toJS(_e.data);
		res["address"] = toJS(_e.address);
		res["topics"] = Json::Value(Json::arrayValue);
		for (auto const& t: _e.topics)
			res["topics"].append(toJS(t));
		res["number"] = _e.number;
		res["hash"] = toJS(_e.transactionHash);
	}
	return res;
}

static Json::Value toJson(dev::eth::LocalisedLogEntries const& _es)
{
	Json::Value res(Json::arrayValue);
	for (dev::eth::LocalisedLogEntry const& e: _es)
		res.append(toJson(e));
	return res;
}

static Json::Value toJson(map<u256, u256> const& _storage)
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
	if (!_json["fromBlock"].empty())
		filter.withEarliest(jsToBlockNumber(_json["fromBlock"].asString()));
	if (!_json["toBlock"].empty())
		filter.withLatest(jsToBlockNumber(_json["toBlock"].asString()));
	if (!_json["address"].empty())
	{
		if (_json["address"].isArray())
			for (auto i : _json["address"])
				filter.address(jsToAddress(i.asString()));
		else
			filter.address(jsToAddress(_json["address"].asString()));
	}
	if (!_json["topics"].empty())
		for (unsigned i = 0; i < _json["topics"].size(); i++)
		{
			if (_json["topics"][i].isArray())
			{
				for (auto t: _json["topics"][i])
					if (!t.isNull())
						filter.topic(i, jsToFixed<32>(t.asString()));
			}
			else if (!_json["topics"][i].isNull()) // if it is anything else then string, it should and will fail
				filter.topic(i, jsToFixed<32>(_json["topics"][i].asString()));
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
		{
			if (i.isArray())
			{
				for (auto j: i)
					if (!j.isNull())
						bt.shift(jsToBytes(j.asString()));
			}
			else if (!i.isNull()) // if it is anything else then string, it should and will fail
				bt.shift(jsToBytes(i.asString()));
		}
	
	return _m.seal(_from, bt, ttl, workToProve);
}

static pair<shh::Topics, Public> toWatch(Json::Value const& _json)
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

WebThreeStubServerBase::WebThreeStubServerBase(AbstractServerConnector& _conn, std::shared_ptr<dev::eth::AccountHolder> const& _ethAccounts, vector<dev::KeyPair> const& _sshAccounts):
	AbstractWebThreeStubServer(_conn),
	m_ethAccounts(_ethAccounts)
{
	setIdentities(_sshAccounts);
}

void WebThreeStubServerBase::setIdentities(vector<dev::KeyPair> const& _ids)
{
	m_shhIds.clear();
	for (auto i: _ids)
		m_shhIds[i.pub()] = i.secret();
}

string WebThreeStubServerBase::web3_sha3(string const& _param1)
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

string WebThreeStubServerBase::eth_protocolVersion()
{
	return toJS(eth::c_protocolVersion);
}

string WebThreeStubServerBase::eth_coinbase()
{
	return toJS(client()->address());
}

string WebThreeStubServerBase::eth_hashrate()
{
	return toJS(client()->hashrate());
}

bool WebThreeStubServerBase::eth_mining()
{
	return client()->isMining();
}

string WebThreeStubServerBase::eth_gasPrice()
{
	return toJS(10 * dev::eth::szabo);
}

Json::Value WebThreeStubServerBase::eth_accounts()
{
	Json::Value ret(Json::arrayValue);
	for (auto const& i: m_ethAccounts->allAccounts())
		ret.append(toJS(i));
	return ret;
}

string WebThreeStubServerBase::eth_blockNumber()
{
	return toJS(client()->number());
}


string WebThreeStubServerBase::eth_getBalance(string const& _address, string const& _blockNumber)
{
	try
	{
		return toJS(client()->balanceAt(jsToAddress(_address), jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getStorageAt(string const& _address, string const& _position, string const& _blockNumber)
{
	try
	{
		return toJS(client()->stateAt(jsToAddress(_address), jsToU256(_position), jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getTransactionCount(string const& _address, string const& _blockNumber)
{
	try
	{
		return toJS(client()->countAt(jsToAddress(_address), jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getBlockTransactionCountByHash(string const& _blockHash)
{
	try
	{
		return toJS(client()->transactionCount(jsToFixed<32>(_blockHash)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}


string WebThreeStubServerBase::eth_getBlockTransactionCountByNumber(string const& _blockNumber)
{
	try
	{
		return toJS(client()->transactionCount(jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getUncleCountByBlockHash(string const& _blockHash)
{
	try
	{
		return toJS(client()->uncleCount(jsToFixed<32>(_blockHash)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getUncleCountByBlockNumber(string const& _blockNumber)
{
	try
	{
		return toJS(client()->uncleCount(jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_getCode(string const& _address, string const& _blockNumber)
{
	try
	{
		return toJS(client()->codeAt(jsToAddress(_address), jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

static TransactionSkeleton toTransaction(Json::Value const& _json)
{
	TransactionSkeleton ret;
	if (!_json.isObject() || _json.empty())
		return ret;
	
	if (!_json["from"].empty())
		ret.from = jsToAddress(_json["from"].asString());
	if (!_json["to"].empty() && _json["to"].asString() != "0x")
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

string WebThreeStubServerBase::eth_sendTransaction(Json::Value const& _json)
{
	try
	{
		string ret;
		TransactionSkeleton t = toTransaction(_json);
	
		if (!t.from)
			t.from = m_ethAccounts->defaultTransactAccount();
		if (t.creation)
			ret = toJS(right160(sha3(rlpList(t.from, client()->countAt(t.from)))));;
		if (t.gasPrice == UndefinedU256)
			t.gasPrice = 10 * dev::eth::szabo;		// TODO: should be determined by user somehow.
		if (t.gas == UndefinedU256)
			t.gas = min<u256>(client()->gasLimitRemaining() / 5, client()->balanceAt(t.from) / t.gasPrice);

		m_ethAccounts->authenticate(t);
	
		return ret;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_signTransaction(Json::Value const& _json)
{
	try
	{
		string ret;
		TransactionSkeleton t = toTransaction(_json);

		if (!t.from)
			t.from = m_ethAccounts->defaultTransactAccount();
		if (t.creation)
			ret = toJS(right160(sha3(rlpList(t.from, client()->countAt(t.from)))));;
		if (t.gasPrice == UndefinedU256)
			t.gasPrice = 10 * dev::eth::szabo;		// TODO: should be determined by user somehow.
		if (t.gas == UndefinedU256)
			t.gas = min<u256>(client()->gasLimitRemaining() / 5, client()->balanceAt(t.from) / t.gasPrice);

		m_ethAccounts->authenticate(t);

		return toJS((t.creation ? Transaction(t.value, t.gasPrice, t.gas, t.data) : Transaction(t.value, t.gasPrice, t.gas, t.to, t.data)).sha3(WithoutSignature));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_inspectTransaction(std::string const& _rlp)
{
	try
	{
		return toJson(Transaction(jsToBytes(_rlp), CheckTransaction::Everything));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

bool WebThreeStubServerBase::eth_injectTransaction(std::string const& _rlp)
{
	try
	{
		return client()->injectTransaction(jsToBytes(_rlp)) == ImportResult::Success;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_call(Json::Value const& _json, string const& _blockNumber)
{
	try
	{
		TransactionSkeleton t = toTransaction(_json);
		if (!t.from)
			t.from = m_ethAccounts->defaultTransactAccount();
	//	if (!m_accounts->isRealAccount(t.from))
	//		return ret;
		if (t.gasPrice == UndefinedU256)
			t.gasPrice = 10 * dev::eth::szabo;
		if (t.gas == UndefinedU256)
			t.gas = client()->gasLimitRemaining();

		return toJS(client()->call(t.from, t.value, t.to, t.data, t.gas, t.gasPrice, jsToBlockNumber(_blockNumber), FudgeFactor::Lenient).output);
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

bool WebThreeStubServerBase::eth_flush()
{
	client()->flushTransactions();
	return true;
}

Json::Value WebThreeStubServerBase::eth_getBlockByHash(string const& _blockHash, bool _includeTransactions)
{
	try
	{
		auto h = jsToFixed<32>(_blockHash);
		if (_includeTransactions)
			return toJson(client()->blockInfo(h), client()->blockDetails(h), client()->uncleHashes(h), client()->transactions(h));
		else
			return toJson(client()->blockInfo(h), client()->blockDetails(h), client()->uncleHashes(h), client()->transactionHashes(h));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getBlockByNumber(string const& _blockNumber, bool _includeTransactions)
{
	try
	{
		auto h = jsToBlockNumber(_blockNumber);
		if (_includeTransactions)
			return toJson(client()->blockInfo(h), client()->blockDetails(h), client()->uncleHashes(h), client()->transactions(h));
		else
			return toJson(client()->blockInfo(h), client()->blockDetails(h), client()->uncleHashes(h), client()->transactionHashes(h));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getTransactionByHash(string const& _transactionHash)
{
	try
	{
		h256 h = jsToFixed<32>(_transactionHash);
		auto l = client()->transactionLocation(h);
		return toJson(client()->transaction(h), l, client()->numberFromHash(l.first));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getTransactionByBlockHashAndIndex(string const& _blockHash, string const& _transactionIndex)
{
	try
	{
		h256 bh = jsToFixed<32>(_blockHash);
		unsigned ti = jsToInt(_transactionIndex);
		Transaction t = client()->transaction(bh, ti);
		return toJson(t, make_pair(bh, ti), client()->numberFromHash(bh));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getTransactionByBlockNumberAndIndex(string const& _blockNumber, string const& _transactionIndex)
{
	try
	{
		BlockNumber bn = jsToBlockNumber(_blockNumber);
		unsigned ti = jsToInt(_transactionIndex);
		Transaction t = client()->transaction(bn, ti);
		return toJson(t, make_pair(client()->hashFromNumber(bn), ti), bn);
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getUncleByBlockHashAndIndex(string const& _blockHash, string const& _uncleIndex)
{
	try
	{
		return toJson(client()->uncle(jsToFixed<32>(_blockHash), jsToInt(_uncleIndex)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getUncleByBlockNumberAndIndex(string const& _blockNumber, string const& _uncleIndex)
{
	try
	{
		return toJson(client()->uncle(jsToBlockNumber(_blockNumber), jsToInt(_uncleIndex)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getCompilers()
{
	Json::Value ret(Json::arrayValue);
	ret.append("lll");
#if ETH_SOLIDITY || !TRUE
	ret.append("solidity");
#endif
#if ETH_SERPENT || !TRUE
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
	(void)_code;
#if ETH_SERPENT || !ETH_TRUE
	try
	{
		res = toJS(dev::asBytes(::compile(_code)));
	}
	catch (string err)
	{
		cwarn << "Serpent compilation error: " << err;
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
	(void)_code;
	string res;
#if ETH_SOLIDITY || !ETH_TRUE
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
#endif
	return res;
}

string WebThreeStubServerBase::eth_newFilter(Json::Value const& _json)
{
	try
	{
		return toJS(client()->installWatch(toLogFilter(_json)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_newBlockFilter(string const& _filter)
{
	h256 filter;
	
	if (_filter.compare("chain") == 0 || _filter.compare("latest") == 0)
		filter = dev::eth::ChainChangedFilter;
	else if (_filter.compare("pending") == 0)
		filter = dev::eth::PendingChangedFilter;
	else
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	
	return toJS(client()->installWatch(filter));
}

bool WebThreeStubServerBase::eth_uninstallFilter(string const& _filterId)
{
	try
	{
		return client()->uninstallWatch(jsToInt(_filterId));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
	

}

Json::Value WebThreeStubServerBase::eth_getFilterChanges(string const& _filterId)
{
	try
	{
		int id = jsToInt(_filterId);
		auto entries = client()->checkWatch(id);
		if (entries.size())
			cnote << "FIRING WATCH" << id << entries.size();
		return toJson(entries);
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getFilterLogs(string const& _filterId)
{
	try
	{
		return toJson(client()->logs(jsToInt(_filterId)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getLogs(Json::Value const& _json)
{
	try
	{
		return toJson(client()->logs(toLogFilter(_json)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getWork()
{
	Json::Value ret(Json::arrayValue);
	auto r = client()->getWork();
	ret.append(toJS(r.headerHash));
	ret.append(toJS(r.seedHash));
	ret.append(toJS(r.boundary));
	return ret;
}

bool WebThreeStubServerBase::eth_submitWork(string const& _nonce, string const&, string const& _mixHash)
{
	try
	{
		return client()->submitWork(ProofOfWork::Solution{jsToFixed<Nonce::size>(_nonce), jsToFixed<32>(_mixHash)});
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_register(string const& _address)
{
	try
	{
		return toJS(m_ethAccounts->addProxyAccount(jsToAddress(_address)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

bool WebThreeStubServerBase::eth_unregister(string const& _accountId)
{
	try
	{
		return m_ethAccounts->removeProxyAccount(jsToInt(_accountId));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_fetchQueuedTransactions(string const& _accountId)
{
	try
	{
		auto id = jsToInt(_accountId);
		Json::Value ret(Json::arrayValue);
		// TODO: throw an error on no account with given id
		for (TransactionSkeleton const& t: m_ethAccounts->queuedTransactions(id))
			ret.append(toJson(t));
		m_ethAccounts->clearQueue(id);
		return ret;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
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
	try
	{
		shh::Message m = toMessage(_json);
		Secret from;
		if (m.from() && m_shhIds.count(m.from()))
		{
			cwarn << "Silently signing message from identity" << m.from() << ": User validation hook goes here.";
			// TODO: insert validification hook here.
			from = m_shhIds[m.from()];
		}

		face()->inject(toSealed(_json, m, from));
		return true;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::shh_newIdentity()
{
	KeyPair kp = KeyPair::create();
	m_shhIds[kp.pub()] = kp.secret();
	return toJS(kp.pub());
}

bool WebThreeStubServerBase::shh_hasIdentity(string const& _identity)
{
	try
	{
		return m_shhIds.count(jsToPublic(_identity)) > 0;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}


string WebThreeStubServerBase::shh_newGroup(string const& _id, string const& _who)
{
	(void)_id;
	(void)_who;
	return "";
}

string WebThreeStubServerBase::shh_addToGroup(string const& _group, string const& _who)
{
	(void)_group;
	(void)_who;
	return "";
}

string WebThreeStubServerBase::shh_newFilter(Json::Value const& _json)
{
	
	try
	{
		pair<shh::Topics, Public> w = toWatch(_json);
		auto ret = face()->installWatch(w.first);
		m_shhWatches.insert(make_pair(ret, w.second));
		return toJS(ret);
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

bool WebThreeStubServerBase::shh_uninstallFilter(string const& _filterId)
{
	try
	{
		face()->uninstallWatch(jsToInt(_filterId));
		return true;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::shh_getFilterChanges(string const& _filterId)
{
	try
	{
		Json::Value ret(Json::arrayValue);

		int id = jsToInt(_filterId);
		auto pub = m_shhWatches[id];
		if (!pub || m_shhIds.count(pub))
			for (h256 const& h: face()->checkWatch(id))
			{
				auto e = face()->envelope(h);
				shh::Message m;
				if (pub)
				{
					cwarn << "Silently decrypting message from identity" << pub << ": User validation hook goes here.";
					m = e.open(face()->fullTopics(id), m_shhIds[pub]);
				}
				else
					m = e.open(face()->fullTopics(id));
				if (!m)
					continue;
				ret.append(toJson(h, e, m));
			}

		return ret;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::shh_getMessages(string const& _filterId)
{
	try
	{
		Json::Value ret(Json::arrayValue);

		int id = jsToInt(_filterId);
		auto pub = m_shhWatches[id];
		if (!pub || m_shhIds.count(pub))
			for (h256 const& h: face()->watchMessages(id))
			{
				auto e = face()->envelope(h);
				shh::Message m;
				if (pub)
				{
					cwarn << "Silently decrypting message from identity" << pub << ": User validation hook goes here.";
					m = e.open(face()->fullTopics(id), m_shhIds[pub]);
				}
				else
					m = e.open(face()->fullTopics(id));
				if (!m)
					continue;
				ret.append(toJson(h, e, m));
			}
		return ret;
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}
