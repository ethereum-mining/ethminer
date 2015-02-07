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
	res["gasLimit"] = (int)_bi.gasLimit;
	res["timestamp"] = (int)_bi.timestamp;
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
	res["gas"] = (int)_t.gas();
	res["gasPrice"] = toJS(_t.gasPrice());
	res["nonce"] = toJS(_t.nonce());
	res["value"] = toJS(_t.value());
	return res;
}

static Json::Value toJson(dev::eth::LocalisedLogEntry const& _e)
{
	Json::Value res;
	
	res["data"] = jsFromBinary(_e.data);
	res["address"] = toJS(_e.address);
	for (auto const& t: _e.topics)
		res["topic"].append(toJS(t));
	res["number"] = _e.number;
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

	if (_json["earliest"].isInt())
		filter.withEarliest(_json["earliest"].asInt());
	if (_json["latest"].isInt())
		filter.withLatest(_json["lastest"].asInt());
	if (_json["max"].isInt())
		filter.withMax(_json["max"].asInt());
	if (_json["skip"].isInt())
		filter.withSkip(_json["skip"].asInt());
	if (!_json["address"].empty())
	{
		if (_json["address"].isArray())
		{
			for (auto i : _json["address"])
				if (i.isString())
					filter.address(jsToAddress(i.asString()));
		}
		else if (_json["address"].isString())
			filter.address(jsToAddress(_json["address"].asString()));
	}
	if (!_json["topic"].empty() && _json["topic"].isArray())
	{
		unsigned i = 0;
		for (auto t: _json["topic"])
		{
			if (t.isArray())
				for (auto tt: t)
					filter.topic(i, jsToFixed<32>(tt.asString()));
			else if (t.isString())
				filter.topic(i, jsToFixed<32>(t.asString()));
			i++;
		}
	}
	return filter;
}

static shh::Message toMessage(Json::Value const& _json)
{
	shh::Message ret;
	if (_json["from"].isString())
		ret.setFrom(jsToPublic(_json["from"].asString()));
	if (_json["to"].isString())
		ret.setTo(jsToPublic(_json["to"].asString()));
	if (_json["payload"].isString())
		ret.setPayload(jsToBytes(_json["payload"].asString()));
	return ret;
}

static shh::Envelope toSealed(Json::Value const& _json, shh::Message const& _m, Secret _from)
{
	unsigned ttl = 50;
	unsigned workToProve = 50;
	shh::BuildTopic bt;

	if (_json["ttl"].isInt())
		ttl = _json["ttl"].asInt();
	if (_json["workToProve"].isInt())
		workToProve = _json["workToProve"].asInt();
	if (!_json["topic"].empty())
	{
		if (_json["topic"].isString())
			bt.shift(jsToBytes(_json["topic"].asString()));
		else if (_json["topic"].isArray())
			for (auto i: _json["topic"])
				if (i.isString())
					bt.shift(jsToBytes(i.asString()));
	}
	return _m.seal(_from, bt, ttl, workToProve);
}

static pair<shh::FullTopic, Public> toWatch(Json::Value const& _json)
{
	shh::BuildTopic bt;
	Public to;

	if (_json["to"].isString())
		to = jsToPublic(_json["to"].asString());

	if (!_json["topic"].empty())
	{
		if (_json["topic"].isString())
			bt.shift(jsToBytes(_json["topic"].asString()));
		else if (_json["topic"].isArray())
			for (auto i: _json["topic"])
				if (i.isString())
					bt.shift(jsToBytes(i.asString()));
	}
	return make_pair(bt, to);
}

static Json::Value toJson(h256 const& _h, shh::Envelope const& _e, shh::Message const& _m)
{
	Json::Value res;
	res["hash"] = toJS(_h);
	res["expiry"] = (int)_e.expiry();
	res["sent"] = (int)_e.sent();
	res["ttl"] = (int)_e.ttl();
	res["workProved"] = (int)_e.workProved();
	for (auto const& t: _e.topic())
		res["topic"].append(toJS(t));
	res["payload"] = toJS(_m.payload());
	res["from"] = toJS(_m.from());
	res["to"] = toJS(_m.to());
	return res;
}

WebThreeStubServerBase::WebThreeStubServerBase(jsonrpc::AbstractServerConnector& _conn, std::vector<dev::KeyPair> const& _accounts):
	AbstractWebThreeStubServer(_conn)
{
	setAccounts(_accounts);
}

void WebThreeStubServerBase::setAccounts(std::vector<dev::KeyPair> const& _accounts)
{
	m_accounts.clear();
	for (auto const& i: _accounts)
	{
		m_accounts.push_back(i.address());
		m_accountsLookup[i.address()] = i;
	}
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

Json::Value WebThreeStubServerBase::eth_accounts()
{
	Json::Value ret(Json::arrayValue);
	for (auto const& i: m_accounts)
		ret.append(toJS(i));
	return ret;
}

std::string WebThreeStubServerBase::shh_addToGroup(std::string const& _group, std::string const& _who)
{
	(void)_group;
	(void)_who;
	return "";
}

std::string WebThreeStubServerBase::eth_balanceAt(string const& _address)
{
	return toJS(client()->balanceAt(jsToAddress(_address), client()->getDefault()));
}

Json::Value WebThreeStubServerBase::eth_blockByHash(std::string const& _hash)
{
	return toJson(client()->blockInfo(jsToFixed<32>(_hash)));
}

Json::Value WebThreeStubServerBase::eth_blockByNumber(int const& _number)
{
	return toJson(client()->blockInfo(client()->hashFromNumber(_number)));
}

static TransactionSkeleton toTransaction(Json::Value const& _json)
{
	TransactionSkeleton ret;
	if (!_json.isObject() || _json.empty())
		return ret;

	if (_json["from"].isString())
		ret.from = jsToAddress(_json["from"].asString());
	if (_json["to"].isString())
		ret.to = jsToAddress(_json["to"].asString());
	if (!_json["value"].empty())
	{
		if (_json["value"].isString())
			ret.value = jsToU256(_json["value"].asString());
		else if (_json["value"].isInt())
			ret.value = u256(_json["value"].asInt());
	}
	if (!_json["gas"].empty())
	{
		if (_json["gas"].isString())
			ret.gas = jsToU256(_json["gas"].asString());
		else if (_json["gas"].isInt())
			ret.gas = u256(_json["gas"].asInt());
	}
	if (!_json["gasPrice"].empty())
	{
		if (_json["gasPrice"].isString())
			ret.gasPrice = jsToU256(_json["gasPrice"].asString());
		else if (_json["gasPrice"].isInt())
			ret.gas = u256(_json["gas"].asInt());
	}
	if (!_json["data"].empty())
	{
		if (_json["data"].isString())							// ethereum.js has preconstructed the data array
			ret.data = jsToBytes(_json["data"].asString());
		else if (_json["data"].isArray())						// old style: array of 32-byte-padded values. TODO: remove PoC-8
			for (auto i: _json["data"])
				dev::operator +=(ret.data, padded(jsToBytes(i.asString()), 32));
	}

	if (_json["code"].isString())
		ret.data = jsToBytes(_json["code"].asString());
	return ret;
}

bool WebThreeStubServerBase::eth_flush()
{
	client()->flushTransactions();
	return true;
}

std::string WebThreeStubServerBase::eth_call(Json::Value const& _json)
{
	std::string ret;
	TransactionSkeleton t = toTransaction(_json);
	if (!t.from && m_accounts.size())
	{
		auto b = m_accounts.front();
		for (auto const& a: m_accounts)
			if (client()->balanceAt(a) > client()->balanceAt(b))
				b = a;
		t.from = b;
	}
	if (!m_accountsLookup.count(t.from))
		return ret;
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);
	ret = toJS(client()->call(m_accountsLookup[t.from].secret(), t.value, t.to, t.data, t.gas, t.gasPrice));
	return ret;
}

Json::Value WebThreeStubServerBase::eth_changed(int const& _id)
{
	auto entries = client()->checkWatch(_id);
	if (entries.size())
		cnote << "FIRING WATCH" << _id << entries.size();
	return toJson(entries);
}

std::string WebThreeStubServerBase::eth_codeAt(string const& _address)
{
	return jsFromBinary(client()->codeAt(jsToAddress(_address), client()->getDefault()));
}

std::string WebThreeStubServerBase::eth_coinbase()
{
	return toJS(client()->address());
}

double WebThreeStubServerBase::eth_countAt(string const& _address)
{
	return (double)(uint64_t)client()->countAt(jsToAddress(_address), client()->getDefault());
}

int WebThreeStubServerBase::eth_defaultBlock()
{
	return client()->getDefault();
}

std::string WebThreeStubServerBase::eth_gasPrice()
{
	return toJS(10 * dev::eth::szabo);
}

std::string WebThreeStubServerBase::db_get(std::string const& _name, std::string const& _key)
{
	string ret = db()->get(_name, _key);
	return toJS(dev::asBytes(ret));
}

Json::Value WebThreeStubServerBase::eth_filterLogs(int const& _id)
{
	return toJson(client()->logs(_id));
}

Json::Value WebThreeStubServerBase::eth_logs(Json::Value const& _json)
{
	return toJson(client()->logs(toLogFilter(_json)));
}

std::string WebThreeStubServerBase::db_getString(std::string const& _name, std::string const& _key)
{
	return db()->get(_name, _key);;
}

bool WebThreeStubServerBase::shh_haveIdentity(std::string const& _id)
{
	return m_ids.count(jsToPublic(_id)) > 0;
}

bool WebThreeStubServerBase::eth_listening()
{
	return network()->isNetworkStarted();
}

bool WebThreeStubServerBase::eth_mining()
{
	return client()->isMining();
}

int WebThreeStubServerBase::eth_newFilter(Json::Value const& _json)
{
	unsigned ret = -1;
	ret = client()->installWatch(toLogFilter(_json));
	return ret;
}

int WebThreeStubServerBase::eth_newFilterString(std::string const& _filter)
{
	unsigned ret = -1;
	if (_filter.compare("chain") == 0)
		ret = client()->installWatch(dev::eth::ChainChangedFilter);
	else if (_filter.compare("pending") == 0)
		ret = client()->installWatch(dev::eth::PendingChangedFilter);
	return ret;
}

Json::Value WebThreeStubServerBase::eth_getWork()
{
	Json::Value ret(Json::arrayValue);
	auto r = client()->getWork();
	ret.append(toJS(r.first));
	ret.append(toJS(r.second));
	return ret;
}

bool WebThreeStubServerBase::eth_submitWork(std::string const& _nonce)
{
	return client()->submitNonce(jsToFixed<32>(_nonce));
}

std::string WebThreeStubServerBase::shh_newGroup(std::string const& _id, std::string const& _who)
{
	(void)_id;
	(void)_who;
	return "";
}

std::string WebThreeStubServerBase::shh_newIdentity()
{
//	cnote << this << m_ids;
	KeyPair kp = KeyPair::create();
	m_ids[kp.pub()] = kp.secret();
	return toJS(kp.pub());
}

Json::Value WebThreeStubServerBase::eth_compilers()
{
	Json::Value ret(Json::arrayValue);
	ret.append("lll");
	ret.append("solidity");
#ifndef _MSC_VER
	ret.append("serpent");
#endif
	return ret;
}

std::string WebThreeStubServerBase::eth_lll(std::string const& _code)
{
	string res;
	vector<string> errors;
	res = toJS(dev::eth::compileLLL(_code, true, &errors));
	cwarn << "LLL compilation errors: " << errors;
	return res;
}

std::string WebThreeStubServerBase::eth_serpent(std::string const& _code)
{
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

std::string WebThreeStubServerBase::eth_solidity(std::string const& _code)
{
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

int WebThreeStubServerBase::eth_number()
{
	return client()->number() + 1;
}

int WebThreeStubServerBase::eth_peerCount()
{
	return network()->peerCount();
}

bool WebThreeStubServerBase::shh_post(Json::Value const& _json)
{
	shh::Message m = toMessage(_json);
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

bool WebThreeStubServerBase::db_put(std::string const& _name, std::string const& _key, std::string const& _value)
{
	string v = asString(jsToBytes(_value));
	db()->put(_name, _key, v);
	return true;
}

bool WebThreeStubServerBase::db_putString(std::string const& _name, std::string const& _key, std::string const& _value)
{
	db()->put(_name, _key,_value);
	return true;
}

bool WebThreeStubServerBase::eth_setCoinbase(std::string const& _address)
{
	client()->setAddress(jsToAddress(_address));
	return true;
}

bool WebThreeStubServerBase::eth_setDefaultBlock(int const& _block)
{
	client()->setDefault(_block);
	return true;
}

bool WebThreeStubServerBase::eth_setListening(bool const& _listening)
{
	if (_listening)
		network()->startNetwork();
	else
		network()->stopNetwork();
	return true;
}

bool WebThreeStubServerBase::eth_setMining(bool const& _mining)
{
	if (_mining)
		client()->startMining();
	else
		client()->stopMining();
	return true;
}

Json::Value WebThreeStubServerBase::shh_changed(int const& _id)
{
	Json::Value ret(Json::arrayValue);
	auto pub = m_shhWatches[_id];
	if (!pub || m_ids.count(pub))
		for (h256 const& h: face()->checkWatch(_id))
		{
			auto e = face()->envelope(h);
			shh::Message m;
			if (pub)
			{
				cwarn << "Silently decrypting message from identity" << pub.abridged() << ": User validation hook goes here.";
				m = e.open(face()->fullTopic(_id), m_ids[pub]);
			}
			else
				m = e.open(face()->fullTopic(_id));
			if (!m)
				continue;
			ret.append(toJson(h, e, m));
		}
	
	return ret;
}

int WebThreeStubServerBase::shh_newFilter(Json::Value const& _json)
{
	auto w = toWatch(_json);
	auto ret = face()->installWatch(w.first);
	m_shhWatches.insert(make_pair(ret, w.second));
	return ret;
}

bool WebThreeStubServerBase::shh_uninstallFilter(int const& _id)
{
	face()->uninstallWatch(_id);
	return true;
}

std::string WebThreeStubServerBase::eth_stateAt(string const& _address, string const& _storage)
{
	return toJS(client()->stateAt(jsToAddress(_address), jsToU256(_storage), client()->getDefault()));
}

Json::Value WebThreeStubServerBase::eth_storageAt(string const& _address)
{
	return toJson(client()->storageAt(jsToAddress(_address)));
}

std::string WebThreeStubServerBase::eth_transact(Json::Value const& _json)
{
	std::string ret;
	TransactionSkeleton t = toTransaction(_json);
	if (!t.from && m_accounts.size())
	{
		auto b = m_accounts.front();
		for (auto const& a: m_accounts)
			if (client()->balanceAt(a) > client()->balanceAt(b))
				b = a;
		t.from = b;
	}
	if (!m_accountsLookup.count(t.from))
		return ret;
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);
	if (authenticate(t))
	{
		if (t.to)
			// TODO: from qethereum, insert validification hook here.
			client()->transact(m_accountsLookup[t.from].secret(), t.value, t.to, t.data, t.gas, t.gasPrice);
		else
			ret = toJS(client()->transact(m_accountsLookup[t.from].secret(), t.value, t.data, t.gas, t.gasPrice));
		client()->flushTransactions();
	}
	return ret;
}

bool WebThreeStubServerBase::authenticate(TransactionSkeleton const& _t)
{
	cwarn << "Silently signing transaction from address" << _t.from.abridged() << ": User validation hook goes here.";
	return true;
}

Json::Value WebThreeStubServerBase::eth_transactionByHash(std::string const& _hash, int const& _i)
{
	return toJson(client()->transaction(jsToFixed<32>(_hash), _i));
}

Json::Value WebThreeStubServerBase::eth_transactionByNumber(int const& _number, int const& _i)
{
	return toJson(client()->transaction(client()->hashFromNumber(_number), _i));
}

Json::Value WebThreeStubServerBase::eth_uncleByHash(std::string const& _hash, int const& _i)
{
	return toJson(client()->uncle(jsToFixed<32>(_hash), _i));
}

Json::Value WebThreeStubServerBase::eth_uncleByNumber(int const& _number, int const& _i)
{
	return toJson(client()->uncle(client()->hashFromNumber(_number), _i));
}

bool WebThreeStubServerBase::eth_uninstallFilter(int const& _id)
{
	client()->uninstallWatch(_id);
	return true;
}

