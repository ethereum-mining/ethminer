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

#include "WebThreeStubServerBase.h"

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
#include "AccountHolder.h"
#include "JsonHelper.h"
using namespace std;
using namespace jsonrpc;
using namespace dev;
using namespace eth;
using namespace shh;

#if ETH_DEBUG
const unsigned dev::SensibleHttpThreads = 1;
#else
const unsigned dev::SensibleHttpThreads = 4;
#endif
const unsigned dev::SensibleHttpPort = 8545;

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
		return toJS(toCompactBigEndian(client()->stateAt(jsToAddress(_address), jsToU256(_position), jsToBlockNumber(_blockNumber)), 1));
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

Json::Value WebThreeStubServerBase::eth_getBlockTransactionCountByHash(string const& _blockHash)
{
	try
	{
		h256 blockHash = jsToFixed<32>(_blockHash);
		if (!client()->isKnown(blockHash))
			return Json::Value(Json::nullValue);

		return toJS(client()->transactionCount(blockHash));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getBlockTransactionCountByNumber(string const& _blockNumber)
{
	try
	{
		BlockNumber blockNumber = jsToBlockNumber(_blockNumber);
		if (!client()->isKnown(blockNumber))
			return Json::Value(Json::nullValue);

		return toJS(client()->transactionCount(jsToBlockNumber(_blockNumber)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getUncleCountByBlockHash(string const& _blockHash)
{
	try
	{
		h256 blockHash = jsToFixed<32>(_blockHash);
		if (!client()->isKnown(blockHash))
			return Json::Value(Json::nullValue);

		return toJS(client()->uncleCount(blockHash));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getUncleCountByBlockNumber(string const& _blockNumber)
{
	try
	{
		BlockNumber blockNumber = jsToBlockNumber(_blockNumber);
		if (!client()->isKnown(blockNumber))
			return Json::Value(Json::nullValue);

		return toJS(client()->uncleCount(blockNumber));
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

string WebThreeStubServerBase::eth_sendTransaction(Json::Value const& _json)
{
	try
	{
		TransactionSkeleton t = toTransactionSkeleton(_json);
	
		if (!t.from)
			t.from = m_ethAccounts->defaultTransactAccount();
		if (t.gasPrice == UndefinedU256)
			t.gasPrice = 10 * dev::eth::szabo;		// TODO: should be determined by user somehow.
		if (t.gas == UndefinedU256)
			t.gas = min<u256>(client()->gasLimitRemaining() / 5, client()->balanceAt(t.from) / t.gasPrice);

		return toJS(m_ethAccounts->authenticate(t));
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
		TransactionSkeleton t = toTransactionSkeleton(_json);

		if (!t.from)
			t.from = m_ethAccounts->defaultTransactAccount();
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
		TransactionSkeleton t = toTransactionSkeleton(_json);
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
		h256 h = jsToFixed<32>(_blockHash);
		if (!client()->isKnown(h))
			return Json::Value(Json::nullValue);

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
		BlockNumber h = jsToBlockNumber(_blockNumber);
		if (!client()->isKnown(h))
			return Json::Value(Json::nullValue);

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
		if (!client()->isKnownTransaction(h))
			return Json::Value(Json::nullValue);

		return toJson(client()->localisedTransaction(h));
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
		if (!client()->isKnownTransaction(bh, ti))
			return Json::Value(Json::nullValue);

		return toJson(client()->localisedTransaction(bh, ti));
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
		h256 bh = client()->hashFromNumber(bn);
		unsigned ti = jsToInt(_transactionIndex);
		if (!client()->isKnownTransaction(bh, ti))
			return Json::Value(Json::nullValue);

		return toJson(client()->localisedTransaction(bh, ti));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getTransactionReceipt(string const& _transactionHash)
{
	try
	{
		h256 h = jsToFixed<32>(_transactionHash);
		if (!client()->isKnownTransaction(h))
			return Json::Value(Json::nullValue);

		return toJson(client()->localisedTransactionReceipt(h));
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


string WebThreeStubServerBase::eth_compileLLL(string const& _source)
{
	// TODO throw here jsonrpc errors
	string res;
	vector<string> errors;
	res = toJS(dev::eth::compileLLL(_source, true, &errors));
	cwarn << "LLL compilation errors: " << errors;
	return res;
}

string WebThreeStubServerBase::eth_compileSerpent(string const& _source)
{
	// TODO throw here jsonrpc errors
	string res;
#if ETH_SERPENT || !ETH_TRUE
	try
	{
		res = toJS(dev::asBytes(::compile(_source)));
	}
	catch (string err)
	{
		cwarn << "Serpent compilation error: " << err;
	}
	catch (...)
	{
		cwarn << "Uncought serpent compilation exception";
	}
#else
	(void)_source;
#endif
	return res;
}

#define ADMIN requires(_session, Privilege::Admin)

bool WebThreeStubServerBase::admin_web3_setVerbosity(int _v, string const& _session)
{
	ADMIN;
	g_logVerbosity = _v;
	return true;
}

bool WebThreeStubServerBase::admin_net_start(std::string const& _session)
{
	ADMIN;
	network()->startNetwork();
	return true;
}

bool WebThreeStubServerBase::admin_net_stop(std::string const& _session)
{
	ADMIN;
	network()->stopNetwork();
	return true;
}

bool WebThreeStubServerBase::admin_net_connect(std::string const& _node, std::string const& _session)
{
	ADMIN;
	p2p::NodeId id;
	bi::tcp::endpoint ep;
	if (_node.substr(0, 8) == "enode://" && _node.find('@') == 136)
	{
		id = p2p::NodeId(_node.substr(8, 128));
		ep = p2p::Network::resolveHost(_node.substr(137));
	}
	else
		ep = p2p::Network::resolveHost(_node);

	if (ep == bi::tcp::endpoint())
		return false;
	network()->requirePeer(id, ep);
	return true;
}

Json::Value WebThreeStubServerBase::admin_net_peers(std::string const& _session)
{
	ADMIN;
	Json::Value ret;
	for (p2p::PeerSessionInfo const& i: network()->peers())
		ret.append(toJson(i));
	return ret;
}

Json::Value WebThreeStubServerBase::admin_net_nodeInfo(const string& _session)
{
	ADMIN;
	Json::Value ret;
	p2p::NodeInfo i = network()->nodeInfo();
	ret["name"] = i.version;
	ret["port"] = i.port;
	ret["address"] = i.address;
	ret["listenAddr"] = i.address + ":" + toString(i.port);
	ret["id"] = i.id.hex();
	ret["enode"] = i.enode();
	return ret;
}

bool WebThreeStubServerBase::admin_eth_setMining(bool _on, std::string const& _session)
{
	ADMIN;
	if (_on)
		client()->startMining();
	else
		client()->stopMining();
	return true;
}

Json::Value WebThreeStubServerBase::eth_compileSolidity(string const& _source)
{
	// TOOD throw here jsonrpc errors
	Json::Value res(Json::objectValue);
#if ETH_SOLIDITY || !ETH_TRUE
	dev::solidity::CompilerStack compiler;
	try
	{
		compiler.addSource("source", _source);
		compiler.compile();

		for (string const& name: compiler.getContractNames())
		{
			Json::Value contract(Json::objectValue);
			contract["code"] = toJS(compiler.getBytecode(name));

			Json::Value info(Json::objectValue);
			info["source"] = _source;
			info["language"] = "";
			info["languageVersion"] = "";
			info["compilerVersion"] = "";

			Json::Reader reader;
			reader.parse(compiler.getInterface(name), info["abiDefinition"]);
			reader.parse(compiler.getMetadata(name, dev::solidity::DocumentationType::NatspecUser), info["userDoc"]);
			reader.parse(compiler.getMetadata(name, dev::solidity::DocumentationType::NatspecDev), info["developerDoc"]);

			contract["info"] = info;
			res[name] = contract;
		}
	}
	catch (dev::Exception const& exception)
	{
		ostringstream error;
		solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
		cwarn << "Solidity compilation error: " << error.str();
		return Json::Value(Json::objectValue);
	}
	catch (...)
	{
		cwarn << "Uncought solidity compilation exception";
		return Json::Value(Json::objectValue);
	}
#else
	(void)_source;
#endif
	return res;
}

string WebThreeStubServerBase::eth_newFilter(Json::Value const& _json)
{
	try
	{
		return toJS(client()->installWatch(toLogFilter(_json, *client())));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

string WebThreeStubServerBase::eth_newFilterEx(Json::Value const& _json)
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

string WebThreeStubServerBase::eth_newBlockFilter()
{
	h256 filter = dev::eth::ChainChangedFilter;
	return toJS(client()->installWatch(filter));
}

string WebThreeStubServerBase::eth_newPendingTransactionFilter()
{
	h256 filter = dev::eth::PendingChangedFilter;
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
//		if (entries.size())
//			cnote << "FIRING WATCH" << id << entries.size();
		return toJson(entries);
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getFilterChangesEx(string const& _filterId)
{
	try
	{
		int id = jsToInt(_filterId);
		auto entries = client()->checkWatch(id);
//		if (entries.size())
//			cnote << "FIRING WATCH" << id << entries.size();
		return toJsonByBlock(entries);
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

Json::Value WebThreeStubServerBase::eth_getFilterLogsEx(string const& _filterId)
{
	try
	{
		return toJsonByBlock(client()->logs(jsToInt(_filterId)));
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
		return toJson(client()->logs(toLogFilter(_json, *client())));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getLogsEx(Json::Value const& _json)
{
	try
	{
		return toJsonByBlock(client()->logs(toLogFilter(_json)));
	}
	catch (...)
	{
		BOOST_THROW_EXCEPTION(JsonRpcException(Errors::ERROR_RPC_INVALID_PARAMS));
	}
}

Json::Value WebThreeStubServerBase::eth_getWork()
{
	Json::Value ret(Json::arrayValue);
	auto r = client()->getEthashWork();
	ret.append(toJS(get<0>(r)));
	ret.append(toJS(get<1>(r)));
	ret.append(toJS(get<2>(r)));
	return ret;
}

bool WebThreeStubServerBase::eth_submitWork(string const& _nonce, string const&, string const& _mixHash)
{
	try
	{
		return client()->submitEthashWork(jsToFixed<32>(_mixHash), jsToFixed<Nonce::size>(_nonce));
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
