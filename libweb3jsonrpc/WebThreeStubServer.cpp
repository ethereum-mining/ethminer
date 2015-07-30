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

#include "WebThreeStubServer.h"
// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <libdevcore/FileSystem.h>
#include <libdevcore/CommonJS.h>
#include <libethcore/KeyManager.h>
#include <libethereum/Executive.h>
#include <libwebthree/WebThree.h>
#include "JsonHelper.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
namespace fs = boost::filesystem;

bool isHex(std::string const& _s)
{
	unsigned i = (_s.size() >= 2 && _s.substr(0, 2) == "0x") ? 2 : 0;
	for (; i < _s.size(); ++i)
		if (fromHex(_s[i], WhenError::DontThrow) == -1)
			return false;
	return true;
}

template <class T> bool isHash(std::string const& _hash)
{
	return (_hash.size() == T::size * 2 || (_hash.size() == T::size * 2 + 2 && _hash.substr(0, 2) == "0x")) && isHex(_hash);
}

WebThreeStubServer::WebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, WebThreeDirect& _web3, shared_ptr<AccountHolder> const& _ethAccounts, std::vector<dev::KeyPair> const& _shhAccounts, KeyManager& _keyMan, dev::eth::TrivialGasPricer& _gp):
	WebThreeStubServerBase(_conn, _ethAccounts, _shhAccounts),
	m_web3(_web3),
	m_keyMan(_keyMan),
	m_gp(_gp)
{
	auto path = getDataDir() + "/.web3";
	fs::create_directories(path);
	fs::permissions(path, fs::owner_all);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path, &m_db);
}

std::string WebThreeStubServer::newSession(SessionPermissions const& _p)
{
	std::string s = toBase64(h64::random().ref());
	m_sessions[s] = _p;
	return s;
}

bool WebThreeStubServer::eth_notePassword(string const& _password)
{
	m_keyMan.notePassword(_password);
	return true;
}

#define ADMIN_GUARD requires(_session, Privilege::Admin)

Json::Value WebThreeStubServer::admin_eth_blockQueueStatus(string const& _session)
{
	ADMIN_GUARD;
	Json::Value ret;
	BlockQueueStatus bqs = m_web3.ethereum()->blockQueue().status();
	ret["importing"] = (int)bqs.importing;
	ret["verified"] = (int)bqs.verified;
	ret["verifying"] = (int)bqs.verifying;
	ret["unverified"] = (int)bqs.unverified;
	ret["future"] = (int)bqs.future;
	ret["unknown"] = (int)bqs.unknown;
	ret["bad"] = (int)bqs.bad;
	return ret;
}

bool WebThreeStubServer::admin_eth_setAskPrice(std::string const& _wei, std::string const& _session)
{
	ADMIN_GUARD;
	m_gp.setAsk(jsToU256(_wei));
	return true;
}

bool WebThreeStubServer::admin_eth_setBidPrice(std::string const& _wei, std::string const& _session)
{
	ADMIN_GUARD;
	m_gp.setBid(jsToU256(_wei));
	return true;
}

dev::eth::BlockChain const& WebThreeStubServer::bc() const
{
	return m_web3.ethereum()->blockChain();
}

dev::eth::BlockQueue const& WebThreeStubServer::bq() const
{
	return m_web3.ethereum()->blockQueue();
}

Json::Value WebThreeStubServer::admin_eth_findBlock(std::string const& _blockHash, std::string const& _session)
{
	ADMIN_GUARD;
	h256 h(_blockHash);
	if (bc().isKnown(h))
		return toJson(bc().info(h));
	switch(bq().blockStatus(h))
	{
	case QueueStatus::Ready:
		return "ready";
	case QueueStatus::Importing:
		return "importing";
	case QueueStatus::UnknownParent:
		return "unknown parent";
	case QueueStatus::Bad:
		return "bad";
	default:
		return "unknown";
	}
}

std::string WebThreeStubServer::admin_eth_blockQueueFirstUnknown(std::string const& _session)
{
	ADMIN_GUARD;
	return bq().firstUnknown().hex();
}

bool WebThreeStubServer::admin_eth_blockQueueRetryUnknown(std::string const& _session)
{
	ADMIN_GUARD;
	m_web3.ethereum()->retryUnknown();
	return true;
}

Json::Value WebThreeStubServer::admin_eth_allAccounts(std::string const& _session)
{
	ADMIN_GUARD;
	Json::Value ret;
	u256 total = 0;
	u256 pendingtotal = 0;
	Address beneficiary;
	for (auto const& address: m_keyMan.accounts())
	{
		auto pending = m_web3.ethereum()->balanceAt(address, PendingBlock);
		auto latest = m_web3.ethereum()->balanceAt(address, LatestBlock);
		Json::Value a;
		if (address == beneficiary)
			a["beneficiary"] = true;
		a["address"] = toJS(address);
		a["balance"] = toJS(latest);
		a["nicebalance"] = formatBalance(latest);
		a["pending"] = toJS(pending);
		a["nicepending"] = formatBalance(pending);
		ret["accounts"][m_keyMan.accountName(address)] = a;
		total += latest;
		pendingtotal += pending;
	}
	ret["total"] = toJS(total);
	ret["nicetotal"] = formatBalance(total);
	ret["pendingtotal"] = toJS(pendingtotal);
	ret["nicependingtotal"] = formatBalance(pendingtotal);
	return ret;
}

Json::Value WebThreeStubServer::admin_eth_newAccount(Json::Value const& _info, std::string const& _session)
{
	ADMIN_GUARD;
	if (!_info.isMember("name"))
		throw jsonrpc::JsonRpcException("No member found: name");
	string name = _info["name"].asString();
	auto s = Secret::random();
	h128 uuid;
	if (_info.isMember("password"))
	{
		string password = _info["password"].asString();
		string hint = _info["passwordHint"].asString();
		uuid = m_keyMan.import(s, name, password, hint);
	}
	else
		uuid = m_keyMan.import(s, name);
	Json::Value ret;
	ret["account"] = toJS(toAddress(s));
	ret["uuid"] = toUUID(uuid);
	return ret;
}

bool WebThreeStubServer::admin_eth_setMiningBenefactor(std::string const& _uuidOrAddress, std::string const& _session)
{
	ADMIN_GUARD;
	Address a;
	h128 uuid = fromUUID(_uuidOrAddress);
	if (uuid)
		a = m_keyMan.address(uuid);
	else if (isHash<Address>(_uuidOrAddress))
		a = Address(_uuidOrAddress);
	else
		throw jsonrpc::JsonRpcException("Invalid UUID or address");
	if (m_setMiningBenefactor)
		m_setMiningBenefactor(a);
	else
		m_web3.ethereum()->setAddress(a);
	return true;
}

Json::Value WebThreeStubServer::admin_eth_inspect(std::string const& _address, std::string const& _session)
{
	ADMIN_GUARD;
	if (!isHash<Address>(_address))
		throw jsonrpc::JsonRpcException("Invalid address given.");

	Json::Value ret;
	auto h = Address(fromHex(_address));
	ret["storage"] = toJson(m_web3.ethereum()->storageAt(h, PendingBlock));
	ret["balance"] = toJS(m_web3.ethereum()->balanceAt(h, PendingBlock));
	ret["nonce"] = toJS(m_web3.ethereum()->countAt(h, PendingBlock));
	ret["code"] = toJS(m_web3.ethereum()->codeAt(h, PendingBlock));
	return ret;
}

h256 WebThreeStubServer::blockHash(std::string const& _blockNumberOrHash) const
{
	if (isHash<h256>(_blockNumberOrHash))
		return h256(_blockNumberOrHash.substr(_blockNumberOrHash.size() - 64, 64));
	try
	{
		return bc().numberHash(stoul(_blockNumberOrHash));
	}
	catch (...)
	{
		throw jsonrpc::JsonRpcException("Invalid argument");
	}
}

Json::Value WebThreeStubServer::admin_eth_reprocess(std::string const& _blockNumberOrHash, std::string const& _session)
{
	ADMIN_GUARD;
	Json::Value ret;
	PopulationStatistics ps;
	m_web3.ethereum()->state(blockHash(_blockNumberOrHash), &ps);
	ret["enact"] = ps.enact;
	ret["verify"] = ps.verify;
	ret["total"] = ps.verify + ps.enact;
	return ret;
}

Json::Value WebThreeStubServer::admin_eth_vmTrace(std::string const& _blockNumberOrHash, int _txIndex, std::string const& _session)
{
	ADMIN_GUARD;

	Json::Value ret;

	auto c = m_web3.ethereum();
	State state = c->state(_txIndex + 1, blockHash(_blockNumberOrHash));

	if (_txIndex < 0)
		throw jsonrpc::JsonRpcException("Negative index");

	if ((unsigned)_txIndex < state.pending().size())
	{
		Executive e(state, bc(), 0);
		Transaction t = state.pending()[_txIndex];
		state = state.fromPending(_txIndex);
		try
		{
			StandardTrace st;
			st.setShowMnemonics();
			e.initialize(t);
			if (!e.execute())
				e.go(st.onOp());
			e.finalize();
			Json::Reader().parse(st.json(), ret);
		}
		catch(Exception const& _e)
		{
			cwarn << diagnostic_information(_e);
		}
	}

	return ret;
}

Json::Value WebThreeStubServer::admin_eth_getReceiptByHashAndIndex(std::string const& _blockNumberOrHash, int _txIndex, std::string const& _session)
{
	ADMIN_GUARD;
	if (_txIndex < 0)
		throw jsonrpc::JsonRpcException("Negative index");
	auto h = blockHash(_blockNumberOrHash);
	if (!bc().isKnown(h))
		throw jsonrpc::JsonRpcException("Invalid/unknown block.");
	auto rs = bc().receipts(h);
	if ((unsigned)_txIndex >= rs.receipts.size())
		throw jsonrpc::JsonRpcException("Index too large.");
	return toJson(rs.receipts[_txIndex]);
}

std::string WebThreeStubServer::web3_clientVersion()
{
	return m_web3.clientVersion();
}

dev::eth::Interface* WebThreeStubServer::client()
{
	return m_web3.ethereum();
}

std::shared_ptr<dev::shh::Interface> WebThreeStubServer::face()
{
	return m_web3.whisper();
}

dev::WebThreeNetworkFace* WebThreeStubServer::network()
{
	return &m_web3;
}

dev::WebThreeStubDatabaseFace* WebThreeStubServer::db()
{
	return this;
}

std::string WebThreeStubServer::get(std::string const& _name, std::string const& _key)
{
	bytes k = sha3(_name).asBytes() + sha3(_key).asBytes();
	string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return ret;
}

void WebThreeStubServer::put(std::string const& _name, std::string const& _key, std::string const& _value)
{
	bytes k = sha3(_name).asBytes() + sha3(_key).asBytes();
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)_value.data(), _value.size()));
}

