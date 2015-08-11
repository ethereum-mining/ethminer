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
/** @file shhrpc.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date July 2015
*/

#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <libdevcore/Log.h>
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonJS.h>
#include <libwebthree/WebThree.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <test/TestHelper.h>
#include <test/libweb3jsonrpc/webthreestubclient.h>
#include <libethcore/KeyManager.h>
#include <libp2p/Common.h>
#include <libwhisper/WhisperHost.h>

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;
namespace js = json_spirit;

WebThreeDirect* web3;
unique_ptr<WebThreeStubServer> jsonrpcServer;
unique_ptr<WebThreeStubClient> jsonrpcClient;
static uint16_t const c_web3port = 30333;
static string const c_version("shhrpc-web3");
static unsigned const c_ttl = 777000;

struct Setup
{
	Setup()
	{
		dev::p2p::NodeIPEndpoint::test_allowLocal = true;

		static bool setup = false;
		if (!setup && !test::Options::get().nonetwork)
		{
			setup = true;
			NetworkPreferences nprefs(std::string(), c_web3port, false);
			web3 = new WebThreeDirect(c_version, "", WithExisting::Trust, {"shh"}, nprefs);
			web3->setIdealPeerCount(1);
			auto server = new jsonrpc::HttpServer(8080);
			vector<KeyPair> v;
			KeyManager keyMan;
			TrivialGasPricer gp;
			jsonrpcServer = unique_ptr<WebThreeStubServer>(new WebThreeStubServer(*server, *web3, nullptr, v, keyMan, gp));
			jsonrpcServer->setIdentities({});
			jsonrpcServer->StartListening();
			auto client = new jsonrpc::HttpClient("http://localhost:8080");
			jsonrpcClient = unique_ptr<WebThreeStubClient>(new WebThreeStubClient(*client));
		}
	}

	~Setup()
	{
		dev::p2p::NodeIPEndpoint::test_allowLocal = false;
	}
};

Json::Value createMessage(string const& _from, string const& _to, string const& _topic = "", string _payload = "")
{
	Json::Value msg;
	msg["from"] = _from;
	msg["to"] = _to;
	msg["ttl"] = toJS(c_ttl);

	if (_payload.empty())
		_payload = string("0x") + h256::random().hex();

	msg["payload"] = _payload;
	
	if (!_topic.empty())
	{
		Json::Value t(Json::arrayValue);
		t.append(_topic);
		msg["topics"] = t;
	}

	return msg;
}

BOOST_FIXTURE_TEST_SUITE(shhrpc, Setup)

BOOST_AUTO_TEST_CASE(basic)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing web3 basic functionality...";

	web3->startNetwork();
	unsigned const step = 10;
	for (unsigned i = 0; i < 3000 && !web3->haveNetwork(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(web3->haveNetwork());

	uint16_t const port2 = 30334;
	NetworkPreferences prefs2("127.0.0.1", port2, false);
	string const version2 = "shhrpc-host2";
	Host host2(version2, prefs2);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();

	for (unsigned i = 0; i < 3000 && !host2.haveNetwork(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host2.haveNetwork());

	web3->addNode(host2.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port2, port2));

	for (unsigned i = 0; i < 3000 && (!web3->peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE_EQUAL(host2.peerCount(), 1);
	BOOST_REQUIRE_EQUAL(web3->peerCount(), 1);

	vector<PeerSessionInfo> vpeers = web3->peers();
	BOOST_REQUIRE(!vpeers.empty());
	PeerSessionInfo const& peer = vpeers.back();
	BOOST_REQUIRE_EQUAL(peer.id, host2.id());
	BOOST_REQUIRE_EQUAL(peer.port, port2);
	BOOST_REQUIRE_EQUAL(peer.clientVersion, version2);

	web3->stopNetwork();

	for (unsigned i = 0; i < 3000 && (web3->haveNetwork() || host2.haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(!web3->peerCount());
	BOOST_REQUIRE(!host2.peerCount());
}

BOOST_AUTO_TEST_CASE(send)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing web3 send...";

	bool sent = false;
	bool ready = false;
	unsigned result = 0;
	unsigned const messageCount = 10;
	unsigned const step = 10;
	uint16_t port2 = 30337;

	Host host2("shhrpc-host2", NetworkPreferences("127.0.0.1", port2, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();
	web3->startNetwork();

	std::thread listener([&]()
	{
		setThreadName("listener");
		ready = true;
		auto w = whost2->installWatch(BuildTopicMask("odd"));
		set<unsigned> received;
		for (unsigned x = 0; x < 7000 && !sent; x += step)
			this_thread::sleep_for(chrono::milliseconds(step));

		for (unsigned x = 0, last = 0; x < 100 && received.size() < messageCount; ++x)
		{
			this_thread::sleep_for(chrono::milliseconds(50));
			for (auto i: whost2->checkWatch(w))
			{
				Message msg = whost2->envelope(i).open(whost2->fullTopics(w));
				last = RLP(msg.payload()).toInt<unsigned>();
				if (received.insert(last).second)
					result += last;
			}
		}
	});

	for (unsigned i = 0; i < 2000 && (!host2.haveNetwork() || !web3->haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host2.haveNetwork());
	BOOST_REQUIRE(web3->haveNetwork());

	web3->requirePeer(host2.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port2, port2));

	for (unsigned i = 0; i < 3000 && (!web3->peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE_EQUAL(host2.peerCount(), 1);
	BOOST_REQUIRE_EQUAL(web3->peerCount(), 1);
	
	KeyPair us = KeyPair::create();
	for (unsigned i = 0; i < messageCount; ++i)
	{
		web3->whisper()->post(us.sec(), RLPStream().append(i * i).out(), BuildTopic(i)(i % 2 ? "odd" : "even"), 777000, 1);
		this_thread::sleep_for(chrono::milliseconds(50));
	}
	
	sent = true;
	auto messages = web3->whisper()->all();
	BOOST_REQUIRE_EQUAL(messages.size(), messageCount);

	listener.join();
	BOOST_REQUIRE_EQUAL(result, 1 + 9 + 25 + 49 + 81);
}

BOOST_AUTO_TEST_CASE(receive)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing web3 receive...";

	bool sent = false;
	bool ready = false;
	unsigned result = 0;
	unsigned const messageCount = 6;
	unsigned const step = 10;
	uint16_t port2 = 30338;
	Host host2("shhrpc-host2", NetworkPreferences("127.0.0.1", port2, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();
	web3->startNetwork();

	std::thread listener([&]()
	{
		setThreadName("listener");
		ready = true;
		auto w = web3->whisper()->installWatch(BuildTopicMask("odd"));
		
		set<unsigned> received;
		for (unsigned x = 0; x < 7000 && !sent; x += step)
			this_thread::sleep_for(chrono::milliseconds(step));

		for (unsigned x = 0, last = 0; x < 100 && received.size() < messageCount; ++x)
		{
			this_thread::sleep_for(chrono::milliseconds(50));
			for (auto i: web3->whisper()->checkWatch(w))
			{
				Message msg = web3->whisper()->envelope(i).open(web3->whisper()->fullTopics(w));
				last = RLP(msg.payload()).toInt<unsigned>();
				if (received.insert(last).second)
					result += last;
			}
		}

		web3->whisper()->uninstallWatch(w);
	});

	for (unsigned i = 0; i < 2000 && (!host2.haveNetwork() || !web3->haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host2.haveNetwork());
	BOOST_REQUIRE(web3->haveNetwork());

	host2.addNode(web3->id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), c_web3port, c_web3port));

	for (unsigned i = 0; i < 3000 && (!web3->peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE_EQUAL(host2.peerCount(), 1);
	BOOST_REQUIRE_EQUAL(web3->peerCount(), 1);
	
	KeyPair us = KeyPair::create();
	for (unsigned i = 0; i < messageCount; ++i)
	{
		web3->whisper()->post(us.sec(), RLPStream().append(i * i * i).out(), BuildTopic(i)(i % 2 ? "odd" : "even"), c_ttl, 1);
		this_thread::sleep_for(chrono::milliseconds(50));
	}
	
	sent = true;
	listener.join();
	BOOST_REQUIRE_EQUAL(result, 1 + 27 + 125);
}

BOOST_AUTO_TEST_CASE(serverBasic)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing basic jsonrpc server...";

	string s = jsonrpcServer->web3_clientVersion();
	BOOST_REQUIRE_EQUAL(s, c_version);

	s = jsonrpcServer->net_version();
	BOOST_REQUIRE(s.empty());

	s = jsonrpcServer->web3_sha3("some pseudo-random string here");
	BOOST_REQUIRE_EQUAL(s.size(), h256::size * 2 + 2);
	BOOST_REQUIRE('0' == s[0] && 'x' == s[1]);

	s = jsonrpcServer->net_peerCount();
	BOOST_REQUIRE_EQUAL(s, "0x0");

	KeyPair src = KeyPair::create();
	KeyPair dst = KeyPair::create();
	Json::Value t1 = createMessage(toJS(src.address()), toJS(dst.address()));
	bool b = jsonrpcServer->shh_post(t1);
	BOOST_REQUIRE(b);

	string const id = jsonrpcServer->shh_newIdentity();
	BOOST_REQUIRE_EQUAL(id.size(), 130);
	BOOST_REQUIRE('0' == id[0] && 'x' == id[1]);

	b = jsonrpcServer->shh_hasIdentity(id);
	BOOST_REQUIRE(b);

	Json::Value t2 = createMessage(id, id);
	b = jsonrpcServer->shh_post(t2);
	BOOST_REQUIRE(b);
}

BOOST_AUTO_TEST_CASE(server)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing server functionality...";

	bool b;
	string s;
	Json::Value j;
	SessionPermissions permissions;
	permissions.privileges.insert(Privilege::Admin);
	string const text = string("0x") + h256::random().hex(); // message must be in raw form

	string sess1 = jsonrpcServer->newSession(permissions);
	string sess2("session number two");
	jsonrpcServer->addSession(sess2, permissions);
	
	b = jsonrpcServer->admin_web3_setVerbosity(5, sess1);
	BOOST_REQUIRE(b);

	b = jsonrpcServer->admin_net_start(sess1);
	BOOST_REQUIRE(b);

	unsigned const step = 10;
	for (unsigned i = 0; i < 3000 && !jsonrpcServer->net_listening(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	b = jsonrpcServer->net_listening();
	BOOST_REQUIRE(b);
	
	b = jsonrpcServer->admin_net_stop(sess1);
	BOOST_REQUIRE(b);

	b = jsonrpcServer->net_listening();
	BOOST_REQUIRE(!b);

	j = jsonrpcServer->admin_net_peers(sess1);
	BOOST_REQUIRE(j.empty());

	j = jsonrpcServer->admin_net_nodeInfo(sess2);
	BOOST_REQUIRE_EQUAL(j["id"].asString(), web3->id().hex());
	BOOST_REQUIRE_EQUAL(j["port"].asUInt(), c_web3port);

	uint16_t port2 = 30339;
	Host host2("shhrpc-host2", NetworkPreferences("127.0.0.1", port2, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();

	b = jsonrpcServer->admin_net_start(sess2);
	BOOST_REQUIRE(b);
	
	for (unsigned i = 0; i < 2000 && !host2.haveNetwork(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	for (unsigned i = 0; i < 2000 && !jsonrpcServer->net_listening(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host2.haveNetwork());
	BOOST_REQUIRE(jsonrpcServer->net_listening());

	string node("enode://");
	node += host2.id().hex();
	node += "@";
	node += "127.0.0.1:30339";
	b = jsonrpcServer->admin_net_connect(node, sess2);

	for (unsigned i = 0; i < 3000 && !host2.peerCount(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE_EQUAL(host2.peerCount(), 1);
	this_thread::sleep_for(chrono::milliseconds(step));

	j = jsonrpcServer->admin_net_peers(sess2);
	BOOST_REQUIRE_EQUAL(j.size(), 1);
	Json::Value peer = j[0];
	s = peer["id"].asString();
	BOOST_REQUIRE_EQUAL(s, host2.id().hex());
	BOOST_REQUIRE_EQUAL(peer["port"].asUInt(), port2);

	s = jsonrpcServer->net_peerCount();
	BOOST_REQUIRE_EQUAL(s, "0x1");

	KeyPair src = KeyPair::create();
	KeyPair dst = KeyPair::create();

	Json::Value t1 = createMessage(toJS(src.address()), toJS(dst.address()));
	b = jsonrpcServer->shh_post(t1);
	BOOST_REQUIRE(b);

	string const id = jsonrpcServer->shh_newIdentity();
	BOOST_REQUIRE_EQUAL(id.size(), 130);
	BOOST_REQUIRE(jsonrpcServer->shh_hasIdentity(id));

	Json::Value t2 = createMessage(id, id);
	b = jsonrpcServer->shh_post(t2);
	BOOST_REQUIRE(b);

	string const nonexistent = "123456789";
	b = jsonrpcServer->shh_uninstallFilter(nonexistent);
	BOOST_REQUIRE(b);

	j = jsonrpcServer->shh_getMessages(nonexistent);
	BOOST_REQUIRE(j.empty());

	string const topic = "unicorns";
	Json::Value t(Json::arrayValue);
	t.append(topic);
	Json::Value f;
	f["to"] = id;
	f["topics"] = t;
	string const filter = jsonrpcServer->shh_newFilter(f);

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE(j.empty());

	j = jsonrpcServer->shh_getMessages(filter);
	BOOST_REQUIRE(j.empty());

	Json::Value msg = createMessage(id, id, topic, text);
	b = jsonrpcServer->shh_post(msg);
	BOOST_REQUIRE(b);
	this_thread::sleep_for(chrono::milliseconds(50));

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE(!j.empty());
	Json::Value m1 = j[0];
	BOOST_REQUIRE_EQUAL(m1["ttl"], toJS(c_ttl));
	BOOST_REQUIRE_EQUAL(m1["from"], id);
	BOOST_REQUIRE_EQUAL(m1["to"], id);
	BOOST_REQUIRE_EQUAL(m1["payload"], text);

	j = jsonrpcServer->shh_getMessages(filter);	
	BOOST_REQUIRE(!j.empty());
	Json::Value m2 = j[0];
	BOOST_REQUIRE_EQUAL(m2["ttl"], toJS(c_ttl));
	BOOST_REQUIRE_EQUAL(m2["from"], id);
	BOOST_REQUIRE_EQUAL(m2["to"], id);
	BOOST_REQUIRE_EQUAL(m2["payload"], text);

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE(j.empty());

	j = jsonrpcServer->shh_getMessages(filter);
	BOOST_REQUIRE(!j.empty());
	m1 = j[0];
	BOOST_REQUIRE_EQUAL(m1["ttl"], toJS(c_ttl));
	BOOST_REQUIRE_EQUAL(m1["from"], id);
	BOOST_REQUIRE_EQUAL(m1["to"], id);
	BOOST_REQUIRE_EQUAL(m1["payload"], text);

	msg = createMessage(id, id, topic);
	b = jsonrpcServer->shh_post(msg);
	BOOST_REQUIRE(b);
	this_thread::sleep_for(chrono::milliseconds(50));

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE_EQUAL(j.size(), 1);

	j = jsonrpcServer->shh_getMessages(filter);
	BOOST_REQUIRE_EQUAL(j.size(), 2);

	b = jsonrpcServer->shh_uninstallFilter(filter);
	BOOST_REQUIRE(b);

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE(j.empty());

	j = jsonrpcServer->shh_getMessages(filter);
	BOOST_REQUIRE(j.empty());

	msg = createMessage(id, id, topic);
	b = jsonrpcServer->shh_post(msg);
	BOOST_REQUIRE(b);
	this_thread::sleep_for(chrono::milliseconds(50));

	j = jsonrpcServer->shh_getFilterChanges(filter);
	BOOST_REQUIRE(j.empty());

	j = jsonrpcServer->shh_getMessages(filter);
	BOOST_REQUIRE(j.empty());

	b = jsonrpcServer->admin_net_stop(sess2);
	BOOST_REQUIRE(b);

	b = jsonrpcServer->net_listening();
	BOOST_REQUIRE(!b);
}

BOOST_AUTO_TEST_SUITE_END()
