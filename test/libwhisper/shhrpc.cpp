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

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
namespace js = json_spirit;

WebThreeDirect* web3;
unique_ptr<WebThreeStubServer> jsonrpcServer;
unique_ptr<WebThreeStubClient> jsonrpcClient;

string fromAscii(string _s) { return toHex(asBytes(_s), 2, HexPrefix::Add); }

struct Setup
{
	Setup()
	{
		static bool setup = false;
		if (!setup)
		{
			setup = true;
			NetworkPreferences nprefs(std::string(), 30333, false);
			web3 = new WebThreeDirect("++eth tests", "", WithExisting::Trust, {"eth", "shh"}, nprefs);		
			web3->setIdealPeerCount(1);
			web3->ethereum()->setForceMining(false);
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
};

BOOST_FIXTURE_TEST_SUITE(shhrpc, Setup)

BOOST_AUTO_TEST_CASE(first)
{
	cnote << "Testing shh rpc...";

	web3->startNetwork();
	unsigned const step = 10;
	for (unsigned i = 0; i < 3000 && !web3->haveNetwork(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(web3->haveNetwork());

	uint16_t const port2 = 30334;	
	NetworkPreferences prefs("127.0.0.1", port2, false);
	Host host2("Test", prefs);
	host2.start();

	for (unsigned i = 0; i < 3000 && !host2.haveNetwork(); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host2.haveNetwork());

	web3->requirePeer(host2.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port2, port2));

	for (unsigned i = 0; i < 4000 && (!web3->peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(web3->peerCount());
	BOOST_REQUIRE(host2.peerCount());	
}

BOOST_AUTO_TEST_SUITE_END()
