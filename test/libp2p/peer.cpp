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
/** @file peer.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Peer Network test functions.
 */

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <thread>
#include <libp2p/Host.h>
#include <test/TestHelper.h>

using namespace std;
using namespace dev;
using namespace dev::p2p;

struct P2PFixture
{
	P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = true; }
	~P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = false; }
};

BOOST_FIXTURE_TEST_SUITE(p2p, P2PFixture)

BOOST_AUTO_TEST_CASE(host)
{
	if (test::Options::get().nonetwork)
		return;

	VerbosityHolder setTemporaryLevel(10);	
	NetworkPreferences host1prefs("127.0.0.1", 30321, false);
	NetworkPreferences host2prefs("127.0.0.1", 30322, false);
	Host host1("Test", host1prefs);
	Host host2("Test", host2prefs);
	host1.start();
	host2.start();
	auto node2 = host2.id();
	int const step = 10;

	for (int i = 0; i < 3000 && (!host1.isStarted() || !host2.isStarted()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.isStarted() && host2.isStarted());
	
	for (int i = 0; i < 3000 && (!host1.haveNetwork() || !host2.haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.haveNetwork() && host2.haveNetwork());
	host1.addNode(node2, NodeIPEndpoint(bi::address::from_string("127.0.0.1"), host2prefs.listenPort, host2prefs.listenPort));

	for (int i = 0; i < 3000 && (!host1.peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE_EQUAL(host1.peerCount(), 1);
	BOOST_REQUIRE_EQUAL(host2.peerCount(), 1);
}

BOOST_AUTO_TEST_CASE(networkConfig)
{
	if (test::Options::get().nonetwork)
		return;

	Host save("Test", NetworkPreferences(false));
	bytes store(save.saveNetwork());
	
	Host restore("Test", NetworkPreferences(false), bytesConstRef(&store));
	BOOST_REQUIRE(save.id() == restore.id());
}

BOOST_AUTO_TEST_CASE(saveNodes)
{
	if (test::Options::get().nonetwork)
		return;

	VerbosityHolder reduceVerbosity(2);

	std::list<Host*> hosts;
	unsigned const c_step = 10;
	unsigned const c_nodes = 6;
	unsigned const c_peers = c_nodes - 1;

	for (unsigned i = 0; i < c_nodes; ++i)
	{
		Host* h = new Host("Test", NetworkPreferences("127.0.0.1", 30325 + i, false));
		h->setIdealPeerCount(10);
		// starting host is required so listenport is available
		h->start();
		while (!h->haveNetwork())
			this_thread::sleep_for(chrono::milliseconds(c_step));
		hosts.push_back(h);
	}
	
	Host& host = *hosts.front();
	for (auto const& h: hosts)
		host.addNode(h->id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), h->listenPort(), h->listenPort()));

	for (unsigned i = 0; i < c_peers * 1000 && host.peerCount() < c_peers; i += c_step)
		this_thread::sleep_for(chrono::milliseconds(c_step));

	Host& host2 = *hosts.back();
	for (auto const& h: hosts)
		host2.addNode(h->id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), h->listenPort(), h->listenPort()));

	for (unsigned i = 0; i < c_peers * 1000 && host2.peerCount() < c_peers; i += c_step)
		this_thread::sleep_for(chrono::milliseconds(c_step));

	BOOST_CHECK_EQUAL(host.peerCount(), c_peers);
	BOOST_CHECK_EQUAL(host2.peerCount(), c_peers);

	bytes firstHostNetwork(host.saveNetwork());
	bytes secondHostNetwork(host.saveNetwork());	
	BOOST_REQUIRE_EQUAL(sha3(firstHostNetwork), sha3(secondHostNetwork));	
	
	RLP r(firstHostNetwork);
	BOOST_REQUIRE(r.itemCount() == 3);
	BOOST_REQUIRE(r[0].toInt<unsigned>() == dev::p2p::c_protocolVersion);
	BOOST_REQUIRE_EQUAL(r[1].toBytes().size(), 32); // secret
	BOOST_REQUIRE(r[2].itemCount() >= c_nodes);
	
	for (auto i: r[2])
	{
		BOOST_REQUIRE(i.itemCount() == 4 || i.itemCount() == 11);
		BOOST_REQUIRE(i[0].size() == 4 || i[0].size() == 16);
	}

	for (auto host: hosts)
		delete host;
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(p2pPeer, P2PFixture)

BOOST_AUTO_TEST_CASE(requirePeer)
{
	if (test::Options::get().nonetwork)
		return;

	VerbosityHolder temporaryLevel(10);

	unsigned const step = 10;
	const char* const localhost = "127.0.0.1";
	NetworkPreferences prefs1(localhost, 30323, false);
	NetworkPreferences prefs2(localhost, 30324, false);
	Host host1("Test", prefs1);
	host1.start();

	Host host2("Test", prefs2);
	host2.start();

	auto node2 = host2.id();
	host1.requirePeer(node2, NodeIPEndpoint(bi::address::from_string(localhost), prefs2.listenPort, prefs2.listenPort));

	for (unsigned i = 0; i < 3000 && (!host1.peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	auto host1peerCount = host1.peerCount();
	auto host2peerCount = host2.peerCount();
	BOOST_REQUIRE_EQUAL(host1peerCount, 1);
	BOOST_REQUIRE_EQUAL(host2peerCount, 1);

	PeerSessionInfos sis1 = host1.peerSessionInfo();
	PeerSessionInfos sis2 = host2.peerSessionInfo();

	BOOST_REQUIRE_EQUAL(sis1.size(), 1);
	BOOST_REQUIRE_EQUAL(sis2.size(), 1);

	Peers peers1 = host1.getPeers();
	Peers peers2 = host2.getPeers();
	BOOST_REQUIRE_EQUAL(peers1.size(), 1);
	BOOST_REQUIRE_EQUAL(peers2.size(), 1);

	DisconnectReason disconnect1 = peers1[0].lastDisconnect();
	DisconnectReason disconnect2 = peers2[0].lastDisconnect();
	BOOST_REQUIRE_EQUAL(disconnect1, disconnect2);

	host1.relinquishPeer(node2);

	for (unsigned i = 0; i < 2000 && (host1.peerCount() || host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	host1peerCount = host1.peerCount();
	host2peerCount = host2.peerCount();
	BOOST_REQUIRE_EQUAL(host1peerCount, 1);
	BOOST_REQUIRE_EQUAL(host2peerCount, 1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(peerTypes)

BOOST_AUTO_TEST_CASE(emptySharedPeer)
{
	if (test::Options::get().nonetwork)
		return;

	shared_ptr<Peer> p;
	BOOST_REQUIRE(!p);
	
	std::map<NodeId, std::shared_ptr<Peer>> peers;
	p = peers[NodeId()];
	BOOST_REQUIRE(!p);
	
	p.reset(new Peer(UnspecifiedNode));
	BOOST_REQUIRE(!p->id);
	BOOST_REQUIRE(!*p);
	
	p.reset(new Peer(Node(NodeId(EmptySHA3), UnspecifiedNodeIPEndpoint)));
	BOOST_REQUIRE(!(!*p));
	BOOST_REQUIRE(*p);
	BOOST_REQUIRE(p);
}

BOOST_AUTO_TEST_SUITE_END()

int peerTest(int argc, char** argv)
{
	Public remoteAlias;
	short listenPort = 30304;
	string remoteHost;
	short remotePort = 30304;
	
	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-l" && i + 1 < argc)
			listenPort = (short)atoi(argv[++i]);
		else if (arg == "-r" && i + 1 < argc)
			remoteHost = argv[++i];
		else if (arg == "-p" && i + 1 < argc)
			remotePort = (short)atoi(argv[++i]);
		else if (arg == "-ra" && i + 1 < argc)
			remoteAlias = Public(dev::fromHex(argv[++i]));
		else
			remoteHost = argv[i];
	}

	Host ph("Test", NetworkPreferences(listenPort));

	if (!remoteHost.empty() && !remoteAlias)
		ph.addNode(remoteAlias, NodeIPEndpoint(bi::address::from_string(remoteHost), remotePort, remotePort));

	this_thread::sleep_for(chrono::milliseconds(200));

	return 0;
}

