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
/** @file whisperTopic.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */
#include <functional>

#include <boost/test/unit_test.hpp>

#include <libp2p/Host.h>
#include <libp2p/Session.h>
#include <libwhisper/WhisperPeer.h>
#include <libwhisper/WhisperHost.h>
#include <test/TestHelper.h>

using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::shh;

struct P2PFixture
{
	P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = true; }
	~P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = false; }
};

BOOST_FIXTURE_TEST_SUITE(whisper, P2PFixture)

BOOST_AUTO_TEST_CASE(topic)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing Whisper...";
	VerbosityHolder setTemporaryLevel(0);

	uint16_t port1 = 30311;
	Host host1("Test", NetworkPreferences("127.0.0.1", port1, false));
	host1.setIdealPeerCount(1);
	auto whost1 = host1.registerCapability(new WhisperHost());
	host1.start();

	bool host1Ready = false;
	unsigned result = 0;
	unsigned const step = 10;

	std::thread listener([&]()
	{
		setThreadName("other");
		
		/// Only interested in odd packets
		auto w = whost1->installWatch(BuildTopicMask("odd"));
		host1Ready = true;
		set<unsigned> received;
		for (int iterout = 0, last = 0; iterout < 200 && last < 81; ++iterout)
		{
			for (auto i: whost1->checkWatch(w))
			{
				Message msg = whost1->envelope(i).open(whost1->fullTopics(w));
				last = RLP(msg.payload()).toInt<unsigned>();
				if (received.count(last))
					continue;
				received.insert(last);
				cnote << "New message from:" << msg.from() << RLP(msg.payload()).toInt<unsigned>();
				result += last;
			}
			this_thread::sleep_for(chrono::milliseconds(50));
		}
	});

	Host host2("Test", NetworkPreferences("127.0.0.1", 30310, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();

	for (unsigned i = 0; i < 3000 && (!host1.haveNetwork() || !host2.haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.haveNetwork());
	BOOST_REQUIRE(host2.haveNetwork());

	host2.addNode(host1.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port1, port1));

	for (unsigned i = 0; i < 3000 && (!host1.peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.peerCount());
	BOOST_REQUIRE(host2.peerCount());

	for (unsigned i = 0; i < 3000 && !host1Ready; i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1Ready);
	
	KeyPair us = KeyPair::create();
	for (int i = 0; i < 10; ++i)
	{
		whost2->post(us.sec(), RLPStream().append(i * i).out(), BuildTopic(i)(i % 2 ? "odd" : "even"));
		this_thread::sleep_for(chrono::milliseconds(50));
	}

	listener.join();
	BOOST_REQUIRE_EQUAL(result, 1 + 9 + 25 + 49 + 81);
}

BOOST_AUTO_TEST_CASE(forwarding)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing Whisper forwarding...";
	VerbosityHolder setTemporaryLevel(0);

	// Host must be configured not to share peers.
	uint16_t port1 = 30312;
	Host host1("Listner", NetworkPreferences("127.0.0.1", port1, false));
	host1.setIdealPeerCount(1);
	auto whost1 = host1.registerCapability(new WhisperHost());
	host1.start();
	while (!host1.haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(2));

	unsigned result = 0;
	bool done = false;

	bool startedListener = false;
	std::thread listener([&]()
	{
		setThreadName("listener");

		startedListener = true;

		/// Only interested in odd packets
		auto w = whost1->installWatch(BuildTopicMask("test"));

		for (int i = 0; i < 200 && !result; ++i)
		{
			for (auto i: whost1->checkWatch(w))
			{
				Message msg = whost1->envelope(i).open(whost1->fullTopics(w));
				unsigned last = RLP(msg.payload()).toInt<unsigned>();
				cnote << "New message from:" << msg.from() << RLP(msg.payload()).toInt<unsigned>();
				result = last;
			}
			this_thread::sleep_for(chrono::milliseconds(50));
		}
	});


	// Host must be configured not to share peers.
	uint16_t port2 = 30313;
	Host host2("Forwarder", NetworkPreferences("127.0.0.1", port2, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	host2.start();
	while (!host2.haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(2));

	Public fwderid;
	bool startedForwarder = false;
	std::thread forwarder([&]()
	{
		setThreadName("forwarder");

		while (!startedListener)
			this_thread::sleep_for(chrono::milliseconds(50));

		this_thread::sleep_for(chrono::milliseconds(500));
		host2.addNode(host1.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port1, port1));

		startedForwarder = true;

		/// Only interested in odd packets
		auto w = whost2->installWatch(BuildTopicMask("test"));

		while (!done)
		{
			for (auto i: whost2->checkWatch(w))
			{
				Message msg = whost2->envelope(i).open(whost2->fullTopics(w));
				cnote << "New message from:" << msg.from() << RLP(msg.payload()).toInt<unsigned>();
			}
			this_thread::sleep_for(chrono::milliseconds(50));
		}
	});

	while (!startedForwarder)
		this_thread::sleep_for(chrono::milliseconds(50));

	Host ph("Sender", NetworkPreferences("127.0.0.1", 30314, false));
	ph.setIdealPeerCount(1);
	shared_ptr<WhisperHost> wh = ph.registerCapability(new WhisperHost());
	ph.start();
	ph.addNode(host2.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port2, port2));
	while (!ph.haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(10));

	while (!ph.peerCount())
		this_thread::sleep_for(chrono::milliseconds(10));

	KeyPair us = KeyPair::create();
	wh->post(us.sec(), RLPStream().append(1).out(), BuildTopic("test"));
	this_thread::sleep_for(chrono::milliseconds(250));

	listener.join();
	done = true;
	forwarder.join();
	BOOST_REQUIRE_EQUAL(result, 1);
}

BOOST_AUTO_TEST_CASE(asyncforwarding)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing Whisper async forwarding...";
	VerbosityHolder setTemporaryLevel(2);
	unsigned const TestValue = 8456;
	unsigned result = 0;
	bool done = false;

	// Host must be configured not to share peers.
	uint16_t port1 = 30315;
	Host host1("Forwarder", NetworkPreferences("127.0.0.1", port1, false));
	host1.setIdealPeerCount(1);
	auto whost1 = host1.registerCapability(new WhisperHost());
	host1.start();
	while (!host1.haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(2));

	auto w = whost1->installWatch(BuildTopicMask("test")); // only interested in odd packets
	bool startedForwarder = false;
	std::thread forwarder([&]()
	{
		setThreadName("forwarder");
		this_thread::sleep_for(chrono::milliseconds(50));
		startedForwarder = true;
		while (!done)
		{
			for (auto i: whost1->checkWatch(w))
			{
				Message msg = whost1->envelope(i).open(whost1->fullTopics(w));
				cnote << "New message from:" << msg.from() << RLP(msg.payload()).toInt<unsigned>();
			}
			this_thread::sleep_for(chrono::milliseconds(50));
		}
	});

	while (!startedForwarder)
		this_thread::sleep_for(chrono::milliseconds(2));

	{
		Host host2("Sender", NetworkPreferences("127.0.0.1", 30316, false));
		host2.setIdealPeerCount(1);
		shared_ptr<WhisperHost> whost2 = host2.registerCapability(new WhisperHost());
		host2.start();
		while (!host2.haveNetwork())
			this_thread::sleep_for(chrono::milliseconds(2));

		host2.requirePeer(host1.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port1, port1));
		while (!host2.peerCount() || !host1.peerCount())
			this_thread::sleep_for(chrono::milliseconds(5));

		KeyPair us = KeyPair::create();
		whost2->post(us.sec(), RLPStream().append(TestValue).out(), BuildTopic("test"), 777000);
		this_thread::sleep_for(chrono::milliseconds(250));
	}

	{
		Host ph("Listener", NetworkPreferences("127.0.0.1", 30317, false));
		ph.setIdealPeerCount(1);
		shared_ptr<WhisperHost> wh = ph.registerCapability(new WhisperHost());
		ph.start();
		while (!ph.haveNetwork())
			this_thread::sleep_for(chrono::milliseconds(2));

		auto w = wh->installWatch(BuildTopicMask("test")); // only interested in odd packets
		ph.requirePeer(host1.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port1, port1));

		for (int i = 0; i < 200 && !result; ++i)
		{
			for (auto i: wh->checkWatch(w))
			{
				Message msg = wh->envelope(i).open(wh->fullTopics(w));
				unsigned last = RLP(msg.payload()).toInt<unsigned>();
				cnote << "New message from:" << msg.from() << RLP(msg.payload()).toInt<unsigned>();
				result = last;
			}
			this_thread::sleep_for(chrono::milliseconds(50));
		}
	}

	done = true;
	forwarder.join();
	BOOST_REQUIRE_EQUAL(result, TestValue);
}

BOOST_AUTO_TEST_CASE(topicAdvertising)
{
	if (test::Options::get().nonetwork)
		return;

	cnote << "Testing Topic Advertising...";
	VerbosityHolder setTemporaryLevel(2);

	Host host1("first", NetworkPreferences("127.0.0.1", 30319, false));
	host1.setIdealPeerCount(1);
	auto whost1 = host1.registerCapability(new WhisperHost());
	host1.start();
	while (!host1.haveNetwork())
		this_thread::sleep_for(chrono::milliseconds(10));

	unsigned const step = 10;
	uint16_t port2 = 30318;
	Host host2("second", NetworkPreferences("127.0.0.1", port2, false));
	host2.setIdealPeerCount(1);
	auto whost2 = host2.registerCapability(new WhisperHost());
	unsigned w2 = whost2->installWatch(BuildTopicMask("test2"));
	host2.start();

	for (unsigned i = 0; i < 3000 && (!host1.haveNetwork() || !host2.haveNetwork()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	host1.addNode(host2.id(), NodeIPEndpoint(bi::address::from_string("127.0.0.1"), port2, port2));

	for (unsigned i = 0; i < 3000 && (!host1.peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	std::vector<std::pair<std::shared_ptr<Session>, std::shared_ptr<Peer>>> sessions;
	TopicBloomFilterHash bf1;

	for (int i = 0; i < 600 && !bf1; ++i)
	{
		sessions = whost1->peerSessions();
		if (!sessions.empty())
			bf1 = sessions.back().first->cap<WhisperPeer>()->bloom();

		this_thread::sleep_for(chrono::milliseconds(step));
	}

	BOOST_REQUIRE(sessions.size());
	TopicBloomFilterHash bf2 = whost2->bloom();
	BOOST_REQUIRE_EQUAL(bf1, bf2);
	BOOST_REQUIRE(bf1);
	BOOST_REQUIRE(!whost1->bloom());

	unsigned w1 = whost1->installWatch(BuildTopicMask("test1"));
	bf2 = TopicBloomFilterHash();

	for (int i = 0; i < 600 && !bf2; ++i)
	{
		sessions = whost2->peerSessions();
		if (!sessions.empty())
			bf2 = sessions.back().first->cap<WhisperPeer>()->bloom();

		this_thread::sleep_for(chrono::milliseconds(step));
	}

	BOOST_REQUIRE(sessions.size());
	BOOST_REQUIRE_EQUAL(sessions.back().second->id, host1.id());
	bf1 = whost1->bloom();
	BOOST_REQUIRE_EQUAL(bf1, bf2);
	BOOST_REQUIRE(bf1);

	unsigned random = 0xC0FFEE;
	whost1->uninstallWatch(w1);
	whost1->uninstallWatch(random);
	whost1->uninstallWatch(w1);
	whost1->uninstallWatch(random);
	whost2->uninstallWatch(random);
	whost2->uninstallWatch(w2);
	whost2->uninstallWatch(random);
	whost2->uninstallWatch(w2);
}

BOOST_AUTO_TEST_CASE(selfAddressed)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing self-addressed messaging with bloom filter matching...";

	char const* text = "deterministic pseudorandom test";
	BuildTopicMask mask(text);

	Host host("first", NetworkPreferences("127.0.0.1", 30320, false));
	auto wh = host.registerCapability(new WhisperHost());
	auto watch = wh->installWatch(BuildTopicMask(text));

	unsigned const sample = 0xFEED;
	KeyPair us = KeyPair::create();
	wh->post(us.sec(), RLPStream().append(sample).out(), BuildTopic(text));

	TopicBloomFilterHash f = wh->bloom();
	Envelope e = Message(RLPStream().append(sample).out()).seal(us.sec(), BuildTopic(text), 50, 50);
	bool ok = e.matchesBloomFilter(f);
	BOOST_REQUIRE(ok);

	this_thread::sleep_for(chrono::milliseconds(50));

	unsigned single = 0;
	unsigned result = 0;
	for (auto j: wh->checkWatch(watch))
	{
		Message msg = wh->envelope(j).open(wh->fullTopics(watch));
		single = RLP(msg.payload()).toInt<unsigned>();
		result += single;
	}

	BOOST_REQUIRE_EQUAL(sample, result);
}

BOOST_AUTO_TEST_SUITE_END()
