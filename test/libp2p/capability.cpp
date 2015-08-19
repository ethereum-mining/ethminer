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
/** @file capability.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date May 2015
*/

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <thread>
#include <libp2p/Common.h>
#include <libp2p/Host.h>
#include <libp2p/Session.h>
#include <libp2p/Capability.h>
#include <libp2p/HostCapability.h>
#include <test/TestHelper.h>

using namespace std;
using namespace dev;
using namespace dev::p2p;

struct P2PFixture
{
	P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = true; }
	~P2PFixture() { dev::p2p::NodeIPEndpoint::test_allowLocal = false; }
};

class TestCapability: public Capability
{
public:
	TestCapability(std::shared_ptr<Session> _s, HostCapabilityFace* _h, unsigned _idOffset, CapDesc const&): Capability(_s, _h, _idOffset), m_cntReceivedMessages(0), m_testSum(0) {}
	virtual ~TestCapability() {}
	int countReceivedMessages() { return m_cntReceivedMessages; }
	int testSum() { return m_testSum; }
	static std::string name() { return "test"; }
	static u256 version() { return 2; }
	static unsigned messageCount() { return UserPacket + 1; }
	void sendTestMessage(int _i) { RLPStream s; sealAndSend(prep(s, UserPacket, 1) << _i); }

protected:
	virtual bool interpret(unsigned _id, RLP const& _r) override;

	int m_cntReceivedMessages;
	int m_testSum;
};

bool TestCapability::interpret(unsigned _id, RLP const& _r) 
{
	cnote << "Capability::interpret(): custom message received";
	++m_cntReceivedMessages;
	m_testSum += _r[0].toInt();
	BOOST_ASSERT(_id == UserPacket);
	return (_id == UserPacket);
}

class TestHostCapability: public HostCapability<TestCapability>, public Worker
{
public:
	TestHostCapability(): Worker("test") {}
	virtual ~TestHostCapability() {}

	void sendTestMessage(NodeID const& _id, int _x)
	{
		for (auto i: peerSessions())
			if (_id == i.second->id)
				i.first->cap<TestCapability>().get()->sendTestMessage(_x);
	}

	std::pair<int, int> retrieveTestData(NodeID const& _id)
	{ 
		int cnt = 0;
		int checksum = 0;
		for (auto i: peerSessions())
			if (_id == i.second->id)
			{
				cnt += i.first->cap<TestCapability>().get()->countReceivedMessages();
				checksum += i.first->cap<TestCapability>().get()->testSum();
			}

		return std::pair<int, int>(cnt, checksum);
	}
};

BOOST_FIXTURE_TEST_SUITE(p2pCapability, P2PFixture)

BOOST_AUTO_TEST_CASE(capability)
{
	if (test::Options::get().nonetwork)
		return;

	if (g_logVerbosity != -1)
		VerbosityHolder verbosityHolder(10);

	cnote << "Testing Capability...";

	const char* const localhost = "127.0.0.1";
	NetworkPreferences prefs1(localhost, 30301, false);
	NetworkPreferences prefs2(localhost, 30302, false);

	Host host1("Test", prefs1);
	auto thc1 = host1.registerCapability(make_shared<TestHostCapability>());
	host1.start();	

	Host host2("Test", prefs2);
	auto thc2 = host2.registerCapability(make_shared<TestHostCapability>());
	host2.start();

	int const step = 10;

	for (int i = 0; i < 3000 && (!host1.isStarted() || !host2.isStarted()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.isStarted() && host2.isStarted());
	host1.requirePeer(host2.id(), NodeIPEndpoint(bi::address::from_string(localhost), prefs2.listenPort, prefs2.listenPort));

	for (int i = 0; i < 3000 && (!host1.peerCount() || !host2.peerCount()); i += step)
		this_thread::sleep_for(chrono::milliseconds(step));

	BOOST_REQUIRE(host1.peerCount() > 0 && host2.peerCount() > 0);

	int const target = 7;
	int checksum = 0;
	for (int i = 0; i < target; checksum += i++)
		thc2->sendTestMessage(host1.id(), i);

	this_thread::sleep_for(chrono::seconds(1));
	std::pair<int, int> testData = thc1->retrieveTestData(host2.id());
	BOOST_REQUIRE_EQUAL(target, testData.first);
	BOOST_REQUIRE_EQUAL(checksum, testData.second);
}

BOOST_AUTO_TEST_SUITE_END()


