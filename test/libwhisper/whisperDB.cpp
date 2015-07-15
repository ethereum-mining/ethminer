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
/** @file whisperMessage.cpp
* @author Vladislav Gluhovsky <vlad@ethdev.com>
* @date July 2015
*/

#include <thread>
#include <boost/test/unit_test.hpp>
#include <libp2p/Host.h>
#include <libwhisper/WhisperDB.h>
#include <libwhisper/WhisperHost.h>

using namespace std;
using namespace dev;
using namespace dev::shh;

BOOST_AUTO_TEST_SUITE(whisperDB)

BOOST_AUTO_TEST_CASE(basic)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing Whisper DB...";

	string s;
	string const text1 = "lorem";
	string const text2 = "ipsum";
	h256 h1(0xBEEF);
	h256 h2(0xC0FFEE);
	WhisperDB db;

	db.kill(h1);
	db.kill(h2);

	s = db.lookup(h1);
	BOOST_REQUIRE(s.empty());

	db.insert(h1, text2);
	s = db.lookup(h2);
	BOOST_REQUIRE(s.empty());
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text2));

	db.insert(h1, text1);
	s = db.lookup(h2);
	BOOST_REQUIRE(s.empty());
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text1));

	db.insert(h2, text2);
	s = db.lookup(h2);
	BOOST_REQUIRE(!s.compare(text2));
	s = db.lookup(h1);
	BOOST_REQUIRE(!s.compare(text1));

	db.kill(h1);
	db.kill(h2);

	s = db.lookup(h2);
	BOOST_REQUIRE(s.empty());
	s = db.lookup(h1);
	BOOST_REQUIRE(s.empty());
}

BOOST_AUTO_TEST_CASE(persistence)
{
	VerbosityHolder setTemporaryLevel(10);
	cnote << "Testing persistence of Whisper DB...";

	string s;
	string const text1 = "sator";
	string const text2 = "arepo";
	h256 const h1(0x12345678);
	h256 const h2(0xBADD00DE);

	{
		WhisperDB db;
		db.kill(h1);
		db.kill(h2);
		s = db.lookup(h1);
		BOOST_REQUIRE(s.empty());
		db.insert(h1, text2);
		s = db.lookup(h2);
		BOOST_REQUIRE(s.empty());
		s = db.lookup(h1);
		BOOST_REQUIRE(!s.compare(text2));
	}

	this_thread::sleep_for(chrono::milliseconds(20));

	{
		WhisperDB db;
		db.insert(h1, text1);
		db.insert(h2, text2);
	}

	this_thread::sleep_for(chrono::milliseconds(20));

	{
		WhisperDB db;
		s = db.lookup(h2);
		BOOST_REQUIRE(!s.compare(text2));
		s = db.lookup(h1);
		BOOST_REQUIRE(!s.compare(text1));
		db.kill(h1);
		db.kill(h2);
	}
}

BOOST_AUTO_TEST_CASE(messages)
{
	cnote << "Testing load/save Whisper messages...";
	VerbosityHolder setTemporaryLevel(2);
	unsigned const TestSize = 3;
	map<h256, Envelope> m1;
	map<h256, Envelope> preexisting;
	KeyPair us = KeyPair::create();

	{
		p2p::Host h("Test");
		auto wh = h.registerCapability(new WhisperHost());
		preexisting = wh->all();
		cnote << preexisting.size() << "preexisting messages in DB";

		for (unsigned i = 0; i < TestSize; ++i)
			wh->post(us.sec(), RLPStream().append(i).out(), BuildTopic("test"), 0xFFFFF);

		m1 = wh->all();
	}

	{
		p2p::Host h("Test");
		auto wh = h.registerCapability(new WhisperHost());
		map<h256, Envelope> m2 = wh->all();
		BOOST_REQUIRE_EQUAL(m1.size(), m2.size());
		BOOST_REQUIRE_EQUAL(m1.size() - preexisting.size(), TestSize);

		for (auto i: m1)
		{
			RLPStream rlp1;
			RLPStream rlp2;
			i.second.streamRLP(rlp1);
			m2[i.first].streamRLP(rlp2);
			BOOST_REQUIRE_EQUAL(rlp1.out().size(), rlp2.out().size());
			for (unsigned j = 0; j < rlp1.out().size(); ++j)
				BOOST_REQUIRE_EQUAL(rlp1.out()[j], rlp2.out()[j]);
		}
	}

	WhisperDB db;
	unsigned x = 0;

	for (auto i: m1)
		if (preexisting.find(i.first) == preexisting.end())
		{
			db.kill(i.first);
			++x;
		}

	BOOST_REQUIRE_EQUAL(x, TestSize);
}

BOOST_AUTO_TEST_CASE(corruptedData)
{
	cnote << "Testing corrupted data...";
	VerbosityHolder setTemporaryLevel(2);
	map<h256, Envelope> m;
	h256 x = h256::random();

	{
		WhisperDB db;
		db.insert(x, "this is a test input, representing corrupt data");
	}

	{
		p2p::Host h("Test");
		auto wh = h.registerCapability(new WhisperHost());
		m = wh->all();
		BOOST_REQUIRE(m.end() == m.find(x));
	}

	{
		WhisperDB db;
		string s = db.lookup(x);
		BOOST_REQUIRE(s.empty());
	}
}

BOOST_AUTO_TEST_SUITE_END()
