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
/** @file main.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * Ethereum client.
 */
#include <functional>
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libdevcore/CommonIO.h>
#include <libp2p/All.h>
#include <libdevcore/RangeMask.h>
#include <libethereum/DownloadMan.h>
#include <libwhisper/WhisperPeer.h>
#include <libwhisper/WhisperHost.h>
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;

#if 1
int main()
{
	DownloadMan man;
	DownloadSub s0(man);
	DownloadSub s1(man);
	DownloadSub s2(man);
	man.resetToChain(h256s({u256(0), u256(1), u256(2), u256(3), u256(4), u256(5), u256(6), u256(7), u256(8)}));
	assert((s0.nextFetch(2) == h256Set{(u256)7, (u256)8}));
	assert((s1.nextFetch(2) == h256Set{(u256)5, (u256)6}));
	assert((s2.nextFetch(2) == h256Set{(u256)3, (u256)4}));
	s0.noteBlock(u256(8));
	s0.doneFetch();
	assert((s0.nextFetch(2) == h256Set{(u256)2, (u256)7}));
	s1.noteBlock(u256(6));
	s1.noteBlock(u256(5));
	s1.doneFetch();
	assert((s1.nextFetch(2) == h256Set{(u256)0, (u256)1}));
	s0.doneFetch();				// TODO: check exact semantics of doneFetch & nextFetch. Not sure if they're right -> doneFetch calls resetFetch which kills all the info of past fetches.
	cdebug << s0.nextFetch(2);
	assert((s0.nextFetch(2) == h256Set{(u256)3, (u256)4}));

/*	RangeMask<unsigned> m(0, 100);
	cnote << m;
	m += UnsignedRange(3, 10);
	cnote << m;
	m += UnsignedRange(11, 16);
	cnote << m;
	m += UnsignedRange(10, 11);
	cnote << m;
	cnote << ~m;
	cnote << (~m).lowest(10);
	for (auto i: (~m).lowest(10))
		cnote << i;*/
	return 0;
}
#endif

/*int other(bool& o_started)
{
	setThreadName("other");

	short listenPort = 30300;

	Host ph("Test", NetworkPreferences(listenPort, "", false, true));
	auto wh = ph.registerCapability(new WhisperHost());

	ph.start();

	o_started = true;

	/// Only interested in odd packets
	auto w = wh->installWatch(BuildTopicMask()("odd"));

	unsigned last = 0;
	unsigned total = 0;

	for (int i = 0; i < 100 && last < 81; ++i)
	{
		for (auto i: wh->checkWatch(w))
		{
			Message msg = wh->envelope(i).open();
			last = RLP(msg.payload()).toInt<unsigned>();
			cnote << "New message from:" << msg.from().abridged() << RLP(msg.payload()).toInt<unsigned>();
			total += last;
		}
		this_thread::sleep_for(chrono::milliseconds(50));
	}
	return total;
}

int main(int, char**)
{
	g_logVerbosity = 0;

	bool started = false;
	unsigned result;
	std::thread listener([&](){ return (result = other(started)); });
	while (!started)
		this_thread::sleep_for(chrono::milliseconds(50));

	short listenPort = 30303;
	string remoteHost = "127.0.0.1";
	short remotePort = 30300;

	Host ph("Test", NetworkPreferences(listenPort, "", false, true));
	auto wh = ph.registerCapability(new WhisperHost());

	ph.start();

	if (!remoteHost.empty())
		ph.connect(remoteHost, remotePort);

	KeyPair us = KeyPair::create();
	for (int i = 0; i < 10; ++i)
	{
		wh->post(us.sec(), RLPStream().append(i * i).out(), BuildTopic(i)(i % 2 ? "odd" : "even"));
		this_thread::sleep_for(chrono::milliseconds(250));
	}

	listener.join();
	assert(result == 1 + 9 + 25 + 49 + 81);

	return 0;
}*/
