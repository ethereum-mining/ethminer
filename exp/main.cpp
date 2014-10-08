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
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;

int main()
{
	DownloadMan man;
	DownloadSub s0(man);
	DownloadSub s1(man);
	DownloadSub s2(man);
	man.resetToChain(h256s({u256(0), u256(1), u256(2), u256(3), u256(4), u256(5), u256(6), u256(7), u256(8)}));
	cnote << s0.nextFetch(2);
	cnote << s1.nextFetch(2);
	cnote << s2.nextFetch(2);
	s0.noteBlock(u256(0));
	s0.doneFetch();
	cnote << s0.nextFetch(2);
	s1.noteBlock(u256(2));
	s1.noteBlock(u256(3));
	s1.doneFetch();
	cnote << s1.nextFetch(2);
	s0.doneFetch();
	cnote << s0.nextFetch(2);

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

/*
int main(int argc, char** argv)
{
	g_logVerbosity = 20;

	short listenPort = 30303;
	string remoteHost;
	short remotePort = 30303;

	for (int i = 1; i < argc; ++i)
	{
		string arg = argv[i];
		if (arg == "-l" && i + 1 < argc)
			listenPort = (short)atoi(argv[++i]);
		else if (arg == "-r" && i + 1 < argc)
			remoteHost = argv[++i];
		else if (arg == "-p" && i + 1 < argc)
			remotePort = (short)atoi(argv[++i]);
		else
			remoteHost = argv[i];
	}

	Host ph("Test", NetworkPreferences(listenPort, "", false, true));
	ph.registerCapability(new WhisperHost());
	auto wh = ph.cap<WhisperHost>();

	ph.start();

	if (!remoteHost.empty())
		ph.connect(remoteHost, remotePort);

	/// Only interested in the packet if the lowest bit is 1
	auto w = wh->installWatch(MessageFilter(std::vector<std::pair<bytes, bytes> >({{fromHex("0000000000000000000000000000000000000000000000000000000000000001"), fromHex("0000000000000000000000000000000000000000000000000000000000000001")}})));


	for (int i = 0; ; ++i)
	{
		wh->sendRaw(h256(u256(i * i)).asBytes(), h256(u256(i)).asBytes(), 1000);
		for (auto i: wh->checkWatch(w))
			cnote << "New message:" << (u256)h256(wh->message(i).payload);
	}
	return 0;
}
*/
