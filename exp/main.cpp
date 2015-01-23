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
#include <libethereum/AccountDiff.h>
#include <libdevcore/RangeMask.h>
#include <libdevcore/Log.h>
#include <libdevcore/Common.h>
#include <libdevcore/CommonData.h>
#include <libdevcore/RLP.h>
#include <libdevcore/CommonIO.h>
#include <libp2p/All.h>
#include <libethereum/DownloadMan.h>
#include <libethereum/All.h>
#include <liblll/All.h>
#include <libwhisper/WhisperPeer.h>
#include <libwhisper/WhisperHost.h>
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;
using namespace dev::shh;

#if 0
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
#elif 0
int main()
{
	KeyPair u = KeyPair::create();
	KeyPair cb = KeyPair::create();
	OverlayDB db;
	State s(cb.address(), db, BaseState::Empty);
	cnote << s.rootHash();
	s.addBalance(u.address(), 1 * ether);
	Address c = s.newContract(1000 * ether, compileLLL("(suicide (caller))"));
	s.commit();
	State before = s;
	cnote << "State before transaction: " << before;
	Transaction t(0, 10000, 10000, c, bytes(), 0, u.secret());
	cnote << "Transaction: " << t;
	cnote << s.balance(c);
	s.execute(LastHashes(), t.rlp());
	cnote << "State after transaction: " << s;
	cnote << before.diff(s);
}
#else
int main()
{
	cnote << KeyPair(Secret("0000000000000000000000000000000000000000000000000000000000000000")).address();
	cnote << KeyPair(Secret("1111111111111111111111111111111111111111111111111111111111111111")).address();
}
#endif

