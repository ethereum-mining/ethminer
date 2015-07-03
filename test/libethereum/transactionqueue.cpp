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
/** @file transactionqueue.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 * TransactionQueue test functions.
 */

#include <libethereum/TransactionQueue.h>
#include "../TestHelper.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

BOOST_AUTO_TEST_SUITE(TransactionQueue)

BOOST_AUTO_TEST_CASE(maxNonce)
{

	dev::eth::TransactionQueue txq;

	// from a94f5374fce5edbc8e2a8697c15331677e6ebf0b
	const u256 gasCost =  10 * szabo;
	const u256 gas = 25000;
	Address dest = Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87");
	Address to = Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b");
	Secret sec = Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8");
	Transaction tx0(0, gasCost, gas, dest, bytes(), 0, sec );
	Transaction tx0_1(1, gasCost, gas, dest, bytes(), 0, sec );
	Transaction tx1(0, gasCost, gas, dest, bytes(), 1, sec );
	Transaction tx2(0, gasCost, gas, dest, bytes(), 2, sec );
	Transaction tx9(0, gasCost, gas, dest, bytes(), 9, sec );

	txq.import(tx0);
	BOOST_CHECK(1 == txq.maxNonce(to));
	txq.import(tx0);
	BOOST_CHECK(1 == txq.maxNonce(to));
	txq.import(tx0_1);
	BOOST_CHECK(1 == txq.maxNonce(to));
	txq.import(tx1);
	BOOST_CHECK(2 == txq.maxNonce(to));
	txq.import(tx9);
	BOOST_CHECK(10 == txq.maxNonce(to));
	txq.import(tx2);
	BOOST_CHECK(10 == txq.maxNonce(to));

}

BOOST_AUTO_TEST_SUITE_END()
