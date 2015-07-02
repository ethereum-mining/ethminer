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
/** @file transaction.cpp
 * @author Christoph Jentzsch <winsvega@mail.ru>
 * @date 2015
 * Transaaction test functions.
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
	Transaction tx0(0, 10 * szabo, 25000, Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"), bytes(), 0, Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8") );
	Transaction tx0_1(1, 10 * szabo, 25000, Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"), bytes(), 0, Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8") );
	Transaction tx1(0, 10 * szabo, 25000, Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"), bytes(), 1, Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8") );
	Transaction tx2(0, 10 * szabo, 25000, Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"), bytes(), 2, Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8") );
	Transaction tx9(0, 10 * szabo, 25000, Address("0x095e7baea6a6c7c4c2dfeb977efac326af552d87"), bytes(), 9, Secret("0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8") );

	txq.import(tx0);
	BOOST_CHECK_EQUAL(1, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
	txq.import(tx0);
	BOOST_CHECK_EQUAL(1, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
	txq.import(tx0_1);
	BOOST_CHECK_EQUAL(1, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
	txq.import(tx1);
	BOOST_CHECK_EQUAL(2, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
	txq.import(tx9);
	BOOST_CHECK_EQUAL(10, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
	txq.import(tx2);
	BOOST_CHECK_EQUAL(10, (unsigned)txq.maxNonce(Address("0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b")));
}

BOOST_AUTO_TEST_SUITE_END()
