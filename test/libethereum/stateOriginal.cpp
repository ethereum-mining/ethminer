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
/** @file stateOriginal.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 * State test functions.
 */

#include <boost/test/unit_test.hpp>
#include <boost/filesystem/operations.hpp>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/Block.h>
#include <libethcore/Farm.h>
#include <libethcore/BasicAuthority.h>
#include <libethereum/Defaults.h>
#include <test/TestHelper.h>

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace test
{

int stateTest();

BOOST_AUTO_TEST_SUITE(StateIntegration)

BOOST_AUTO_TEST_CASE(Basic)
{
	Block s;
}

BOOST_AUTO_TEST_CASE(Complex)
{
	//Mining Changed on block branch
	//Review this test

	/*if (test::Options::get().nodag)
		return;

	cnote << "Testing State...";

	KeyPair me = Secret(sha3("Gav Wood"));
	KeyPair myMiner = Secret(sha3("Gav's Miner"));
//	KeyPair you = sha3("123");

	Defaults::setDBPath(boost::filesystem::temp_directory_path().string() + "/" + toString(chrono::system_clock::now().time_since_epoch().count()));

	OverlayDB stateDB = State::openDB(h256());
	CanonBlockChain<BasicAuthority> bc;
	cout << bc;

	Block s = bc.genesisBlock(stateDB);
	s.setBeneficiary(myMiner.address());
	cout << s;

	// Sync up - this won't do much until we use the last state.
	s.sync(bc);

	cout << s;

	// Mine to get some ether!
	mine(s, bc);

	bc.attemptImport(s.blockData(), stateDB);

	cout << bc;

	s.sync(bc);

	cout << s;

	// Inject a transaction to transfer funds from miner to me.
	Transaction t(1000, 10000, 100000, me.address(), bytes(), s.transactionsFrom(myMiner.address()), myMiner.secret());
	assert(t.sender() == myMiner.address());
	s.execute(bc.lastHashes(), t);

	cout << s;

	// Mine to get some ether and set in stone.
	s.commitToSeal(bc);
	s.commitToSeal(bc);
	mine(s, bc);
	bc.attemptImport(s.blockData(), stateDB);

	cout << bc;

	s.sync(bc);

	cout << s;*/
}

BOOST_AUTO_TEST_SUITE_END()

}
}
