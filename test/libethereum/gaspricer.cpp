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
/** @file gasPricer.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 * Gas pricer tests
 */

#include <libtestutils/BlockChainLoader.h>
#include <libethcore/Ethash.h>
#include <libethereum/BlockChain.h>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/GasPricer.h>
#include <libethereum/BasicGasPricer.h>
#include "../TestHelper.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev {  namespace test {

void executeGasPricerTest(string const& name, double _etherPrice, double _blockFee, string const& bcTestPath, TransactionPriority _txPrio, u256 _expectedAsk, u256 _expectedBid)
{
	cnote << name;
	BasicGasPricer gp(u256(double(ether / 1000) / _etherPrice), u256(_blockFee * 1000));

	Json::Value vJson = test::loadJsonFromFile(test::getTestPath() + bcTestPath);
	test::BlockChainLoader bcLoader(vJson[name]);
	BlockChain const& bc = bcLoader.bc();

	gp.update(bc);
	BOOST_CHECK_EQUAL(gp.ask(Block()), _expectedAsk);
	BOOST_CHECK_EQUAL(gp.bid(_txPrio), _expectedBid);
}
} }

BOOST_AUTO_TEST_SUITE(GasPricer)

BOOST_AUTO_TEST_CASE(trivialGasPricer)
{
	cnote << "trivialGasPricer";
	std::shared_ptr<dev::eth::GasPricer> gp(new TrivialGasPricer);
	BOOST_CHECK_EQUAL(gp->ask(Block()), c_defaultGasPrice);
	BOOST_CHECK_EQUAL(gp->bid(), c_defaultGasPrice);

	bytes bl = CanonBlockChain<Ethash>::createGenesisBlock();
	gp->update(FullBlockChain<Ethash>(bl, AccountMap(), TransientDirectory().path(), WithExisting::Kill));
	BOOST_CHECK_EQUAL(gp->ask(Block()), c_defaultGasPrice);
	BOOST_CHECK_EQUAL(gp->bid(), c_defaultGasPrice);
}

BOOST_AUTO_TEST_CASE(basicGasPricerNoUpdate)
{
	cnote << "basicGasPricer";
	BasicGasPricer gp(u256(double(ether / 1000) / 30.679), u256(15.0 * 1000));
	BOOST_CHECK_EQUAL(gp.ask(Block()), 155632494086);
	BOOST_CHECK_EQUAL(gp.bid(), 155632494086);

	gp.setRefPrice(u256(0));
	BOOST_CHECK_EQUAL(gp.ask(Block()), 0);
	BOOST_CHECK_EQUAL(gp.bid(), 0);

	gp.setRefPrice(u256(1));
	gp.setRefBlockFees(u256(0));
	BOOST_CHECK_EQUAL(gp.ask(Block()), 0);
	BOOST_CHECK_EQUAL(gp.bid(), 0);

	gp.setRefPrice(u256("0x100000000000000000000000000000000"));
	BOOST_CHECK_THROW(gp.setRefBlockFees(u256("0x100000000000000000000000000000000")), Overflow);
	BOOST_CHECK_EQUAL(gp.ask(Block()), 0);
	BOOST_CHECK_EQUAL(gp.bid(), 0);

	gp.setRefPrice(1);
	gp.setRefBlockFees(u256("0x100000000000000000000000000000000"));
	BOOST_CHECK_THROW(gp.setRefPrice(u256("0x100000000000000000000000000000000")), Overflow);
	BOOST_CHECK_EQUAL(gp.ask(Block()), u256("108315264019305646138446560671076"));
	BOOST_CHECK_EQUAL(gp.bid(), u256("108315264019305646138446560671076"));
}

BOOST_AUTO_TEST_CASE(basicGasPricer_RPC_API_Test)
{
	dev::test::executeGasPricerTest("RPC_API_Test", 30.679, 15.0, "/BlockchainTests/bcRPC_API_Test.json", TransactionPriority::Medium, 155632494086, 1);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_bcValidBlockTest)
{
	dev::test::executeGasPricerTest("SimpleTx", 30.679, 15.0, "/BlockchainTests/bcValidBlockTest.json", TransactionPriority::Medium, 155632494086, 10);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_bcUncleTest)
{
	dev::test::executeGasPricerTest("twoUncle", 30.679, 15.0, "/BlockchainTests/bcUncleTest.json", TransactionPriority::Medium, 155632494086, 1);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_bcUncleHeaderValiditiy)
{
	dev::test::executeGasPricerTest("correct", 30.679, 15.0, "/BlockchainTests/bcUncleHeaderValiditiy.json", TransactionPriority::Medium, 155632494086, 1);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_notxs)
{
	dev::test::executeGasPricerTest("notxs", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::Medium, 155632494086, 155632494086);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_highGasUsage_LowestPrio)
{
	dev::test::executeGasPricerTest("highGasUsage", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::Lowest, 15731282021, 10000000000000);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_highGasUsage_LowPrio)
{
	dev::test::executeGasPricerTest("highGasUsage", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::Low, 15731282021, 15734152261884);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_highGasUsage_MediumPrio)
{
	dev::test::executeGasPricerTest("highGasUsage", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::Medium, 15731282021, 20000000000000);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_highGasUsage_HighPrio)
{
	dev::test::executeGasPricerTest("highGasUsage", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::High, 15731282021, 24265847738115);
}

BOOST_AUTO_TEST_CASE(basicGasPricer_highGasUsage_HighestPrio)
{
	dev::test::executeGasPricerTest("highGasUsage", 30.679, 15.0, "/BlockchainTests/bcGasPricerTest.json", TransactionPriority::Highest, 15731282021, 30000000000000);
}
BOOST_AUTO_TEST_SUITE_END()
