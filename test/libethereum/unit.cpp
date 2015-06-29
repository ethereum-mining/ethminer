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
/** @file unit.cpp
 * @author Dimitry Khokhlov <Dimitry@ethdev.com>
 * @date 2015
 * libethereum unit test functions coverage.
 */

#include <boost/filesystem/operations.hpp>
#include <boost/test/unit_test.hpp>
#include <test/TestHelper.h>
#include "../JsonSpiritHeaders.h"

#include <libdevcore/TransientDirectory.h>

#include <libethereum/Defaults.h>
#include <libethereum/AccountDiff.h>
#include <libethereum/BlockChain.h>
#include <libethereum/BlockQueue.h>

using namespace dev;
using namespace eth;

BOOST_AUTO_TEST_SUITE(libethereum)

BOOST_AUTO_TEST_CASE(AccountDiff)
{
	dev::eth::AccountDiff accDiff;

	// Testing changeType
	// exist = true	   exist_from = true		AccountChange::Deletion
	accDiff.exist = dev::Diff<bool>(true, false);
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Deletion, "Account change type expected to be Deletion!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), "XXX") == 0, "Deletion lead expected to be 'XXX'!");

	// exist = true	   exist_from = false		AccountChange::Creation
	accDiff.exist = dev::Diff<bool>(false, true);
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Creation, "Account change type expected to be Creation!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), "+++") == 0, "Creation lead expected to be '+++'!");

	// exist = false	   bn = true	sc = true	AccountChange::All
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::All, "Account change type expected to be All!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), "***") == 0, "All lead expected to be '***'!");

	// exist = false	   bn = true	sc = false  AccountChange::Intrinsic
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("00"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::Intrinsic, "Account change type expected to be Intrinsic!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), " * ") == 0, "Intrinsic lead expected to be ' * '!");

	// exist = false	   bn = false   sc = true	AccountChange::CodeStorage
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 1);
	accDiff.balance = dev::Diff<dev::u256>(1, 1);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::CodeStorage, "Account change type expected to be CodeStorage!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), "* *") == 0, "CodeStorage lead expected to be '* *'!");

	// exist = false	   bn = false   sc = false	AccountChange::None
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 1);
	accDiff.balance = dev::Diff<dev::u256>(1, 1);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("00"));
	BOOST_CHECK_MESSAGE(accDiff.changeType() == dev::eth::AccountChange::None, "Account change type expected to be None!");
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead(accDiff.changeType()), "   ") == 0, "None lead expected to be '   '!");

	//ofstream
	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.balance = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	std::map<dev::u256, dev::Diff<dev::u256>> storage;
	storage[1] = accDiff.nonce;
	accDiff.storage = storage;
	std::stringstream buffer;

	//if (!_s.exist.to())
	buffer << accDiff;
	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "") == 0,	"Not expected output: '" + buffer.str() + "'");
	buffer.str(std::string());

	accDiff.exist = dev::Diff<bool>(false, true);
	buffer << accDiff;
	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "#2 (+1) 2 (+1) $[1] ([0]) \n *     0000000000000000000000000000000000000000000000000000000000000001: 2 (1)") == 0,	"Not expected output: '" + buffer.str() + "'");
	buffer.str(std::string());

	storage[1] = dev::Diff<dev::u256>(0, 0);
	accDiff.storage = storage;
	buffer << accDiff;
	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "#2 (+1) 2 (+1) $[1] ([0]) \n +     0000000000000000000000000000000000000000000000000000000000000001: 0") == 0,	"Not expected output: '" + buffer.str() + "'");
	buffer.str(std::string());

	storage[1] = dev::Diff<dev::u256>(1, 0);
	accDiff.storage = storage;
	buffer << accDiff;
	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "#2 (+1) 2 (+1) $[1] ([0]) \nXXX    0000000000000000000000000000000000000000000000000000000000000001 (1)") == 0,	"Not expected output: '" + buffer.str() + "'");
	buffer.str(std::string());

	BOOST_CHECK_MESSAGE(accDiff.changed() == true, "dev::eth::AccountDiff::changed(): incorrect return value");

	//unexpected value
	BOOST_CHECK_MESSAGE(strcmp(dev::eth::lead((dev::eth::AccountChange)123), "") != 0, "Not expected output when dev::eth::lead on unexpected value");
}

BOOST_AUTO_TEST_CASE(StateDiff)
{
	dev::eth::StateDiff stateDiff;
	dev::eth::AccountDiff accDiff;

	accDiff.exist = dev::Diff<bool>(false, false);
	accDiff.nonce = dev::Diff<dev::u256>(1, 2);
	accDiff.balance = dev::Diff<dev::u256>(1, 2);
	accDiff.code = dev::Diff<dev::bytes>(dev::fromHex("00"), dev::fromHex("01"));
	std::map<dev::u256, dev::Diff<dev::u256>> storage;
	storage[1] = accDiff.nonce;
	accDiff.storage = storage;
	std::stringstream buffer;

	dev::Address address("001122334455667788991011121314151617181920");
	stateDiff.accounts[address] = accDiff;
	buffer << stateDiff;

	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "1 accounts changed:\n***  0000000000000000000000000000000000000000: \n") == 0,	"Not expected output: '" + buffer.str() + "'");
}

BOOST_AUTO_TEST_CASE(BlockChain)
{
	std::string genesisRLP = "0xf901fcf901f7a00000000000000000000000000000000000000000000000000000000000000000a01dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347948888f1f195afa192cfee860698584c030f4c9db1a0cafd881ab193703b83816c49ff6c2bf6ba6f464a1be560c42106128c8dbc35e7a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421a056e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421b90100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008302000080832fefd8808454c98c8142a09eb47d85ccdf855f34cce66288b11d43a90d288dfc12eb7a900dcb7953a69a5a88448f2f62ce8e0392c0c0";
	dev::bytes genesisBlockRLP = dev::test::importByteArray(genesisRLP);
	BlockInfo biGenesisBlock(genesisBlockRLP);

	std::string blockRLP = "0xf90260f901f9a01f08d9b73350445d921aa74a44861b5cc12d1258a4da8ac3e0a41d1757aa8112a01dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347948888f1f195afa192cfee860698584c030f4c9db1a0ef1552a40b7165c3cd773806b9e0c165b75356e0314bf0706f279c729f51e017a0dff021d89bbd62a468fc16a27430f8fbf0cc9e41473f26601fa9e4bebd0660d5a0bc37d79753ad738a6dac4921e57392f145d8887476de3f783dfa7edae9283e52b90100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008302000001832fefd882520884558c98ed80a0f8e40e5173c7d73b2214e311c109fec63eb116b8ff90f78989e1a65082d6e74f8871f5f6541955ba68f861f85f800a82c35094095e7baea6a6c7c4c2dfeb977efac326af552d870a801ca0a1da1d695f9ff1595dcd950cd8f001bfb69e0ed6e731ad51353f52ed9c7117e3a0fe5f0ef2425069a91b2619d3147ab3a8386542be9b4c7ef738a0553d22361646c0";
	dev::bytes blockRLPbytes = dev::test::importByteArray(blockRLP);
	BlockInfo biBlock(blockRLPbytes);

	TransientDirectory td1, td2;
	State trueState(OverlayDB(State::openDB(td1.path())), BaseState::Empty, biGenesisBlock.coinbaseAddress);
	dev::eth::BlockChain trueBc(genesisBlockRLP, td2.path(), WithExisting::Verify);

	json_spirit::mObject o;
	json_spirit::mObject accountaddress;
	json_spirit::mObject account;
	account["balance"] = "0x02540be400";
	account["code"] = "0x";
	account["nonce"] = "0x00";
	account["storage"] = json_spirit::mObject();
	accountaddress["a94f5374fce5edbc8e2a8697c15331677e6ebf0b"] = account;
	o["pre"] = accountaddress;

	dev::test::ImportTest importer(o["pre"].get_obj());
	importer.importState(o["pre"].get_obj(), trueState);
	trueState.commit();

	trueBc.import(blockRLPbytes, trueState.db());

	std::stringstream buffer;
	buffer << trueBc;
	BOOST_CHECK_MESSAGE(strcmp(buffer.str().c_str(), "96d3fba448912a3f714221956ad5b6b7984236d4a1748c7f8ff5d637ca88e19f00:   1 @ 1f08d9b73350445d921aa74a44861b5cc12d1258a4da8ac3e0a41d1757aa8112\n") == 0,	"Not expected output: '" + buffer.str() + "'");

	trueBc.garbageCollect(true);

	//Block Queue Test
	BlockQueue bcQueue;
	bcQueue.tick(trueBc);
	QueueStatus bStatus = bcQueue.blockStatus(trueBc.info().hash());
	BOOST_CHECK(bStatus == QueueStatus::Unknown);

	dev::bytesConstRef bytesConst(&blockRLPbytes.at(0), blockRLPbytes.size());
	bcQueue.import(bytesConst, trueBc);
	bStatus = bcQueue.blockStatus(biBlock.hash());
	BOOST_CHECK(bStatus == QueueStatus::Unknown);

	bcQueue.clear();
}

BOOST_AUTO_TEST_CASE(BlockQueue)
{

}

BOOST_AUTO_TEST_SUITE_END()
