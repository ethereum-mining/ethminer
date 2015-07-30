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
/** @file block.cpp
 * @author Christoph Jentzsch <cj@ethdev.com>
 * @date 2015
 * block test functions.
 */
#include "test/fuzzTesting/fuzzHelper.h"
#include <boost/filesystem.hpp>
#include <libdevcore/FileSystem.h>
#include <libdevcore/TransientDirectory.h>
#include <libethcore/Params.h>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/TransactionQueue.h>
#include <test/TestHelper.h>

using namespace std;
using namespace json_spirit;
using namespace dev;
using namespace dev::eth;

namespace dev {  namespace test {

typedef std::vector<bytes> uncleList;
typedef std::pair<bytes, uncleList> blockSet;

using BlockHeader = Ethash::BlockHeader;

BlockHeader constructBlock(mObject& _o, h256 const& _stateRoot = h256{});
bytes createBlockRLPFromFields(mObject& _tObj, h256 const&  _stateRoot = h256{});
RLPStream createFullBlockFromHeader(BlockHeader const& _bi, bytes const& _txs = RLPEmptyList, bytes const& _uncles = RLPEmptyList);

mArray writeTransactionsToJson(Transactions const& txs);
mObject writeBlockHeaderToJson(mObject& _o, BlockHeader const& _bi);
void overwriteBlockHeader(BlockHeader& _current_BlockHeader, mObject& _blObj, const BlockHeader& _parent);
void updatePoW(BlockHeader& _bi);
mArray importUncles(mObject const& _blObj, vector<BlockHeader>& _vBiUncles, vector<BlockHeader> const& _vBiBlocks, std::vector<blockSet> _blockSet);

void doBlockchainTests(json_spirit::mValue& _v, bool _fillin)
{
	for (auto& i: _v.get_obj())
	{
		mObject& o = i.second.get_obj();
		if (test::Options::get().singleTest && test::Options::get().singleTestName != i.first)
		{
			o.clear();
			continue;
		}

		cerr << i.first << endl;
		TBOOST_REQUIRE(o.count("genesisBlockHeader"));

		TBOOST_REQUIRE(o.count("pre"));
		ImportTest importer(o["pre"].get_obj());
		TransientDirectory td_stateDB_tmp;
		BlockHeader biGenesisBlock = constructBlock(o["genesisBlockHeader"].get_obj(), h256{});
		State trueState(OverlayDB(State::openDB(td_stateDB_tmp.path(), h256{}, WithExisting::Kill)), BaseState::Empty, biGenesisBlock.coinbaseAddress());

		//Imported blocks from the start
		std::vector<blockSet> blockSets;

		importer.importState(o["pre"].get_obj(), trueState);
		o["pre"] = fillJsonWithState(trueState);
		trueState.commit();

		if (_fillin)
			biGenesisBlock = constructBlock(o["genesisBlockHeader"].get_obj(), trueState.rootHash());
		else
			TBOOST_CHECK_MESSAGE((biGenesisBlock.stateRoot() == trueState.rootHash()), "root hash does not match");

		if (_fillin)
		{
			// find new valid nonce
			updatePoW(biGenesisBlock);

			//update genesis block in json file
			writeBlockHeaderToJson(o["genesisBlockHeader"].get_obj(), biGenesisBlock);
		}

		// create new "genesis" block
		RLPStream rlpGenesisBlock = createFullBlockFromHeader(biGenesisBlock);
		biGenesisBlock.verifyInternals(&rlpGenesisBlock.out());
		o["genesisRLP"] = toHex(rlpGenesisBlock.out(), 2, HexPrefix::Add);

		// construct true blockchain
		TransientDirectory td;
		FullBlockChain<Ethash> trueBc(rlpGenesisBlock.out(), StateDefinition(), td.path(), WithExisting::Kill);

		if (_fillin)
		{
			TBOOST_REQUIRE(o.count("blocks"));
			mArray blArray;

			blockSet genesis;
			genesis.first = rlpGenesisBlock.out();
			genesis.second = uncleList();
			blockSets.push_back(genesis);
			vector<BlockHeader> vBiBlocks;
			vBiBlocks.push_back(biGenesisBlock);

			size_t importBlockNumber = 0;
			for (auto const& bl: o["blocks"].get_array())
			{
				mObject blObj = bl.get_obj();
				if (blObj.count("blocknumber") > 0)
					importBlockNumber = std::max((int)toInt(blObj["blocknumber"]), 1);
				else
					importBlockNumber++;

				//each time construct a new blockchain up to importBlockNumber (to generate next block header)
				vBiBlocks.clear();
				vBiBlocks.push_back(biGenesisBlock);

				TransientDirectory td_stateDB, td_bc;
				FullBlockChain<Ethash> bc(rlpGenesisBlock.out(), StateDefinition(), td_bc.path(), WithExisting::Kill);
				State state(OverlayDB(State::openDB(td_stateDB.path(), h256{}, WithExisting::Kill)), BaseState::Empty);
				state.setAddress(biGenesisBlock.coinbaseAddress());
				importer.importState(o["pre"].get_obj(), state);
				state.commit();
				state.sync(bc);

				for (size_t i = 1; i < importBlockNumber; i++) //0 block is genesis
				{
					BlockQueue uncleQueue;
					uncleQueue.setChain(bc);
					uncleList uncles = blockSets.at(i).second;
					for (size_t j = 0; j < uncles.size(); j++)
						uncleQueue.import(&uncles.at(j), false);

					const bytes block = blockSets.at(i).first;
					bc.sync(uncleQueue, state.db(), 4);
					bc.attemptImport(block, state.db());
					vBiBlocks.push_back(BlockHeader(block));
					state.sync(bc);
				}

				// get txs
				TransactionQueue txs;
				ZeroGasPricer gp;
				TBOOST_REQUIRE(blObj.count("transactions"));
				for (auto const& txObj: blObj["transactions"].get_array())
				{
					mObject tx = txObj.get_obj();
					importer.importTransaction(tx);
					if (txs.import(importer.m_transaction.rlp()) != ImportResult::Success)
						cnote << "failed importing transaction\n";
				}

				//get uncles
				vector<BlockHeader> vBiUncles;
				blObj["uncleHeaders"] = importUncles(blObj, vBiUncles, vBiBlocks, blockSets);

				BlockQueue uncleBlockQueue;
				uncleBlockQueue.setChain(bc);
				uncleList uncleBlockQueueList;
				cnote << "import uncle in blockQueue";
				for (size_t i = 0; i < vBiUncles.size(); i++)
				{
					RLPStream uncle = createFullBlockFromHeader(vBiUncles.at(i));
					try
					{
						uncleBlockQueue.import(&uncle.out(), false);
						uncleBlockQueueList.push_back(uncle.out());
						// wait until block is verified
						this_thread::sleep_for(chrono::seconds(1));
					}
					catch(...)
					{
						cnote << "error in importing uncle! This produces an invalid block (May be by purpose for testing).";
					}
				} 
				bc.sync(uncleBlockQueue, state.db(), 4);
				state.commitToMine(bc);

				try
				{
					state.sync(bc);
					state.sync(bc, txs, gp);
					mine(state, bc);
				}
				catch (Exception const& _e)
				{
					cnote << "state sync or mining did throw an exception: " << diagnostic_information(_e);
					return;
				}
				catch (std::exception const& _e)
				{
					cnote << "state sync or mining did throw an exception: " << _e.what();
					return;
				}

				blObj["rlp"] = toHex(state.blockData(), 2, HexPrefix::Add);

				//get valid transactions
				Transactions txList;
				for (auto const& txi: txs.topTransactions(std::numeric_limits<unsigned>::max()))
					txList.push_back(txi);
				blObj["transactions"] = writeTransactionsToJson(txList);

				BlockHeader current_BlockHeader(state.blockData());

				RLPStream uncleStream;
				uncleStream.appendList(vBiUncles.size());
				for (unsigned i = 0; i < vBiUncles.size(); ++i)
				{
					RLPStream uncleRlp;
					vBiUncles[i].streamRLP(uncleRlp);
					uncleStream.appendRaw(uncleRlp.out());
				}

				if (blObj.count("blockHeader"))
					overwriteBlockHeader(current_BlockHeader, blObj, vBiBlocks[vBiBlocks.size()-1]);

				if (blObj.count("blockHeader") && blObj["blockHeader"].get_obj().count("bruncle"))
					current_BlockHeader.populateFromParent(vBiBlocks[vBiBlocks.size() -1]);

				if (vBiUncles.size())
				{
					// update unclehash in case of invalid uncles
					current_BlockHeader.setSha3Uncles(sha3(uncleStream.out()));
					updatePoW(current_BlockHeader);
				}

				// write block header
				mObject oBlockHeader;
				writeBlockHeaderToJson(oBlockHeader, current_BlockHeader);
				blObj["blockHeader"] = oBlockHeader;
				vBiBlocks.push_back(current_BlockHeader);

				// compare blocks from state and from rlp
				RLPStream txStream;
				txStream.appendList(txList.size());
				for (unsigned i = 0; i < txList.size(); ++i)
				{
					RLPStream txrlp;
					txList[i].streamRLP(txrlp);
					txStream.appendRaw(txrlp.out());
				}

				RLPStream block2 = createFullBlockFromHeader(current_BlockHeader, txStream.out(), uncleStream.out());

				blObj["rlp"] = toHex(block2.out(), 2, HexPrefix::Add);

				if (sha3(RLP(state.blockData())[0].data()) != sha3(RLP(block2.out())[0].data()))
				{
					cnote << "block header mismatch state.blockData() vs updated state.info()\n";
					cerr << toHex(state.blockData()) << "vs" << toHex(block2.out());
				}

				if (sha3(RLP(state.blockData())[1].data()) != sha3(RLP(block2.out())[1].data()))
					cnote << "txs mismatch\n";

				if (sha3(RLP(state.blockData())[2].data()) != sha3(RLP(block2.out())[2].data()))
					cnote << "uncle list mismatch\n" << RLP(state.blockData())[2].data() << "\n" << RLP(block2.out())[2].data();

				try
				{
					state.sync(bc);
					bc.import(block2.out(), state.db());
					state.sync(bc);
					state.commit();

					//there we get new blockchain status in state which could have more difficulty than we have in trueState
					//attempt to import new block to the true blockchain
					trueBc.sync(uncleBlockQueue, trueState.db(), 4);
					trueBc.attemptImport(block2.out(), trueState.db());
					trueState.sync(trueBc);

					blockSet newBlock;
					newBlock.first = block2.out();
					newBlock.second = uncleBlockQueueList;
					if (importBlockNumber < blockSets.size())
					{
						//make new correct history of imported blocks
						blockSets[importBlockNumber] = newBlock;
						for (size_t i = importBlockNumber + 1; i < blockSets.size(); i++)
							blockSets.pop_back();
					}
					else
						blockSets.push_back(newBlock);
				}
				// if exception is thrown, RLP is invalid and no blockHeader, Transaction list, or Uncle list should be given
				catch (...)
				{
					cnote << "block is invalid!\n";
					blObj.erase(blObj.find("blockHeader"));
					blObj.erase(blObj.find("uncleHeaders"));
					blObj.erase(blObj.find("transactions"));
				}
				blArray.push_back(blObj);
				this_thread::sleep_for(chrono::seconds(1));
			} //for blocks

			if (o.count("expect") > 0)
			{
				stateOptionsMap expectStateMap;
				State stateExpect(OverlayDB(), BaseState::Empty, biGenesisBlock.coinbaseAddress());
				importer.importState(o["expect"].get_obj(), stateExpect, expectStateMap);
				ImportTest::checkExpectedState(stateExpect, trueState, expectStateMap, Options::get().checkState ? WhenError::Throw : WhenError::DontThrow);
				o.erase(o.find("expect"));
			}

			o["blocks"] = blArray;
			o["postState"] = fillJsonWithState(trueState);
			o["lastblockhash"] = toString(trueBc.info().hash());

			//make all values hex in pre section
			State prestate(OverlayDB(), BaseState::Empty, biGenesisBlock.coinbaseAddress());
			importer.importState(o["pre"].get_obj(), prestate);
			o["pre"] = fillJsonWithState(prestate);
		}//_fillin

		else
		{
			for (auto const& bl: o["blocks"].get_array())
			{
				bool importedAndBest = true;
				mObject blObj = bl.get_obj();
				bytes blockRLP;
				try
				{
					blockRLP = importByteArray(blObj["rlp"].get_str());
					trueState.sync(trueBc);
					trueBc.import(blockRLP, trueState.db());
					if (trueBc.info() != BlockHeader(blockRLP))
						importedAndBest  = false;
					trueState.sync(trueBc);
				}
				// if exception is thrown, RLP is invalid and no blockHeader, Transaction list, or Uncle list should be given
				catch (Exception const& _e)
				{
					cnote << "state sync or block import did throw an exception: " << diagnostic_information(_e);
					TBOOST_CHECK((blObj.count("blockHeader") == 0));
					TBOOST_CHECK((blObj.count("transactions") == 0));
					TBOOST_CHECK((blObj.count("uncleHeaders") == 0));
					continue;
				}
				catch (std::exception const& _e)
				{
					cnote << "state sync or block import did throw an exception: " << _e.what();
					TBOOST_CHECK((blObj.count("blockHeader") == 0));
					TBOOST_CHECK((blObj.count("transactions") == 0));
					TBOOST_CHECK((blObj.count("uncleHeaders") == 0));
					continue;
				}
				catch (...)
				{
					cnote << "state sync or block import did throw an exception\n";
					TBOOST_CHECK((blObj.count("blockHeader") == 0));
					TBOOST_CHECK((blObj.count("transactions") == 0));
					TBOOST_CHECK((blObj.count("uncleHeaders") == 0));
					continue;
				}

				TBOOST_REQUIRE(blObj.count("blockHeader"));

				mObject tObj = blObj["blockHeader"].get_obj();
				BlockHeader blockHeaderFromFields;
				const bytes c_rlpBytesBlockHeader = createBlockRLPFromFields(tObj);
				const RLP c_blockHeaderRLP(c_rlpBytesBlockHeader);
				blockHeaderFromFields.populateFromHeader(c_blockHeaderRLP, IgnoreSeal);

				BlockHeader blockFromRlp(trueBc.header());

				if (importedAndBest)
				{
					//Check the fields restored from RLP to original fields
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.headerHash(WithProof) == blockFromRlp.headerHash(WithProof)), "hash in given RLP not matching the block hash!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.parentHash() == blockFromRlp.parentHash()), "parentHash in given RLP not matching the block parentHash!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.sha3Uncles() == blockFromRlp.sha3Uncles()), "sha3Uncles in given RLP not matching the block sha3Uncles!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.coinbaseAddress() == blockFromRlp.coinbaseAddress()),"coinbaseAddress in given RLP not matching the block coinbaseAddress!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.stateRoot() == blockFromRlp.stateRoot()), "stateRoot in given RLP not matching the block stateRoot!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.transactionsRoot() == blockFromRlp.transactionsRoot()), "transactionsRoot in given RLP not matching the block transactionsRoot!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.receiptsRoot() == blockFromRlp.receiptsRoot()), "receiptsRoot in given RLP not matching the block receiptsRoot!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.logBloom() == blockFromRlp.logBloom()), "logBloom in given RLP not matching the block logBloom!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.difficulty() == blockFromRlp.difficulty()), "difficulty in given RLP not matching the block difficulty!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.number() == blockFromRlp.number()), "number in given RLP not matching the block number!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.gasLimit() == blockFromRlp.gasLimit()),"gasLimit in given RLP not matching the block gasLimit!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.gasUsed() == blockFromRlp.gasUsed()), "gasUsed in given RLP not matching the block gasUsed!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.timestamp() == blockFromRlp.timestamp()), "timestamp in given RLP not matching the block timestamp!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.extraData() == blockFromRlp.extraData()), "extraData in given RLP not matching the block extraData!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.mixHash() == blockFromRlp.mixHash()), "mixHash in given RLP not matching the block mixHash!");
					TBOOST_CHECK_MESSAGE((blockHeaderFromFields.nonce() == blockFromRlp.nonce()), "nonce in given RLP not matching the block nonce!");

					TBOOST_CHECK_MESSAGE((blockHeaderFromFields == blockFromRlp), "However, blockHeaderFromFields != blockFromRlp!");

					//Check transaction list

					Transactions txsFromField;

					for (auto const& txObj: blObj["transactions"].get_array())
					{
						mObject tx = txObj.get_obj();

						TBOOST_REQUIRE(tx.count("nonce"));
						TBOOST_REQUIRE(tx.count("gasPrice"));
						TBOOST_REQUIRE(tx.count("gasLimit"));
						TBOOST_REQUIRE(tx.count("to"));
						TBOOST_REQUIRE(tx.count("value"));
						TBOOST_REQUIRE(tx.count("v"));
						TBOOST_REQUIRE(tx.count("r"));
						TBOOST_REQUIRE(tx.count("s"));
						TBOOST_REQUIRE(tx.count("data"));

						try
						{
							Transaction t(createRLPStreamFromTransactionFields(tx).out(), CheckTransaction::Everything);
							txsFromField.push_back(t);
						}
						catch (Exception const& _e)
						{
							TBOOST_ERROR("Failed transaction constructor with Exception: " << diagnostic_information(_e));
						}
						catch (exception const& _e)
						{
							cnote << _e.what();
						}
					}

					Transactions txsFromRlp;
					RLP root(blockRLP);
					for (auto const& tr: root[1])
					{
						Transaction tx(tr.data(), CheckTransaction::Everything);
						txsFromRlp.push_back(tx);
					}

					TBOOST_CHECK_MESSAGE((txsFromRlp.size() == txsFromField.size()), "transaction list size does not match");

					for (size_t i = 0; i < txsFromField.size(); ++i)
					{
						TBOOST_CHECK_MESSAGE((txsFromField[i].data() == txsFromRlp[i].data()), "transaction data in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].gas() == txsFromRlp[i].gas()), "transaction gasLimit in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].gasPrice() == txsFromRlp[i].gasPrice()), "transaction gasPrice in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].nonce() == txsFromRlp[i].nonce()), "transaction nonce in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].signature().r == txsFromRlp[i].signature().r), "transaction r in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].signature().s == txsFromRlp[i].signature().s), "transaction s in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].signature().v == txsFromRlp[i].signature().v), "transaction v in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].receiveAddress() == txsFromRlp[i].receiveAddress()), "transaction receiveAddress in rlp and in field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].value() == txsFromRlp[i].value()), "transaction receiveAddress in rlp and in field do not match");

						TBOOST_CHECK_MESSAGE((txsFromField[i] == txsFromRlp[i]), "transactions from  rlp and transaction from field do not match");
						TBOOST_CHECK_MESSAGE((txsFromField[i].rlp() == txsFromRlp[i].rlp()), "transactions rlp do not match");
					}

					// check uncle list

					// uncles from uncle list field
					vector<BlockHeader> uBlHsFromField;
					if (blObj["uncleHeaders"].type() != json_spirit::null_type)
						for (auto const& uBlHeaderObj: blObj["uncleHeaders"].get_array())
						{
							mObject uBlH = uBlHeaderObj.get_obj();
							TBOOST_REQUIRE((uBlH.size() == 16));
							bytes uncleRLP = createBlockRLPFromFields(uBlH);
							const RLP c_uRLP(uncleRLP);
							BlockHeader uncleBlockHeader;
							try
							{
								uncleBlockHeader.populateFromHeader(c_uRLP);
							}
							catch(...)
							{
								TBOOST_ERROR("invalid uncle header");
							}
							uBlHsFromField.push_back(uncleBlockHeader);
						}

					// uncles from block RLP
					vector<BlockHeader> uBlHsFromRlp;
					for	(auto const& uRLP: root[2])
					{
						BlockHeader uBl;
						uBl.populateFromHeader(uRLP);
						uBlHsFromRlp.push_back(uBl);
					}

					TBOOST_REQUIRE_EQUAL(uBlHsFromField.size(), uBlHsFromRlp.size());

					for (size_t i = 0; i < uBlHsFromField.size(); ++i)
						TBOOST_CHECK_MESSAGE((uBlHsFromField[i] == uBlHsFromRlp[i]), "block header in rlp and in field do not match");
				}//importedAndBest
			}//all blocks

			TBOOST_REQUIRE((o.count("lastblockhash") > 0));
			TBOOST_CHECK_MESSAGE((toString(trueBc.info().hash()) == o["lastblockhash"].get_str()),
					"Boost check: " + i.first + " lastblockhash does not match " + toString(trueBc.info().hash()) + " expected: " + o["lastblockhash"].get_str());
		}
	}
}

// helping functions

mArray importUncles(mObject const& _blObj, vector<BlockHeader>& _vBiUncles, vector<BlockHeader> const& _vBiBlocks, std::vector<blockSet> _blockSet)
{
	// write uncle list
	mArray aUncleList;
	mObject uncleHeaderObj_pre;

	for (auto const& uHObj: _blObj.at("uncleHeaders").get_array())
	{
		mObject uncleHeaderObj = uHObj.get_obj();
		if (uncleHeaderObj.count("sameAsPreviousSibling"))
		{
			writeBlockHeaderToJson(uncleHeaderObj_pre, _vBiUncles[_vBiUncles.size()-1]);
			aUncleList.push_back(uncleHeaderObj_pre);
			_vBiUncles.push_back(_vBiUncles[_vBiUncles.size()-1]);
			uncleHeaderObj_pre = uncleHeaderObj;
			continue;
		}

		if (uncleHeaderObj.count("sameAsBlock"))
		{
			size_t number = (size_t)toInt(uncleHeaderObj["sameAsBlock"]);
			uncleHeaderObj.erase("sameAsBlock");
			BlockHeader currentUncle = _vBiBlocks[number];
			writeBlockHeaderToJson(uncleHeaderObj, currentUncle);
			aUncleList.push_back(uncleHeaderObj);
			_vBiUncles.push_back(currentUncle);
			uncleHeaderObj_pre = uncleHeaderObj;
			continue;
		}

		if (uncleHeaderObj.count("sameAsPreviousBlockUncle"))
		{
			bytes uncleRLP = _blockSet[(size_t)toInt(uncleHeaderObj["sameAsPreviousBlockUncle"])].second[0];
			BlockHeader uncleHeader(uncleRLP);
			writeBlockHeaderToJson(uncleHeaderObj, uncleHeader);
			aUncleList.push_back(uncleHeaderObj);

			_vBiUncles.push_back(uncleHeader);
			uncleHeaderObj_pre = uncleHeaderObj;
			continue;
		}

		string overwrite = "false";
		if (uncleHeaderObj.count("overwriteAndRedoPoW"))
		{
			overwrite = uncleHeaderObj["overwriteAndRedoPoW"].get_str();
			uncleHeaderObj.erase("overwriteAndRedoPoW");
		}

		BlockHeader uncleBlockFromFields = constructBlock(uncleHeaderObj);

		// make uncle header valid
		uncleBlockFromFields.setTimestamp((u256)time(0));
		cnote << "uncle block n = " << toString(uncleBlockFromFields.number());
		if (_vBiBlocks.size() > 2)
		{
			if (uncleBlockFromFields.number() - 1 < _vBiBlocks.size())
				uncleBlockFromFields.populateFromParent(_vBiBlocks[(size_t)uncleBlockFromFields.number() - 1]);
			else
				uncleBlockFromFields.populateFromParent(_vBiBlocks[_vBiBlocks.size() - 2]);
		}
		else
			continue;

		if (overwrite != "false")
		{
			uncleBlockFromFields = constructHeader(
				overwrite == "parentHash" ? h256(uncleHeaderObj["parentHash"].get_str()) : uncleBlockFromFields.parentHash(),
				uncleBlockFromFields.sha3Uncles(),
				uncleBlockFromFields.coinbaseAddress(),
				overwrite == "stateRoot" ? h256(uncleHeaderObj["stateRoot"].get_str()) : uncleBlockFromFields.stateRoot(),
				uncleBlockFromFields.transactionsRoot(),
				uncleBlockFromFields.receiptsRoot(),
				uncleBlockFromFields.logBloom(),
				overwrite == "difficulty" ? toInt(uncleHeaderObj["difficulty"]) : overwrite == "timestamp" ? uncleBlockFromFields.calculateDifficulty(_vBiBlocks[(size_t)uncleBlockFromFields.number() - 1]) : uncleBlockFromFields.difficulty(),
				uncleBlockFromFields.number(),
				overwrite == "gasLimit" ? toInt(uncleHeaderObj["gasLimit"]) : uncleBlockFromFields.gasLimit(),
				overwrite == "gasUsed" ? toInt(uncleHeaderObj["gasUsed"]) : uncleBlockFromFields.gasUsed(),
				overwrite == "timestamp" ? toInt(uncleHeaderObj["timestamp"]) : uncleBlockFromFields.timestamp(),
				uncleBlockFromFields.extraData());

			if (overwrite == "parentHashIsBlocksParent")
				uncleBlockFromFields.populateFromParent(_vBiBlocks[_vBiBlocks.size() - 1]);
		}

		updatePoW(uncleBlockFromFields);

		if (overwrite == "nonce")
			updateEthashSeal(uncleBlockFromFields, uncleBlockFromFields.mixHash(), Nonce(uncleHeaderObj["nonce"].get_str()));

		if (overwrite == "mixHash")
			updateEthashSeal(uncleBlockFromFields, h256(uncleHeaderObj["mixHash"].get_str()), uncleBlockFromFields.nonce());

		writeBlockHeaderToJson(uncleHeaderObj, uncleBlockFromFields);

		aUncleList.push_back(uncleHeaderObj);
		_vBiUncles.push_back(uncleBlockFromFields);

		uncleHeaderObj_pre = uncleHeaderObj;
	} //for _blObj["uncleHeaders"].get_array()

	return aUncleList;
}

bytes createBlockRLPFromFields(mObject& _tObj, h256 const& _stateRoot)
{
	RLPStream rlpStream;
	rlpStream.appendList(_tObj.count("hash") > 0 ? (_tObj.size() - 1) : _tObj.size());

	if (_tObj.count("parentHash"))
		rlpStream << importByteArray(_tObj["parentHash"].get_str());

	if (_tObj.count("uncleHash"))
		rlpStream << importByteArray(_tObj["uncleHash"].get_str());

	if (_tObj.count("coinbase"))
		rlpStream << importByteArray(_tObj["coinbase"].get_str());

	if (_stateRoot)
		rlpStream << _stateRoot;
	else if (_tObj.count("stateRoot"))
		rlpStream << importByteArray(_tObj["stateRoot"].get_str());

	if (_tObj.count("transactionsTrie"))
		rlpStream << importByteArray(_tObj["transactionsTrie"].get_str());

	if (_tObj.count("receiptTrie"))
		rlpStream << importByteArray(_tObj["receiptTrie"].get_str());

	if (_tObj.count("bloom"))
		rlpStream << importByteArray(_tObj["bloom"].get_str());

	if (_tObj.count("difficulty"))
		rlpStream << bigint(_tObj["difficulty"].get_str());

	if (_tObj.count("number"))
		rlpStream << bigint(_tObj["number"].get_str());

	if (_tObj.count("gasLimit"))
		rlpStream << bigint(_tObj["gasLimit"].get_str());

	if (_tObj.count("gasUsed"))
		rlpStream << bigint(_tObj["gasUsed"].get_str());

	if (_tObj.count("timestamp"))
		rlpStream << bigint(_tObj["timestamp"].get_str());

	if (_tObj.count("extraData"))
		rlpStream << fromHex(_tObj["extraData"].get_str());

	if (_tObj.count("mixHash"))
		rlpStream << importByteArray(_tObj["mixHash"].get_str());

	if (_tObj.count("nonce"))
		rlpStream << importByteArray(_tObj["nonce"].get_str());

	return rlpStream.out();
}

void overwriteBlockHeader(BlockHeader& _header, mObject& _blObj, BlockHeader const& _parent)
{
	auto ho = _blObj["blockHeader"].get_obj();
	if (ho.size() != 14)
	{
		BlockHeader tmp = constructHeader(
			ho.count("parentHash") ? h256(ho["parentHash"].get_str()) : _header.parentHash(),
			ho.count("uncleHash") ? h256(ho["uncleHash"].get_str()) : _header.sha3Uncles(),
			ho.count("coinbase") ? Address(ho["coinbase"].get_str()) : _header.coinbaseAddress(),
			ho.count("stateRoot") ? h256(ho["stateRoot"].get_str()): _header.stateRoot(),
			ho.count("transactionsTrie") ? h256(ho["transactionsTrie"].get_str()) : _header.transactionsRoot(),
			ho.count("receiptTrie") ? h256(ho["receiptTrie"].get_str()) : _header.receiptsRoot(),
			ho.count("bloom") ? LogBloom(ho["bloom"].get_str()) : _header.logBloom(),
			ho.count("difficulty") ? toInt(ho["difficulty"]) : _header.difficulty(),
			ho.count("number") ? toInt(ho["number"]) : _header.number(),
			ho.count("gasLimit") ? toInt(ho["gasLimit"]) : _header.gasLimit(),
			ho.count("gasUsed") ? toInt(ho["gasUsed"]) : _header.gasUsed(),
			ho.count("timestamp") ? toInt(ho["timestamp"]) : _header.timestamp(),
			ho.count("extraData") ? importByteArray(ho["extraData"].get_str()) : _header.extraData());

		if (ho.count("RelTimestamp"))
		{
			tmp.setTimestamp(toInt(ho["RelTimestamp"]) +_parent.timestamp());
			tmp.setDifficulty(tmp.calculateDifficulty(_parent));
			this_thread::sleep_for(chrono::seconds((int)toInt(ho["RelTimestamp"])));
		}

		// find new valid nonce
		if (static_cast<BlockInfo>(tmp) != static_cast<BlockInfo>(_header) && tmp.difficulty())
			mine(tmp);


		if (ho.count("mixHash"))
			updateEthashSeal(tmp, h256(ho["mixHash"].get_str()), tmp.nonce());
		if (ho.count("nonce"))
			updateEthashSeal(tmp, tmp.mixHash(), Nonce(ho["nonce"].get_str()));

		tmp.noteDirty();
		_header = tmp;
	}
	else
	{
		// take the blockheader as is
		const bytes c_blockRLP = createBlockRLPFromFields(ho);
		const RLP c_bRLP(c_blockRLP);
		_header.populateFromHeader(c_bRLP, IgnoreSeal);
	}
}

BlockHeader constructBlock(mObject& _o, h256 const& _stateRoot)
{
	BlockHeader ret;
	try
	{
		// construct genesis block
		const bytes c_blockRLP = createBlockRLPFromFields(_o, _stateRoot);
		const RLP c_bRLP(c_blockRLP);
		ret.populateFromHeader(c_bRLP, IgnoreSeal);
	}
	catch (Exception const& _e)
	{
		cnote << "block population did throw an exception: " << diagnostic_information(_e);
	}
	catch (std::exception const& _e)
	{
		TBOOST_ERROR("Failed block population with Exception: " << _e.what());
	}
	catch(...)
	{
		TBOOST_ERROR("block population did throw an unknown exception\n");
	}
	return ret;
}

void updatePoW(BlockHeader& _bi)
{
	mine(_bi);
	_bi.noteDirty();
}

mArray writeTransactionsToJson(Transactions const& txs)
{
	mArray txArray;
	for (auto const& txi: txs)
	{
		mObject txObject = fillJsonWithTransaction(txi);
		txArray.push_back(txObject);
	}
	return txArray;
}

mObject writeBlockHeaderToJson(mObject& _o, BlockHeader const& _bi)
{
	_o["parentHash"] = toString(_bi.parentHash());
	_o["uncleHash"] = toString(_bi.sha3Uncles());
	_o["coinbase"] = toString(_bi.coinbaseAddress());
	_o["stateRoot"] = toString(_bi.stateRoot());
	_o["transactionsTrie"] = toString(_bi.transactionsRoot());
	_o["receiptTrie"] = toString(_bi.receiptsRoot());
	_o["bloom"] = toString(_bi.logBloom());
	_o["difficulty"] = toCompactHex(_bi.difficulty(), HexPrefix::Add, 1);
	_o["number"] = toCompactHex(_bi.number(), HexPrefix::Add, 1);
	_o["gasLimit"] = toCompactHex(_bi.gasLimit(), HexPrefix::Add, 1);
	_o["gasUsed"] = toCompactHex(_bi.gasUsed(), HexPrefix::Add, 1);
	_o["timestamp"] = toCompactHex(_bi.timestamp(), HexPrefix::Add, 1);
	_o["extraData"] = toHex(_bi.extraData(), 2, HexPrefix::Add);
	_o["mixHash"] = toString(_bi.mixHash());
	_o["nonce"] = toString(_bi.nonce());
	_o["hash"] = toString(_bi.hash());
	return _o;
}

RLPStream createFullBlockFromHeader(BlockHeader const& _bi, bytes const& _txs, bytes const& _uncles)
{
	RLPStream rlpStream;
	_bi.streamRLP(rlpStream, WithProof);

	RLPStream ret(3);
	ret.appendRaw(rlpStream.out());
	ret.appendRaw(_txs);
	ret.appendRaw(_uncles);

	return ret;
}

} }// Namespace Close

BOOST_AUTO_TEST_SUITE(BlockChainTests)

BOOST_AUTO_TEST_CASE(bcForkBlockTest)
{
	dev::test::executeTests("bcForkBlockTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcTotalDifficultyTest)
{
	dev::test::executeTests("bcTotalDifficultyTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcInvalidRLPTest)
{
	dev::test::executeTests("bcInvalidRLPTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcRPC_API_Test)
{
	dev::test::executeTests("bcRPC_API_Test", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcValidBlockTest)
{
	dev::test::executeTests("bcValidBlockTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcInvalidHeaderTest)
{
	dev::test::executeTests("bcInvalidHeaderTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcUncleTest)
{
	dev::test::executeTests("bcUncleTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcUncleHeaderValiditiy)
{
	dev::test::executeTests("bcUncleHeaderValiditiy", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcGasPricerTest)
{
	dev::test::executeTests("bcGasPricerTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

//BOOST_AUTO_TEST_CASE(bcBruncleTest)
//{
//	if (c_network != Network::Frontier)
//		dev::test::executeTests("bcBruncleTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
//}

BOOST_AUTO_TEST_CASE(bcBlockGasLimitTest)
{
	dev::test::executeTests("bcBlockGasLimitTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(bcWalletTest)
{
	if (test::Options::get().wallet)
		dev::test::executeTests("bcWalletTest", "/BlockchainTests",dev::test::getFolder(__FILE__) + "/BlockchainTestsFiller", dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_CASE(userDefinedFile)
{
	dev::test::userDefinedTest(dev::test::doBlockchainTests);
}

BOOST_AUTO_TEST_SUITE_END()
