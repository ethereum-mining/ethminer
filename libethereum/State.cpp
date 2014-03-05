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
/** @file State.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "State.h"

#include <secp256k1.h>
#include <boost/filesystem.hpp>
#include <time.h>
#include <random>
#include "BlockChain.h"
#include "Instruction.h"
#include "Exceptions.h"
#include "Dagger.h"
#include "Defaults.h"
#include "VM.h"
using namespace std;
using namespace eth;

u256 eth::c_genesisDifficulty = (u256)1 << 22;

std::map<Address, AddressState> const& eth::genesisState()
{
	static std::map<Address, AddressState> s_ret;
	if (s_ret.empty())
	{
		// Initialise.
		s_ret[Address(fromHex("8a40bfaa73256b60764c1bf40675a99083efb075"))] = AddressState(u256(1) << 200, 0, AddressType::Normal);
		s_ret[Address(fromHex("e6716f9544a56c530d868e4bfbacb172315bdead"))] = AddressState(u256(1) << 200, 0, AddressType::Normal);
		s_ret[Address(fromHex("1e12515ce3e0f817a4ddef9ca55788a1d66bd2df"))] = AddressState(u256(1) << 200, 0, AddressType::Normal);
		s_ret[Address(fromHex("1a26338f0d905e295fccb71fa9ea849ffa12aaf4"))] = AddressState(u256(1) << 200, 0, AddressType::Normal);
	}
	return s_ret;
}

Overlay State::openDB(std::string _path, bool _killExisting)
{
	if (_path.empty())
		_path = Defaults::get()->m_dbPath;
	boost::filesystem::create_directory(_path);

	if (_killExisting)
		boost::filesystem::remove_all(_path + "/state");

	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB* db = nullptr;
	ldb::DB::Open(o, _path + "/state", &db);
	return Overlay(db);
}

State::State(Address _coinbaseAddress, Overlay const& _db):
	m_db(_db),
	m_state(&m_db),
	m_ourAddress(_coinbaseAddress)
{
	m_blockReward = 1500 * finney;
	m_fees.setMultiplier(100 * szabo);

	secp256k1_start();

	// Initialise to the state entailed by the genesis block; this guarantees the trie is built correctly.
	m_state.init();
	eth::commit(genesisState(), m_db, m_state);
//	cnote << "State root: " << m_state.root();

	m_previousBlock = BlockInfo::genesis();
//	cnote << "Genesis hash:" << m_previousBlock.hash;
	resetCurrent();

	assert(m_state.root() == m_previousBlock.stateRoot);
}

State::State(State const& _s):
	m_db(_s.m_db),
	m_state(&m_db, _s.m_state.root()),
	m_transactions(_s.m_transactions),
	m_transactionSet(_s.m_transactionSet),
	m_cache(_s.m_cache),
	m_previousBlock(_s.m_previousBlock),
	m_currentBlock(_s.m_currentBlock),
	m_currentNumber(_s.m_currentNumber),
	m_ourAddress(_s.m_ourAddress),
	m_fees(_s.m_fees),
	m_blockReward(_s.m_blockReward)
{
}

State& State::operator=(State const& _s)
{
	m_db = _s.m_db;
	m_state.open(&m_db, _s.m_state.root());
	m_transactions = _s.m_transactions;
	m_transactionSet = _s.m_transactionSet;
	m_cache = _s.m_cache;
	m_previousBlock = _s.m_previousBlock;
	m_currentBlock = _s.m_currentBlock;
	m_currentNumber = _s.m_currentNumber;
	m_ourAddress = _s.m_ourAddress;
	m_fees = _s.m_fees;
	m_blockReward = _s.m_blockReward;
	return *this;
}

void State::ensureCached(Address _a, bool _requireMemory, bool _forceCreate) const
{
	auto it = m_cache.find(_a);
	if (it == m_cache.end())
	{
		// populate basic info.
		string stateBack = m_state.at(_a);
		if (stateBack.empty() && !_forceCreate)
			return;
		RLP state(stateBack);
		AddressState s;
		if (state.isNull())
			s = AddressState(0, 0);
		else if (state.itemCount() == 2)
			s = AddressState(state[0].toInt<u256>(), state[1].toInt<u256>());
		else
			s = AddressState(state[0].toInt<u256>(), state[1].toInt<u256>(), state[2].toHash<h256>());
		bool ok;
		tie(it, ok) = m_cache.insert(make_pair(_a, s));
	}
	if (_requireMemory && !it->second.haveMemory())
	{
		// Populate memory.
		assert(it->second.type() == AddressType::Contract);
		TrieDB<h256, Overlay> memdb(const_cast<Overlay*>(&m_db), it->second.oldRoot());		// promise we won't alter the overlay! :)
		map<u256, u256>& mem = it->second.setHaveMemory();
		for (auto const& i: memdb)
#ifdef __clang__
			if (mem.find(i.first) == mem.end())
				mem.insert(make_pair(i.first, RLP(i.second).toInt<u256>()));
			else
				mem.at(i.first) = RLP(i.second).toInt<u256>();
#else
			mem[i.first] = RLP(i.second).toInt<u256>();
#endif
	}
}

void State::commit()
{
	eth::commit(m_cache, m_db, m_state);
	m_cache.clear();
}

bool State::sync(BlockChain const& _bc)
{
return sync(_bc, _bc.currentHash());
}

bool State::sync(BlockChain const& _bc, h256 _block)
{
	bool ret = false;
	// BLOCK
	BlockInfo bi;
	try
	{
		auto b = _bc.block(_block);
		bi.populate(b);
		bi.verifyInternals(_bc.block(_block));
	}
	catch (...)
	{
		// TODO: Slightly nicer handling? :-)
		cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
		exit(1);
	}

	if (bi == m_currentBlock)
	{
		// We mined the last block.
		// Our state is good - we just need to move on to next.
		m_previousBlock = m_currentBlock;
		resetCurrent();
		m_currentNumber++;
		ret = true;
	}
	else if (bi == m_previousBlock)
	{
		// No change since last sync.
		// Carry on as we were.
	}
	else
	{
		// New blocks available, or we've switched to a different branch. All change.
		// Find most recent state dump and replay what's left.
		// (Most recent state dump might end up being genesis.)

		std::vector<h256> chain;
		while (bi.stateRoot != BlockInfo::genesis().hash && m_db.lookup(bi.stateRoot).empty())	// while we don't have the state root of the latest block...
		{
			chain.push_back(bi.hash);				// push back for later replay.
			bi.populate(_bc.block(bi.parentHash));	// move to parent.
		}

		m_previousBlock = bi;
		resetCurrent();

		// Iterate through in reverse, playing back each of the blocks.
		for (auto it = chain.rbegin(); it != chain.rend(); ++it)
			playback(_bc.block(*it), true);

		m_currentNumber = _bc.details(_block).number + 1;
		resetCurrent();
		ret = true;
	}
	return ret;
}

map<Address, u256> State::addresses() const
{
	map<Address, u256> ret;
	for (auto i: m_cache)
		if (i.second.type() != AddressType::Dead)
			ret[i.first] = i.second.balance();
	for (auto const& i: m_state)
		if (m_cache.find(i.first) == m_cache.end())
			ret[i.first] = RLP(i.second)[0].toInt<u256>();
	return ret;
}

void State::resetCurrent()
{
	m_transactions.clear();
	m_transactionSet.clear();
	m_cache.clear();
	m_currentBlock = BlockInfo();
	m_currentBlock.coinbaseAddress = m_ourAddress;
	m_currentBlock.stateRoot = m_previousBlock.stateRoot;
	m_currentBlock.parentHash = m_previousBlock.hash;
	m_currentBlock.sha3Transactions = h256();
	m_currentBlock.sha3Uncles = h256();

	// Update timestamp according to clock.
	m_currentBlock.timestamp = time(0);

	m_state.setRoot(m_currentBlock.stateRoot);
}

bool State::cull(TransactionQueue& _tq) const
{
	bool ret = false;
	auto ts = _tq.transactions();
	for (auto const& i: ts)
	{
		if (!m_transactionSet.count(i.first))
		{
			try
			{
				Transaction t(i.second);
				if (t.nonce <= transactionsFrom(t.sender()))
				{
					_tq.drop(i.first);
					ret = true;
				}
			}
			catch (...)
			{
				_tq.drop(i.first);
				ret = true;
			}
		}
	}
	return ret;
}

bool State::sync(TransactionQueue& _tq)
{
	// TRANSACTIONS
	bool ret = false;
	auto ts = _tq.transactions();
	vector<pair<h256, bytes>> futures;

	for (int goodTxs = 1; goodTxs;)
	{
		goodTxs = 0;
		for (auto const& i: ts)
		{
			if (!m_transactionSet.count(i.first))
			{
				// don't have it yet! Execute it now.
				try
				{
					execute(i.second);
					ret = true;
					_tq.noteGood(i);
					++goodTxs;
				}
				catch (InvalidNonce const& in)
				{
					if (in.required > in.candidate)
					{
						// too old
						_tq.drop(i.first);
						ret = true;
					}
					else
						_tq.setFuture(i);
				}
				catch (std::exception const&)
				{
					// Something else went wrong - drop it.
					_tq.drop(i.first);
					ret = true;
				}
			}
		}
	}
	return ret;
}

u256 State::playback(bytesConstRef _block, bool _fullCommit)
{
	try
	{
		m_currentBlock.populate(_block);
		m_currentBlock.verifyInternals(_block);
		return playback(_block, BlockInfo(), _fullCommit);
	}
	catch (...)
	{
		// TODO: Slightly nicer handling? :-)
		cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
		exit(1);
	}
}

u256 State::playback(bytesConstRef _block, BlockInfo const& _bi, BlockInfo const& _parent, BlockInfo const& _grandParent, bool _fullCommit)
{
	m_currentBlock = _bi;
	m_previousBlock = _parent;
	return playback(_block, _grandParent, _fullCommit);
}

u256 State::playback(bytesConstRef _block, BlockInfo const& _grandParent, bool _fullCommit)
{
	if (m_currentBlock.parentHash != m_previousBlock.hash)
		throw InvalidParentHash();

//	cnote << "playback begins:" << m_state.root();
//	cnote << m_state;

	// All ok with the block generally. Play back the transactions now...
	for (auto const& i: RLP(_block)[1])
		execute(i.data());

	// Initialise total difficulty calculation.
	u256 tdIncrease = m_currentBlock.difficulty;

	// Check uncles & apply their rewards to state.
	Addresses rewarded;
	for (auto const& i: RLP(_block)[2])
	{
		BlockInfo uncle = BlockInfo::fromHeader(i.data());
		if (m_previousBlock.parentHash != uncle.parentHash)
			throw InvalidUncle();
		if (_grandParent)
			uncle.verifyParent(_grandParent);
		tdIncrease += uncle.difficulty;
		rewarded.push_back(uncle.coinbaseAddress);
	}
	applyRewards(rewarded);

	// Commit all cached state changes to the state trie.
	commit();

	// Hash the state trie and check against the state_root hash in m_currentBlock.
	if (m_currentBlock.stateRoot != rootHash())
	{
		cwarn << "Bad state root!";
		cnote << "Given to be:" << m_currentBlock.stateRoot;
		cnote << TrieDB<Address, Overlay>(&m_db, m_currentBlock.stateRoot);
		cnote << "Calculated to be:" << rootHash();
		cnote << m_state;
		cnote << *this;
		// Rollback the trie.
		m_db.rollback();
		throw InvalidStateRoot();
	}

	if (_fullCommit)
	{
		// Commit the new trie to disk.
		m_db.commit();

		m_previousBlock = m_currentBlock;
		resetCurrent();
	}
	else
	{
		m_db.rollback();
		resetCurrent();
	}

	return tdIncrease;
}

// @returns the block that represents the difference between m_previousBlock and m_currentBlock.
// (i.e. all the transactions we executed).
void State::commitToMine(BlockChain const& _bc)
{
	if (m_currentBlock.sha3Transactions != h256() || m_currentBlock.sha3Uncles != h256())
	{
		Addresses uncleAddresses;
		for (auto i: RLP(m_currentUncles))
			uncleAddresses.push_back(i[2].toHash<Address>());
		unapplyRewards(uncleAddresses);
	}

	cnote << "Commiting to mine on" << m_previousBlock.hash;

	RLPStream uncles;
	Addresses uncleAddresses;

	if (m_previousBlock != BlockInfo::genesis())
	{
		// Find uncles if we're not a direct child of the genesis.
//		cout << "Checking " << m_previousBlock.hash << ", parent=" << m_previousBlock.parentHash << endl;
		auto us = _bc.details(m_previousBlock.parentHash).children;
		assert(us.size() >= 1);	// must be at least 1 child of our grandparent - it's our own parent!
		uncles.appendList(us.size() - 1);	// one fewer - uncles precludes our parent from the list of grandparent's children.
		for (auto const& u: us)
			if (u != m_previousBlock.hash)	// ignore our own parent - it's not an uncle.
			{
				BlockInfo ubi(_bc.block(u));
				ubi.fillStream(uncles, true);
				uncleAddresses.push_back(ubi.coinbaseAddress);
			}
	}
	else
		uncles.appendList(0);

	applyRewards(uncleAddresses);

	RLPStream txs(m_transactions.size());
	for (auto const& i: m_transactions)
		i.fillStream(txs);

	txs.swapOut(m_currentTxs);
	uncles.swapOut(m_currentUncles);

	m_currentBlock.sha3Transactions = sha3(m_currentTxs);
	m_currentBlock.sha3Uncles = sha3(m_currentUncles);

	// Commit any and all changes to the trie that are in the cache, then update the state root accordingly.
	commit();

	cnote << "stateRoot:" << m_state.root();
//	cnote << m_state;
//	cnote << *this;

	m_currentBlock.stateRoot = m_state.root();
	m_currentBlock.parentHash = m_previousBlock.hash;
}

MineInfo State::mine(uint _msTimeout)
{
	// Update difficulty according to timestamp.
	m_currentBlock.difficulty = m_currentBlock.calculateDifficulty(m_previousBlock);

	// TODO: Miner class that keeps dagger between mine calls (or just non-polling mining).
	MineInfo ret = m_dagger.mine(/*out*/m_currentBlock.nonce, m_currentBlock.headerHashWithoutNonce(), m_currentBlock.difficulty, _msTimeout);
	if (ret.completed)
	{
		// Got it!

		// Commit to disk.
		m_db.commit();

		// Compile block:
		RLPStream ret;
		ret.appendList(3);
		m_currentBlock.fillStream(ret, true);
		ret.appendRaw(m_currentTxs);
		ret.appendRaw(m_currentUncles);
		ret.swapOut(m_currentBytes);
		m_currentBlock.hash = sha3(m_currentBytes);
		cnote << "Mined " << m_currentBlock.hash << "(parent: " << m_currentBlock.parentHash << ")";
	}
	else
		m_currentBytes.clear();

	return ret;
}

bool State::isNormalAddress(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return it->second.type() == AddressType::Normal;
}

bool State::isContractAddress(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return it->second.type() == AddressType::Contract;
}

u256 State::balance(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return 0;
	return it->second.balance();
}

void State::noteSending(Address _id)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		m_cache[_id] = AddressState(0, 1);
	else
		it->second.incNonce();
}

void State::addBalance(Address _id, u256 _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		m_cache[_id] = AddressState(_amount, 0);
	else
		it->second.addBalance(_amount);
}

void State::subBalance(Address _id, bigint _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end() || (bigint)it->second.balance() < _amount)
		throw NotEnoughCash();
	else
		it->second.addBalance(-_amount);
}

u256 State::transactionsFrom(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return 0;
	else
		return it->second.nonce();
}

u256 State::contractMemory(Address _id, u256 _memory) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end() || it->second.type() != AddressType::Contract)
		return 0;
	else if (it->second.haveMemory())
	{
		auto mit = it->second.memory().find(_memory);
		if (mit == it->second.memory().end())
			return 0;
		return mit->second;
	}
	// Memory not cached - just grab one item from the DB rather than cache the lot.
	TrieDB<h256, Overlay> memdb(const_cast<Overlay*>(&m_db), it->second.oldRoot());			// promise we won't change the overlay! :)
	string ret = memdb.at(_memory);
	return ret.size() ? RLP(ret).toInt<u256>() : 0;
}

map<u256, u256> const& State::contractMemory(Address _contract) const
{
	if (!isContractAddress(_contract))
		return EmptyMapU256U256;
	ensureCached(_contract, true, true);
	return m_cache[_contract].memory();
}

void State::execute(bytesConstRef _rlp)
{
	// Entry point for a user-executed transaction.
	Transaction t(_rlp);
	executeBare(t, t.sender());

	// Add to the user-originated transactions that we've executed.
	// NOTE: Here, contract-originated transactions will not get added to the transaction list.
	// If this is wrong, move this line into execute(Transaction const& _t, Address _sender) and
	// don't forget to allow unsigned transactions in the tx list if they concur with the script execution.
	m_transactions.push_back(t);
	m_transactionSet.insert(t.sha3());
}

void State::applyRewards(Addresses const& _uncleAddresses)
{
	u256 r = m_blockReward;
	for (auto const& i: _uncleAddresses)
	{
		addBalance(i, m_blockReward * 3 / 4);
		r += m_blockReward / 8;
	}
	addBalance(m_currentBlock.coinbaseAddress, r);
}

void State::unapplyRewards(Addresses const& _uncleAddresses)
{
	u256 r = m_blockReward;
	for (auto const& i: _uncleAddresses)
	{
		subBalance(i, m_blockReward * 3 / 4);
		r += m_blockReward / 8;
	}
	subBalance(m_currentBlock.coinbaseAddress, r);
}

void State::executeBare(Transaction const& _t, Address _sender)
{
#if ETH_DEBUG
	commit();
	clog(StateChat) << "State:" << rootHash();
	clog(StateChat) << "Executing TX:" << _t;
#endif

	// Entry point for a contract-originated transaction.

	// Ignore invalid transactions.
	auto nonceReq = transactionsFrom(_sender);
	if (_t.nonce != nonceReq)
	{
		clog(StateChat) << "Invalid Nonce.";
		throw InvalidNonce(nonceReq, _t.nonce);
	}

	unsigned nonZeroData = 0;
	for (auto i: _t.data)
		if (i)
			nonZeroData++;
	u256 fee = _t.receiveAddress ? m_fees.m_txFee : (nonZeroData * m_fees.m_memoryFee + m_fees.m_newContractFee);

	// Not considered invalid - just pointless.
	if (balance(_sender) < _t.value + fee)
	{
		clog(StateChat) << "Not enough cash.";
		throw NotEnoughCash();
	}

	if (_t.receiveAddress)
	{
		// Increment associated nonce for sender.
		noteSending(_sender);

		// Pay...
		subBalance(_sender, _t.value + fee);
		addBalance(_t.receiveAddress, _t.value);

		if (isContractAddress(_t.receiveAddress))
		{
			// Once we get here, there's no going back.
			try
			{
				MinerFeeAdder feeAdder({this, 0});	// will add fee on destruction.
				execute(_t.receiveAddress, _sender, _t.value, _t.data, &feeAdder.fee);
			}
			catch (VMException const& _e)
			{
				clog(StateChat) << "VM Exception: " << _e.description();
			}
			catch (Exception const& _e)
			{
				clog(StateChat) << "Exception in VM: " << _e.description();
			}
			catch (std::exception const& _e)
			{
				clog(StateChat) << "std::exception in VM: " << _e.what();
			}
		}
	}
	else
	{
		Address newAddress = right160(_t.sha3());

		if (isContractAddress(newAddress) || isNormalAddress(newAddress))
		{
			clog(StateChat) << "Contract address collision.";
			throw ContractAddressCollision();
		}

		// Increment associated nonce for sender.
		noteSending(_sender);

		// Pay out of sender...
		subBalance(_sender, _t.value + fee);

		// Set up new account...
		m_cache[newAddress] = AddressState(_t.value, 0, AddressType::Contract);
		auto& mem = m_cache[newAddress].memory();
		for (uint i = 0; i < _t.data.size(); ++i)
#ifdef __clang__
			if (mem.find(i) == mem.end())
				mem.insert(make_pair(i, _t.data[i]));
			else
				mem.at(i) = _t.data[i];
#else
			mem[i] = _t.data[i];
#endif
	}

#if ETH_DEBUG
	commit();
	clog(StateChat) << "New state:" << rootHash();
#endif
}

void State::execute(Address _myAddress, Address _txSender, u256 _txValue, u256s const& _txData, u256* _totalFee)
{
	VM vm;
	ExtVM evm(*this, _myAddress, _txSender, _txValue, _txData);
	vm.go(evm);
	*_totalFee = vm.runFee();
}
