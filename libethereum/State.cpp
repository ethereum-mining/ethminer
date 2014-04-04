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
	m_blockReward = _s.m_blockReward;
	return *this;
}

void State::ensureCached(Address _a, bool _requireMemory, bool _forceCreate) const
{
	ensureCached(m_cache, _a, _requireMemory, _forceCreate);
}

void State::ensureCached(std::map<Address, AddressState>& _cache, Address _a, bool _requireMemory, bool _forceCreate) const
{
	auto it = _cache.find(_a);
	if (it == _cache.end())
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
			s = AddressState(state[0].toInt<u256>(), state[1].toInt<u256>(), state[2].toHash<h256>(), state[3].toHash<h256>());
		bool ok;
		tie(it, ok) = _cache.insert(make_pair(_a, s));
	}
	if (_requireMemory && !it->second.isComplete())
	{
		// Populate memory.
		assert(it->second.type() == AddressType::Contract);
		TrieDB<h256, Overlay> memdb(const_cast<Overlay*>(&m_db), it->second.oldRoot());		// promise we won't alter the overlay! :)
		map<u256, u256>& mem = it->second.setIsComplete(bytesConstRef(m_db.lookup(it->second.codeHash())));
		for (auto const& i: memdb)
			mem[i.first] = RLP(i.second).toInt<u256>();
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
	// TODO: Check for uniqueness of uncles.
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

u256 State::contractStorage(Address _id, u256 _memory) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end() || it->second.type() != AddressType::Contract)
		return 0;
	else if (it->second.isComplete())
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

map<u256, u256> const& State::contractStorage(Address _contract) const
{
	if (!isContractAddress(_contract))
		return EmptyMapU256U256;
	ensureCached(_contract, true, true);
	return m_cache[_contract].memory();
}

bytes const& State::contractCode(Address _contract) const
{
	if (!isContractAddress(_contract))
		return EmptyBytes;
	ensureCached(_contract, true, true);
	return m_cache[_contract].code();
}

void State::execute(bytesConstRef _rlp)
{
	// Entry point for a user-executed transaction.
	Transaction t(_rlp);

	auto sender = t.sender();

	// Avoid invalid transactions.
	auto nonceReq = transactionsFrom(sender);
	if (t.nonce != nonceReq)
	{
		clog(StateChat) << "Invalid Nonce.";
		throw InvalidNonce(nonceReq, t.nonce);
	}

	// Don't like transactions whose gas price is too low. NOTE: this won't stay here forever - it's just until we get a proper gas proce discovery protocol going.
	if (t.gasPrice < 10 * szabo)
	{
		clog(StateChat) << "Offered gas-price is too low.";
		throw GasPriceTooLow();
	}

	// Check gas cost is enough.
	u256 gasCost;
	if (t.isCreation())
		gasCost = (t.init.size() + t.data.size()) * c_txDataGas + c_createGas;
	else
		gasCost = t.data.size() * c_txDataGas + c_callGas;

	if (t.gas < gasCost)
	{
		clog(StateChat) << "Not enough gas to pay for the transaction.";
		throw OutOfGas();
	}

	u256 cost = t.value + t.gas * t.gasPrice;

	// Avoid unaffordable transactions.
	if (balance(sender) < cost)
	{
		clog(StateChat) << "Not enough cash.";
		throw NotEnoughCash();
	}

	u256 gas = t.gas - gasCost;

	// Increment associated nonce for sender.
	noteSending(sender);

	// Pay...
	cnote << "Paying" << formatBalance(cost) << "from sender (includes" << t.gas << "gas at" << formatBalance(t.gasPrice) << ")";
	subBalance(sender, cost);

	if (t.isCreation())
		create(sender, t.value, t.gasPrice, &gas, &t.data, &t.init);
	else
		call(t.receiveAddress, sender, t.value, t.gasPrice, bytesConstRef(&t.data), &gas, bytesRef());

	cnote << "Refunding" << formatBalance(gas * t.gasPrice) << "to sender (=" << gas << "*" << formatBalance(t.gasPrice) << ")";
	addBalance(sender, gas * t.gasPrice);

	u256 gasSpent = (t.gas - gas) * t.gasPrice;
/*	unsigned c_feesKept = 8;
	u256 feesEarned = gasSpent - (gasSpent / c_feesKept);
	cnote << "Transferring" << (100.0 - 100.0 / c_feesKept) << "% of" << formatBalance(gasSpent) << "=" << formatBalance(feesEarned) << "to miner (" << formatBalance(gasSpent - feesEarned) << "is burnt).";
*/
	u256 feesEarned = gasSpent;
	cnote << "Transferring" << formatBalance(gasSpent) << "to miner.";
	addBalance(m_currentBlock.coinbaseAddress, feesEarned);

	// Add to the user-originated transactions that we've executed.
	m_transactions.push_back(t);
	m_transactionSet.insert(t.sha3());
}

bool State::call(Address _receiveAddress, Address _senderAddress, u256 _value, u256 _gasPrice, bytesConstRef _data, u256* _gas, bytesRef _out, Address _originAddress)
{
	if (!_originAddress)
		_originAddress = _senderAddress;

	cnote << "Transferring" << formatBalance(_value) << "to receiver.";
	addBalance(_receiveAddress, _value);

	if (isContractAddress(_receiveAddress))
	{
		VM vm(*_gas);
		ExtVM evm(*this, _receiveAddress, _senderAddress, _originAddress, _value, _gasPrice, _data, &contractCode(_receiveAddress));
		bool revert = false;

		try
		{
			auto out = vm.go(evm);
			memcpy(_out.data(), out.data(), std::min(out.size(), _out.size()));
		}
		catch (OutOfGas const& /*_e*/)
		{
			clog(StateChat) << "Out of Gas! Reverting.";
			revert = true;
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

		// Write state out only in the case of a non-excepted transaction.
		if (revert)
			evm.revert();

		*_gas = vm.gas();

		return !revert;
	}
	return true;
}

h160 State::create(Address _sender, u256 _endowment, u256 _gasPrice, u256* _gas, bytesConstRef _code, bytesConstRef _init, Address _origin)
{
	if (!_origin)
		_origin = _sender;

	Address newAddress = right160(sha3(rlpList(_sender, transactionsFrom(_sender) - 1)));
	while (isContractAddress(newAddress) || isNormalAddress(newAddress))
		newAddress = (u160)newAddress + 1;

	// Set up new account...
	m_cache[newAddress] = AddressState(0, 0, _code);

	// Execute _init.
	VM vm(*_gas);
	ExtVM evm(*this, newAddress, _sender, _origin, _endowment, _gasPrice, bytesConstRef(), _init);
	bool revert = false;

	try
	{
		/*auto out =*/ vm.go(evm);
		// Don't do anything with the output (yet).
		//memcpy(_out.data(), out.data(), std::min(out.size(), _out.size()));
	}
	catch (OutOfGas const& /*_e*/)
	{
		clog(StateChat) << "Out of Gas! Reverting.";
		revert = true;
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

	// Write state out only in the case of a non-excepted transaction.
	if (revert)
	{
		evm.revert();
		m_cache.erase(newAddress);
		newAddress = Address();
	}

	*_gas = vm.gas();

	return newAddress;
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
