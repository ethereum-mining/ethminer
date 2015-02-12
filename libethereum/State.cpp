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

#include <ctime>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <secp256k1/secp256k1.h>
#include <libdevcore/CommonIO.h>
#include <libevmcore/Instruction.h>
#include <libethcore/Exceptions.h>
#include <libevm/VMFactory.h>
#include "BlockChain.h"
#include "Defaults.h"
#include "ExtVM.h"
#include "Executive.h"
#include "CachedAddressState.h"
#include "CanonBlockChain.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

#define ctrace clog(StateTrace)

static const u256 c_blockReward = 1500 * finney;

OverlayDB State::openDB(std::string _path, bool _killExisting)
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
	if (!db)
		BOOST_THROW_EXCEPTION(DatabaseAlreadyOpen());

	cnote << "Opened state DB.";
	return OverlayDB(db);
}

State::State(Address _coinbaseAddress, OverlayDB const& _db, BaseState _bs):
	m_db(_db),
	m_state(&m_db),
	m_ourAddress(_coinbaseAddress),
	m_blockReward(c_blockReward)
{
	// Initialise to the state entailed by the genesis block; this guarantees the trie is built correctly.
	m_state.init();

	paranoia("beginning of normal construction.", true);

	if (_bs == BaseState::CanonGenesis)
	{
		dev::eth::commit(genesisState(), m_db, m_state);
		m_db.commit();

		paranoia("after DB commit of normal construction.", true);
		m_previousBlock = CanonBlockChain::genesis();
	}
	else
		m_previousBlock.setEmpty();

	resetCurrent();

	assert(m_state.root() == m_previousBlock.stateRoot);

	paranoia("end of normal construction.", true);
}

State::State(OverlayDB const& _db, BlockChain const& _bc, h256 _h):
	m_db(_db),
	m_state(&m_db),
	m_blockReward(c_blockReward)
{
	// TODO THINK: is this necessary?
	m_state.init();

	auto b = _bc.block(_h);
	BlockInfo bi;
	BlockInfo bip;
	if (_h)
		bi.populate(b);
	if (bi && bi.number)
		bip.populate(_bc.block(bi.parentHash));
	if (!_h || !bip)
		return;
	m_ourAddress = bi.coinbaseAddress;

	sync(_bc, bi.parentHash, bip);
	enact(&b, _bc);
}

State::State(State const& _s):
	m_db(_s.m_db),
	m_state(&m_db, _s.m_state.root()),
	m_transactions(_s.m_transactions),
	m_receipts(_s.m_receipts),
	m_transactionSet(_s.m_transactionSet),
	m_cache(_s.m_cache),
	m_previousBlock(_s.m_previousBlock),
	m_currentBlock(_s.m_currentBlock),
	m_ourAddress(_s.m_ourAddress),
	m_blockReward(_s.m_blockReward)
{
	paranoia("after state cloning (copy cons).", true);
}

void State::paranoia(std::string const& _when, bool _enforceRefs) const
{
#if ETH_PARANOIA
	// TODO: variable on context; just need to work out when there should be no leftovers
	// [in general this is hard since contract alteration will result in nodes in the DB that are no directly part of the state DB].
	if (!isTrieGood(_enforceRefs, false))
	{
		cwarn << "BAD TRIE" << _when;
		BOOST_THROW_EXCEPTION(InvalidTrie());
	}
#else
	(void)_when;
	(void)_enforceRefs;
#endif
}

State& State::operator=(State const& _s)
{
	m_db = _s.m_db;
	m_state.open(&m_db, _s.m_state.root());
	m_transactions = _s.m_transactions;
	m_receipts = _s.m_receipts;
	m_transactionSet = _s.m_transactionSet;
	m_cache = _s.m_cache;
	m_previousBlock = _s.m_previousBlock;
	m_currentBlock = _s.m_currentBlock;
	m_ourAddress = _s.m_ourAddress;
	m_blockReward = _s.m_blockReward;
	m_lastTx = _s.m_lastTx;
	paranoia("after state cloning (assignment op)", true);
	return *this;
}

State::~State()
{
}

Address State::nextActiveAddress(Address _a) const
{
	auto it = m_state.lower_bound(_a);
	if ((*it).first == _a)
		++it;
	if (it == m_state.end())
		// exchange comments if we want to wraparound
//		it = m_state.begin();
		return Address();
	return (*it).first;
}

StateDiff State::diff(State const& _c) const
{
	StateDiff ret;

	std::set<Address> ads;
	std::set<Address> trieAds;
	std::set<Address> trieAdsD;

	auto trie = TrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&m_db), rootHash());
	auto trieD = TrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&_c.m_db), _c.rootHash());

	for (auto i: trie)
		ads.insert(i.first), trieAds.insert(i.first);
	for (auto i: trieD)
		ads.insert(i.first), trieAdsD.insert(i.first);
	for (auto i: m_cache)
		ads.insert(i.first);
	for (auto i: _c.m_cache)
		ads.insert(i.first);

//	cnote << *this;
//	cnote << _c;

	for (auto i: ads)
	{
		auto it = m_cache.find(i);
		auto itD = _c.m_cache.find(i);
		CachedAddressState source(trieAds.count(i) ? trie.at(i) : "", it != m_cache.end() ? &it->second : nullptr, &m_db);
		CachedAddressState dest(trieAdsD.count(i) ? trieD.at(i) : "", itD != _c.m_cache.end() ? &itD->second : nullptr, &_c.m_db);
		AccountDiff acd = source.diff(dest);
		if (acd.changed())
			ret.accounts[i] = acd;
	}

	return ret;
}

void State::ensureCached(Address _a, bool _requireCode, bool _forceCreate) const
{
	ensureCached(m_cache, _a, _requireCode, _forceCreate);
}

void State::ensureCached(std::map<Address, Account>& _cache, Address _a, bool _requireCode, bool _forceCreate) const
{
	auto it = _cache.find(_a);
	if (it == _cache.end())
	{
		// populate basic info.
		string stateBack = m_state.at(_a);
		if (stateBack.empty() && !_forceCreate)
			return;
		RLP state(stateBack);
		Account s;
		if (state.isNull())
			s = Account(0, Account::NormalCreation);
		else
			s = Account(state[0].toInt<u256>(), state[1].toInt<u256>(), state[2].toHash<h256>(), state[3].toHash<h256>());
		bool ok;
		tie(it, ok) = _cache.insert(make_pair(_a, s));
	}
	if (_requireCode && it != _cache.end() && !it->second.isFreshCode() && !it->second.codeCacheValid())
		it->second.noteCode(it->second.codeHash() == EmptySHA3 ? bytesConstRef() : bytesConstRef(m_db.lookup(it->second.codeHash())));
}

void State::commit()
{
	dev::eth::commit(m_cache, m_db, m_state);
	m_cache.clear();
}

bool State::sync(BlockChain const& _bc)
{
	return sync(_bc, _bc.currentHash());
}

bool State::sync(BlockChain const& _bc, h256 _block, BlockInfo const& _bi)
{
	bool ret = false;
	// BLOCK
	BlockInfo bi = _bi;
	if (!bi)
		while (1)
		{
			try
			{
				auto b = _bc.block(_block);
				bi.populate(b);
	//			bi.verifyInternals(_bc.block(_block));	// Unneeded - we already verify on import into the blockchain.
				break;
			}
			catch (Exception const& _e)
			{
				// TODO: Slightly nicer handling? :-)
				cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
				cerr << diagnostic_information(_e) << endl;
			}
			catch (std::exception const& _e)
			{
				// TODO: Slightly nicer handling? :-)
				cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
				cerr << _e.what() << endl;
			}
		}
	if (bi == m_currentBlock)
	{
		// We mined the last block.
		// Our state is good - we just need to move on to next.
		m_previousBlock = m_currentBlock;
		resetCurrent();
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
		while (bi.number != 0 && m_db.lookup(bi.stateRoot).empty())	// while we don't have the state root of the latest block...
		{
			chain.push_back(bi.hash);				// push back for later replay.
			bi.populate(_bc.block(bi.parentHash));	// move to parent.
		}

		m_previousBlock = bi;
		resetCurrent();

		// Iterate through in reverse, playing back each of the blocks.
		try
		{
			for (auto it = chain.rbegin(); it != chain.rend(); ++it)
			{
				auto b = _bc.block(*it);
				enact(&b, _bc);
				cleanup(true);
			}
		}
		catch (...)
		{
			// TODO: Slightly nicer handling? :-)
			cerr << "ERROR: Corrupt block-chain! Delete your block-chain DB and restart." << endl;
			cerr << boost::current_exception_diagnostic_information() << endl;
			exit(1);
		}

		resetCurrent();
		ret = true;
	}
	return ret;
}

u256 State::enactOn(bytesConstRef _block, BlockInfo const& _bi, BlockChain const& _bc)
{
	// Check family:
	BlockInfo biParent(_bc.block(_bi.parentHash));
	_bi.verifyParent(biParent);
	BlockInfo biGrandParent;
	if (biParent.number)
		biGrandParent.populate(_bc.block(biParent.parentHash));
	sync(_bc, _bi.parentHash);
	resetCurrent();
	m_previousBlock = biParent;
	return enact(_block, _bc);
}

map<Address, u256> State::addresses() const
{
	map<Address, u256> ret;
	for (auto i: m_cache)
		if (i.second.isAlive())
			ret[i.first] = i.second.balance();
	for (auto const& i: m_state)
		if (m_cache.find(i.first) == m_cache.end())
			ret[i.first] = RLP(i.second)[1].toInt<u256>();
	return ret;
}

void State::resetCurrent()
{
	m_transactions.clear();
	m_receipts.clear();
	m_transactionSet.clear();
	m_cache.clear();
	m_currentBlock = BlockInfo();
	m_currentBlock.coinbaseAddress = m_ourAddress;
	m_currentBlock.timestamp = max(m_previousBlock.timestamp + 1, (u256)time(0));
	m_currentBlock.transactionsRoot = h256();
	m_currentBlock.sha3Uncles = h256();
	m_currentBlock.populateFromParent(m_previousBlock);

	// Update timestamp according to clock.
	// TODO: check.

	m_lastTx = m_db;
	m_state.setRoot(m_previousBlock.stateRoot);

	paranoia("begin resetCurrent", true);
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
				Transaction t(i.second, CheckSignature::Sender);
				if (t.nonce() <= transactionsFrom(t.sender()))
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

TransactionReceipts State::sync(BlockChain const& _bc, TransactionQueue& _tq, bool* o_transactionQueueChanged)
{
	// TRANSACTIONS
	TransactionReceipts ret;
	auto ts = _tq.transactions();

	auto lh = getLastHashes(_bc, _bc.number());

	for (int goodTxs = 1; goodTxs;)
	{
		goodTxs = 0;
		for (auto const& i: ts)
			if (!m_transactionSet.count(i.first))
			{
				// don't have it yet! Execute it now.
				try
				{
					uncommitToMine();
//					boost::timer t;
					execute(lh, i.second);
					ret.push_back(m_receipts.back());
					_tq.noteGood(i);
					++goodTxs;
//					cnote << "TX took:" << t.elapsed() * 1000;
				}
				catch (InvalidNonce const& in)
				{
					if (in.required > in.candidate)
					{
						// too old
						_tq.drop(i.first);
						if (o_transactionQueueChanged)
							*o_transactionQueueChanged = true;
					}
					else
						_tq.setFuture(i);
				}
				catch (Exception const& _e)
				{
					// Something else went wrong - drop it.
					_tq.drop(i.first);
					if (o_transactionQueueChanged)
						*o_transactionQueueChanged = true;
					cwarn << "Sync went wrong\n" << diagnostic_information(_e);
				}
				catch (std::exception const&)
				{
					// Something else went wrong - drop it.
					_tq.drop(i.first);
					if (o_transactionQueueChanged)
						*o_transactionQueueChanged = true;
				}
			}
	}
	return ret;
}

u256 State::enact(bytesConstRef _block, BlockChain const& _bc, bool _checkNonce)
{
	// m_currentBlock is assumed to be prepopulated and reset.

#if !ETH_RELEASE
	BlockInfo bi(_block, _checkNonce);
	assert(m_previousBlock.hash == bi.parentHash);
	assert(m_currentBlock.parentHash == bi.parentHash);
	assert(rootHash() == m_previousBlock.stateRoot);
#endif

	if (m_currentBlock.parentHash != m_previousBlock.hash)
		BOOST_THROW_EXCEPTION(InvalidParentHash());

	// Populate m_currentBlock with the correct values.
	m_currentBlock.populate(_block, _checkNonce);
	m_currentBlock.verifyInternals(_block);

//	cnote << "playback begins:" << m_state.root();
//	cnote << m_state;

	MemoryDB tm;
	GenericTrieDB<MemoryDB> transactionsTrie(&tm);
	transactionsTrie.init();

	MemoryDB rm;
	GenericTrieDB<MemoryDB> receiptsTrie(&rm);
	receiptsTrie.init();

	LastHashes lh = getLastHashes(_bc, (unsigned)m_previousBlock.number);

	// All ok with the block generally. Play back the transactions now...
	unsigned i = 0;
	for (auto const& tr: RLP(_block)[1])
	{
		RLPStream k;
		k << i;

		transactionsTrie.insert(&k.out(), tr.data());
		execute(lh, tr.data());

		RLPStream receiptrlp;
		m_receipts.back().streamRLP(receiptrlp);
		receiptsTrie.insert(&k.out(), &receiptrlp.out());
		++i;
	}

	if (transactionsTrie.root() != m_currentBlock.transactionsRoot)
	{
		cwarn << "Bad transactions state root!";
		BOOST_THROW_EXCEPTION(InvalidTransactionsStateRoot());
	}

	if (receiptsTrie.root() != m_currentBlock.receiptsRoot)
	{
		cwarn << "Bad receipts state root.";
		cwarn << "Block:" << toHex(_block);
		cwarn << "Block RLP:" << RLP(_block);
		cwarn << "Calculated: " << receiptsTrie.root();
		for (unsigned j = 0; j < i; ++j)
		{
			RLPStream k;
			k << j;
			auto b = asBytes(receiptsTrie.at(&k.out()));
			cwarn << j << ": ";
			cwarn << "RLP: " << RLP(b);
			cwarn << "Hex: " << toHex(b);
			cwarn << TransactionReceipt(&b);
		}
        cwarn << "Recorded: " << m_currentBlock.receiptsRoot;
		auto rs = _bc.receipts(m_currentBlock.hash);
		for (unsigned j = 0; j < rs.receipts.size(); ++j)
		{
			auto b = rs.receipts[j].rlp();
			cwarn << j << ": ";
			cwarn << "RLP: " << RLP(b);
			cwarn << "Hex: " << toHex(b);
			cwarn << rs.receipts[j];
		}
		BOOST_THROW_EXCEPTION(InvalidReceiptsStateRoot());
	}

	if (m_currentBlock.logBloom != logBloom())
	{
		cwarn << "Bad log bloom!";
		BOOST_THROW_EXCEPTION(InvalidLogBloom());
	}

	// Initialise total difficulty calculation.
	u256 tdIncrease = m_currentBlock.difficulty;

	// Check uncles & apply their rewards to state.
	set<h256> nonces = { m_currentBlock.nonce };
	Addresses rewarded;
	set<h256> knownUncles = _bc.allUnclesFrom(m_currentBlock.parentHash);
	for (auto const& i: RLP(_block)[2])
	{
		if (knownUncles.count(sha3(i.data())))
			BOOST_THROW_EXCEPTION(UncleInChain(knownUncles, sha3(i.data()) ));

		BlockInfo uncle = BlockInfo::fromHeader(i.data());
		if (nonces.count(uncle.nonce))
			BOOST_THROW_EXCEPTION(DuplicateUncleNonce());

		BlockInfo uncleParent(_bc.block(uncle.parentHash));
		if ((bigint)uncleParent.number < (bigint)m_currentBlock.number - 7)
			BOOST_THROW_EXCEPTION(UncleTooOld());
		uncle.verifyParent(uncleParent);

		nonces.insert(uncle.nonce);
		tdIncrease += uncle.difficulty;
		rewarded.push_back(uncle.coinbaseAddress);
	}
	applyRewards(rewarded);

	// Commit all cached state changes to the state trie.
	commit();

	// Hash the state trie and check against the state_root hash in m_currentBlock.
	if (m_currentBlock.stateRoot != m_previousBlock.stateRoot && m_currentBlock.stateRoot != rootHash())
	{
		cwarn << "Bad state root!";
		cnote << "Given to be:" << m_currentBlock.stateRoot;
		cnote << TrieDB<Address, OverlayDB>(&m_db, m_currentBlock.stateRoot);
		cnote << "Calculated to be:" << rootHash();
		cnote << m_state;
		cnote << *this;
		// Rollback the trie.
		m_db.rollback();
		BOOST_THROW_EXCEPTION(InvalidStateRoot());
	}

	return tdIncrease;
}

void State::cleanup(bool _fullCommit)
{
	if (_fullCommit)
	{
		paranoia("immediately before database commit", true);

		// Commit the new trie to disk.
		m_db.commit();

		paranoia("immediately after database commit", true);
		m_previousBlock = m_currentBlock;
	}
	else
		m_db.rollback();

	resetCurrent();
}

void State::uncommitToMine()
{
	if (m_currentBlock.sha3Uncles)
	{
		m_cache.clear();
		if (!m_transactions.size())
			m_state.setRoot(m_previousBlock.stateRoot);
		else
			m_state.setRoot(m_receipts[m_receipts.size() - 1].stateRoot());
		m_db = m_lastTx;
		paranoia("Uncommited to mine", true);
		m_currentBlock.sha3Uncles = h256();
	}
}

bool State::amIJustParanoid(BlockChain const& _bc)
{
	commitToMine(_bc);

	// Update difficulty according to timestamp.
	m_currentBlock.difficulty = m_currentBlock.calculateDifficulty(m_previousBlock);

	// Compile block:
	RLPStream block;
	block.appendList(3);
	m_currentBlock.streamRLP(block, WithNonce);
	block.appendRaw(m_currentTxs);
	block.appendRaw(m_currentUncles);

	State s(*this);
	s.resetCurrent();
	try
	{
		cnote << "PARANOIA root:" << s.rootHash();
//		s.m_currentBlock.populate(&block.out(), false);
//		s.m_currentBlock.verifyInternals(&block.out());
		s.enact(&block.out(), _bc, false);	// don't check nonce for this since we haven't mined it yet.
		s.cleanup(false);
		return true;
	}
	catch (Exception const& _e)
	{
		cwarn << "Bad block: " << diagnostic_information(_e);
	}
	catch (std::exception const& _e)
	{
		cwarn << "Bad block: " << _e.what();
	}

	return false;
}

LogBloom State::logBloom() const
{
	LogBloom ret;
	for (TransactionReceipt const& i: m_receipts)
		ret |= i.bloom();
	return ret;
}

void State::commitToMine(BlockChain const& _bc)
{
	uncommitToMine();

//	cnote << "Committing to mine on block" << m_previousBlock.hash.abridged();
#ifdef ETH_PARANOIA
	commit();
	cnote << "Pre-reward stateRoot:" << m_state.root();
#endif

	m_lastTx = m_db;

	Addresses uncleAddresses;

	RLPStream unclesData;
	unsigned unclesCount = 0;
	if (m_previousBlock.number != 0)
	{
		// Find great-uncles (or second-cousins or whatever they are) - children of great-grandparents, great-great-grandparents... that were not already uncles in previous generations.
//		cout << "Checking " << m_previousBlock.hash << ", parent=" << m_previousBlock.parentHash << endl;
		set<h256> knownUncles = _bc.allUnclesFrom(m_currentBlock.parentHash);
		auto p = m_previousBlock.parentHash;
		for (unsigned gen = 0; gen < 6 && p != _bc.genesisHash(); ++gen, p = _bc.details(p).parent)
		{
			auto us = _bc.details(p).children;
			assert(us.size() >= 1);	// must be at least 1 child of our grandparent - it's our own parent!
			for (auto const& u: us)
				if (!knownUncles.count(u))	// ignore any uncles/mainline blocks that we know about.
				{
					BlockInfo ubi(_bc.block(u));
					ubi.streamRLP(unclesData, WithNonce);
					++unclesCount;
					uncleAddresses.push_back(ubi.coinbaseAddress);
				}
		}
	}

	MemoryDB tm;
	GenericTrieDB<MemoryDB> transactionsTrie(&tm);
	transactionsTrie.init();

	MemoryDB rm;
	GenericTrieDB<MemoryDB> receiptsTrie(&rm);
	receiptsTrie.init();

	RLPStream txs;
	txs.appendList(m_transactions.size());

	for (unsigned i = 0; i < m_transactions.size(); ++i)
	{
		RLPStream k;
		k << i;

		RLPStream receiptrlp;
		m_receipts[i].streamRLP(receiptrlp);
		receiptsTrie.insert(&k.out(), &receiptrlp.out());

		RLPStream txrlp;
		m_transactions[i].streamRLP(txrlp);
		transactionsTrie.insert(&k.out(), &txrlp.out());

		txs.appendRaw(txrlp.out());
	}

	txs.swapOut(m_currentTxs);

	RLPStream(unclesCount).appendRaw(unclesData.out(), unclesCount).swapOut(m_currentUncles);

	m_currentBlock.transactionsRoot = transactionsTrie.root();
	m_currentBlock.receiptsRoot = receiptsTrie.root();
	m_currentBlock.logBloom = logBloom();
	m_currentBlock.sha3Uncles = sha3(m_currentUncles);

	// Apply rewards last of all.
	applyRewards(uncleAddresses);

	// Commit any and all changes to the trie that are in the cache, then update the state root accordingly.
	commit();

//	cnote << "Post-reward stateRoot:" << m_state.root().abridged();
//	cnote << m_state;
//	cnote << *this;

	m_currentBlock.gasUsed = gasUsed();
	m_currentBlock.stateRoot = m_state.root();
	m_currentBlock.parentHash = m_previousBlock.hash;
}

MineInfo State::mine(unsigned _msTimeout, bool _turbo)
{
	// Update difficulty according to timestamp.
	m_currentBlock.difficulty = m_currentBlock.calculateDifficulty(m_previousBlock);

	MineInfo ret;
	// TODO: Miner class that keeps dagger between mine calls (or just non-polling mining).
	tie(ret, m_currentBlock.nonce) = m_pow.mine(m_currentBlock.headerHash(WithoutNonce), m_currentBlock.difficulty, _msTimeout, true, _turbo);

	if (!ret.completed)
		m_currentBytes.clear();
	else
		cnote << "Completed" << m_currentBlock.headerHash(WithoutNonce).abridged() << m_currentBlock.nonce.abridged() << m_currentBlock.difficulty << ProofOfWork::verify(m_currentBlock.headerHash(WithoutNonce), m_currentBlock.nonce, m_currentBlock.difficulty);

	return ret;
}

bool State::completeMine(h256 const& _nonce)
{
	if (!m_pow.verify(m_currentBlock.headerHash(WithoutNonce), _nonce, m_currentBlock.difficulty))
		return false;

	m_currentBlock.nonce = _nonce;
	cnote << "Completed" << m_currentBlock.headerHash(WithoutNonce).abridged() << m_currentBlock.nonce.abridged() << m_currentBlock.difficulty << ProofOfWork::verify(m_currentBlock.headerHash(WithoutNonce), m_currentBlock.nonce, m_currentBlock.difficulty);

	completeMine();

	return true;
}

void State::completeMine()
{
	cdebug << "Completing mine!";
	// Got it!

	// Compile block:
	RLPStream ret;
	ret.appendList(3);
	m_currentBlock.streamRLP(ret, WithNonce);
	ret.appendRaw(m_currentTxs);
	ret.appendRaw(m_currentUncles);
	ret.swapOut(m_currentBytes);
	m_currentBlock.hash = sha3(RLP(m_currentBytes)[0].data());
	cnote << "Mined " << m_currentBlock.hash.abridged() << "(parent: " << m_currentBlock.parentHash.abridged() << ")";

	// Quickly reset the transactions.
	// TODO: Leave this in a better state than this limbo, or at least record that it's in limbo.
	m_transactions.clear();
	m_receipts.clear();
	m_transactionSet.clear();
	m_lastTx = m_db;
}

bool State::addressInUse(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return true;
}

bool State::addressHasCode(Address _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return it->second.isFreshCode() || it->second.codeHash() != EmptySHA3;
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
	if (asserts(it != m_cache.end()))
	{
		cwarn << "Sending from non-existant account. How did it pay!?!";
		// this is impossible. but we'll continue regardless...
		m_cache[_id] = Account(1, 0);
	}
	else
		it->second.incNonce();
}

void State::addBalance(Address _id, u256 _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		m_cache[_id] = Account(_amount, Account::NormalCreation);
	else
		it->second.addBalance(_amount);
}

void State::subBalance(Address _id, bigint _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end() || (bigint)it->second.balance() < _amount)
		BOOST_THROW_EXCEPTION(NotEnoughCash());
	else
		it->second.addBalance(-_amount);
}

Address State::newContract(u256 _balance, bytes const& _code)
{
	auto h = sha3(_code);
	m_db.insert(h, &_code);
	while (true)
	{
		Address ret = Address::random();
		ensureCached(ret, false, false);
		auto it = m_cache.find(ret);
		if (it == m_cache.end())
		{
			m_cache[ret] = Account(0, _balance, EmptyTrie, h);
			return ret;
		}
	}
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

u256 State::storage(Address _id, u256 _memory) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);

	// Account doesn't exist - exit now.
	if (it == m_cache.end())
		return 0;

	// See if it's in the account's storage cache.
	auto mit = it->second.storageOverlay().find(_memory);
	if (mit != it->second.storageOverlay().end())
		return mit->second;

	// Not in the storage cache - go to the DB.
	TrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&m_db), it->second.baseRoot());			// promise we won't change the overlay! :)
	string payload = memdb.at(_memory);
	u256 ret = payload.size() ? RLP(payload).toInt<u256>() : 0;
	it->second.setStorage(_memory, ret);
	return ret;
}

map<u256, u256> State::storage(Address _id) const
{
	map<u256, u256> ret;

	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it != m_cache.end())
	{
		// Pull out all values from trie storage.
		if (it->second.baseRoot())
		{
			TrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&m_db), it->second.baseRoot());		// promise we won't alter the overlay! :)
			for (auto const& i: memdb)
				ret[i.first] = RLP(i.second).toInt<u256>();
		}

		// Then merge cached storage over the top.
		for (auto const& i: it->second.storageOverlay())
			if (i.second)
				ret[i.first] = i.second;
			else
				ret.erase(i.first);
	}
	return ret;
}

h256 State::storageRoot(Address _id) const
{
	string s = m_state.at(_id);
	if (s.size())
	{
		RLP r(s);
		return r[2].toHash<h256>();
	}
	return EmptyTrie;
}

bytes const& State::code(Address _contract) const
{
	if (!addressHasCode(_contract))
		return NullBytes;
	ensureCached(_contract, true, false);
	return m_cache[_contract].code();
}

h256 State::codeHash(Address _contract) const
{
	if (!addressHasCode(_contract))
		return EmptySHA3;
	return m_cache[_contract].codeHash();
}

bool State::isTrieGood(bool _enforceRefs, bool _requireNoLeftOvers) const
{
	for (int e = 0; e < (_enforceRefs ? 2 : 1); ++e)
		try
		{
			EnforceRefs r(m_db, !!e);
			auto lo = m_state.leftOvers();
			if (!lo.empty() && _requireNoLeftOvers)
			{
				cwarn << "LEFTOVERS" << (e ? "[enforced" : "[unenforced") << "refs]";
				cnote << "Left:" << lo;
				cnote << "Keys:" << m_db.keys();
				m_state.debugStructure(cerr);
				return false;
			}
			// TODO: Enable once fixed.
			for (auto const& i: m_state)
			{
				RLP r(i.second);
				TrieDB<h256, OverlayDB> storageDB(const_cast<OverlayDB*>(&m_db), r[2].toHash<h256>());	// promise not to alter OverlayDB.
				for (auto const& j: storageDB) { (void)j; }
				if (!e && r[3].toHash<h256>() != EmptySHA3 && m_db.lookup(r[3].toHash<h256>()).empty())
					return false;
			}
		}
		catch (InvalidTrie const&)
		{
			cwarn << "BAD TRIE" << (e ? "[enforced" : "[unenforced") << "refs]";
			cnote << m_db.keys();
			m_state.debugStructure(cerr);
			return false;
		}
	return true;
}

LastHashes State::getLastHashes(BlockChain const& _bc, unsigned _n) const
{
	LastHashes ret;
	ret.resize(256);
	if (c_protocolVersion > 49)
	{
		ret[0] = _bc.numberHash(_n);
		for (unsigned i = 1; i < 256; ++i)
			ret[i] = ret[i - 1] ? _bc.details(ret[i - 1]).parent : h256();
	}
	return ret;
}

u256 State::execute(BlockChain const& _bc, bytes const& _rlp, bytes* o_output, bool _commit)
{
	return execute(getLastHashes(_bc, _bc.number()), &_rlp, o_output, _commit);
}

u256 State::execute(BlockChain const& _bc, bytesConstRef _rlp, bytes* o_output, bool _commit)
{
	return execute(getLastHashes(_bc, _bc.number()), _rlp, o_output, _commit);
}

// TODO: maintain node overlay revisions for stateroots -> each commit gives a stateroot + OverlayDB; allow overlay copying for rewind operations.
u256 State::execute(LastHashes const& _lh, bytesConstRef _rlp, bytes* o_output, bool _commit)
{
#ifndef ETH_RELEASE
	commit();	// get an updated hash
#endif

	paranoia("start of execution.", true);

	State old(*this);
#if ETH_PARANOIA
	auto h = rootHash();
#endif

	Executive e(*this, _lh, 0);
	e.setup(_rlp);

	u256 startGasUsed = gasUsed();

#if ETH_PARANOIA
	ctrace << "Executing" << e.t() << "on" << h;
	ctrace << toHex(e.t().rlp());
#endif
#if ETH_VMTRACE
	e.go(e.simpleTrace());
#else
	e.go();
#endif
	e.finalize();

#if ETH_PARANOIA
	ctrace << "Ready for commit;";
	ctrace << old.diff(*this);
#endif

	if (o_output)
		*o_output = e.out().toBytes();

	if (!_commit)
	{
		m_cache.clear();
		return e.gasUsed();
	}

	commit();

#if ETH_PARANOIA
	ctrace << "Executed; now" << rootHash();
	ctrace << old.diff(*this);

	paranoia("after execution commit.", true);

	if (e.t().receiveAddress())
	{
		EnforceRefs r(m_db, true);
		if (storageRoot(e.t().receiveAddress()) && m_db.lookup(storageRoot(e.t().receiveAddress())).empty())
		{
			cwarn << "TRIE immediately after execution; no node for receiveAddress";
			BOOST_THROW_EXCEPTION(InvalidTrie());
		}
	}
#endif

	// TODO: CHECK TRIE after level DB flush to make sure exactly the same.

	// Add to the user-originated transactions that we've executed.
	m_transactions.push_back(e.t());
	m_receipts.push_back(TransactionReceipt(rootHash(), startGasUsed + e.gasUsed(), e.logs()));
	m_transactionSet.insert(e.t().sha3());
	return e.gasUsed();
}

State State::fromPending(unsigned _i) const
{
	State ret = *this;
	ret.m_cache.clear();
	_i = min<unsigned>(_i, m_transactions.size());
	if (!_i)
		ret.m_state.setRoot(m_previousBlock.stateRoot);
	else
		ret.m_state.setRoot(m_receipts[_i - 1].stateRoot());
	while (ret.m_transactions.size() > _i)
	{
		ret.m_transactionSet.erase(ret.m_transactions.back().sha3());
		ret.m_transactions.pop_back();
		ret.m_receipts.pop_back();
	}
	return ret;
}

void State::applyRewards(Addresses const& _uncleAddresses)
{
	u256 r = m_blockReward;
	for (auto const& i: _uncleAddresses)
	{
		addBalance(i, m_blockReward * 15 / 16);
		r += m_blockReward / 32;
	}
	addBalance(m_currentBlock.coinbaseAddress, r);
}

std::ostream& dev::eth::operator<<(std::ostream& _out, State const& _s)
{
	_out << "--- " << _s.rootHash() << std::endl;
	std::set<Address> d;
	std::set<Address> dtr;
	auto trie = TrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&_s.m_db), _s.rootHash());
	for (auto i: trie)
		d.insert(i.first), dtr.insert(i.first);
	for (auto i: _s.m_cache)
		d.insert(i.first);

	for (auto i: d)
	{
		auto it = _s.m_cache.find(i);
		Account* cache = it != _s.m_cache.end() ? &it->second : nullptr;
		string rlpString = dtr.count(i) ? trie.at(i) : "";
		RLP r(rlpString);
		assert(cache || r);

		if (cache && !cache->isAlive())
			_out << "XXX  " << i << std::endl;
		else
		{
			string lead = (cache ? r ? " *   " : " +   " : "     ");
			if (cache && r && cache->nonce() == r[0].toInt<u256>() && cache->balance() == r[1].toInt<u256>())
				lead = " .   ";

			stringstream contout;

			if ((cache && cache->codeBearing()) || (!cache && r && (h256)r[3] != EmptySHA3))
			{
				std::map<u256, u256> mem;
				std::set<u256> back;
				std::set<u256> delta;
				std::set<u256> cached;
				if (r)
				{
					TrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&_s.m_db), r[2].toHash<h256>());		// promise we won't alter the overlay! :)
					for (auto const& j: memdb)
						mem[j.first] = RLP(j.second).toInt<u256>(), back.insert(j.first);
				}
				if (cache)
					for (auto const& j: cache->storageOverlay())
					{
						if ((!mem.count(j.first) && j.second) || (mem.count(j.first) && mem.at(j.first) != j.second))
							mem[j.first] = j.second, delta.insert(j.first);
						else if (j.second)
							cached.insert(j.first);
					}
				if (delta.size())
					lead = (lead == " .   ") ? "*.*  " : "***  ";

				contout << " @:";
				if (delta.size())
					contout << "???";
				else
					contout << r[2].toHash<h256>();
				if (cache && cache->isFreshCode())
					contout << " $" << toHex(cache->code());
				else
					contout << " $" << (cache ? cache->codeHash() : r[3].toHash<h256>());

				for (auto const& j: mem)
					if (j.second)
						contout << std::endl << (delta.count(j.first) ? back.count(j.first) ? " *     " : " +     " : cached.count(j.first) ? " .     " : "       ") << std::hex << nouppercase << std::setw(64) << j.first << ": " << std::setw(0) << j.second ;
					else
						contout << std::endl << "XXX    " << std::hex << nouppercase << std::setw(64) << j.first << "";
			}
			else
				contout << " [SIMPLE]";
			_out << lead << i << ": " << std::dec << (cache ? cache->nonce() : r[0].toInt<u256>()) << " #:" << (cache ? cache->balance() : r[1].toInt<u256>()) << contout.str() << std::endl;
		}
	}
	return _out;
}
