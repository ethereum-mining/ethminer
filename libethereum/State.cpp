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
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <libdevcore/CommonIO.h>
#include <libdevcore/Assertions.h>
#include <libdevcore/StructuredLogger.h>
#include <libdevcore/TrieHash.h>
#include <libevmcore/Instruction.h>
#include <libethcore/Exceptions.h>
#include <libethcore/Params.h>
#include <libevm/VMFactory.h>
#include "BlockChain.h"
#include "Defaults.h"
#include "ExtVM.h"
#include "Executive.h"
#include "CachedAddressState.h"
#include "CanonBlockChain.h"
#include "TransactionQueue.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
namespace fs = boost::filesystem;

#define ctrace clog(StateTrace)
#define ETH_TIMED_ENACTMENTS 0

static const unsigned c_maxSyncTransactions = 256;

const char* StateSafeExceptions::name() { return EthViolet "⚙" EthBlue " ℹ"; }
const char* StateDetail::name() { return EthViolet "⚙" EthWhite " ◌"; }
const char* StateTrace::name() { return EthViolet "⚙" EthGray " ◎"; }
const char* StateChat::name() { return EthViolet "⚙" EthWhite " ◌"; }

OverlayDB State::openDB(std::string const& _basePath, h256 const& _genesisHash, WithExisting _we)
{
	std::string path = _basePath.empty() ? Defaults::get()->m_dbPath : _basePath;

	if (_we == WithExisting::Kill)
	{
		cnote << "Killing state database (WithExisting::Kill).";
		boost::filesystem::remove_all(path + "/state");
	}

	path += "/" + toHex(_genesisHash.ref().cropped(0, 4)) + "/" + toString(c_databaseVersion);
	boost::filesystem::create_directories(path);
	fs::permissions(path, fs::owner_all);

	ldb::Options o;
	o.max_open_files = 256;
	o.create_if_missing = true;
	ldb::DB* db = nullptr;
	ldb::Status status = ldb::DB::Open(o, path + "/state", &db);
	if (!status.ok() || !db)
	{
		if (boost::filesystem::space(path + "/state").available < 1024)
		{
			cwarn << "Not enough available space found on hard drive. Please free some up and then re-run. Bailing.";
			BOOST_THROW_EXCEPTION(NotEnoughAvailableSpace());
		}
		else
		{
			cwarn << status.ToString();
			cwarn << "Database already open. You appear to have another instance of ethereum running. Bailing.";
			BOOST_THROW_EXCEPTION(DatabaseAlreadyOpen());
		}
	}

	cnote << "Opened state DB.";
	return OverlayDB(db);
}

State::State(OverlayDB const& _db, BaseState _bs, Address _coinbaseAddress):
	m_db(_db),
	m_state(&m_db),
	m_ourAddress(_coinbaseAddress),
	m_blockReward(c_blockReward)
{
	if (_bs != BaseState::PreExisting)
		// Initialise to the state entailed by the genesis block; this guarantees the trie is built correctly.
		m_state.init();

	paranoia("beginning of Genesis construction.", true);

	m_previousBlock.clear();
	m_currentBlock.clear();
//	assert(m_state.root() == m_previousBlock.stateRoot());

	paranoia("end of normal construction.", true);
}

PopulationStatistics State::populateFromChain(BlockChain const& _bc, h256 const& _h, ImportRequirements::value _ir)
{
	PopulationStatistics ret { 0.0, 0.0 };

	if (!_bc.isKnown(_h))
	{
		// Might be worth throwing here.
		cwarn << "Invalid block given for state population: " << _h;
		BOOST_THROW_EXCEPTION(BlockNotFound() << errinfo_target(_h));
	}

	auto b = _bc.block(_h);
	BlockInfo bi(b);
	if (bi.number())
	{
		// Non-genesis:

		// 1. Start at parent's end state (state root).
		BlockInfo bip(_bc.block(bi.parentHash()));
		sync(_bc, bi.parentHash(), bip);

		// 2. Enact the block's transactions onto this state.
		m_ourAddress = bi.coinbaseAddress();
		Timer t;
		auto vb = _bc.verifyBlock(&b, function<void(Exception&)>(), _ir | ImportRequirements::TransactionBasic);
		ret.verify = t.elapsed();
		t.restart();
		enact(vb, _bc);
		ret.enact = t.elapsed();
	}
	else
	{
		// Genesis required:
		// We know there are no transactions, so just populate directly.
		m_state.init();
		sync(_bc, _h, bi);
	}

	return ret;
}

State::State(State const& _s):
	m_db(_s.m_db),
	m_state(&m_db, _s.m_state.root(), Verification::Skip),
	m_transactions(_s.m_transactions),
	m_receipts(_s.m_receipts),
	m_transactionSet(_s.m_transactionSet),
	m_touched(_s.m_touched),
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
#if ETH_PARANOIA && !ETH_FATDB
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
	m_state.open(&m_db, _s.m_state.root(), Verification::Skip);
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

	m_committedToMine = false;
	return *this;
}

StateDiff State::diff(State const& _c, bool _quick) const
{
	StateDiff ret;

	std::unordered_set<Address> ads;
	std::unordered_set<Address> trieAds;
	std::unordered_set<Address> trieAdsD;

	auto trie = SecureTrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&m_db), rootHash());
	auto trieD = SecureTrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&_c.m_db), _c.rootHash());

	if (_quick)
	{
		trieAds = m_touched;
		trieAdsD = _c.m_touched;
		(ads += m_touched) += _c.m_touched;
	}
	else
	{
		for (auto const& i: trie)
			ads.insert(i.first), trieAds.insert(i.first);
		for (auto const& i: trieD)
			ads.insert(i.first), trieAdsD.insert(i.first);
	}

	for (auto const& i: m_cache)
		ads.insert(i.first);
	for (auto const& i: _c.m_cache)
		ads.insert(i.first);

	for (auto const& i: ads)
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

void State::ensureCached(Address const& _a, bool _requireCode, bool _forceCreate) const
{
	ensureCached(m_cache, _a, _requireCode, _forceCreate);
}

void State::ensureCached(std::unordered_map<Address, Account>& _cache, const Address& _a, bool _requireCode, bool _forceCreate) const
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
			s = Account(state[0].toInt<u256>(), state[1].toInt<u256>(), state[2].toHash<h256>(), state[3].toHash<h256>(), Account::Unchanged);
		bool ok;
		tie(it, ok) = _cache.insert(make_pair(_a, s));
	}
	if (_requireCode && it != _cache.end() && !it->second.isFreshCode() && !it->second.codeCacheValid())
		it->second.noteCode(it->second.codeHash() == EmptySHA3 ? bytesConstRef() : bytesConstRef(m_db.lookup(it->second.codeHash())));
}

void State::commit()
{
	m_touched += dev::eth::commit(m_cache, m_state);
	m_cache.clear();
}

bool State::sync(BlockChain const& _bc)
{
	return sync(_bc, _bc.currentHash());
}

bool State::sync(BlockChain const& _bc, h256 const& _block, BlockInfo const& _bi)
{
	bool ret = false;
	// BLOCK
	BlockInfo bi = _bi ? _bi : _bc.info(_block);
#if ETH_PARANOIA
	if (!bi)
		while (1)
		{
			try
			{
				auto b = _bc.block(_block);
				bi.populate(b);
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
#endif
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

		if (m_db.lookup(bi.stateRoot()).empty())
		{
			cwarn << "Unable to sync to" << bi.hash() << "; state root" << bi.stateRoot() << "not found in database.";
			cwarn << "Database corrupt: contains block without stateRoot:" << bi;
			cwarn << "Try rescuing the database by running: eth --rescue";
			BOOST_THROW_EXCEPTION(InvalidStateRoot() << errinfo_target(bi.stateRoot()));
		}
		m_previousBlock = bi;
		resetCurrent();
		ret = true;
	}
#if ALLOW_REBUILD
	else
	{
		// New blocks available, or we've switched to a different branch. All change.
		// Find most recent state dump and replay what's left.
		// (Most recent state dump might end up being genesis.)

		std::vector<h256> chain;
		while (bi.number() != 0 && m_db.lookup(bi.stateRoot()).empty())	// while we don't have the state root of the latest block...
		{
			chain.push_back(bi.hash());				// push back for later replay.
			bi.populate(_bc.block(bi.parentHash()));	// move to parent.
		}

		m_previousBlock = bi;
		resetCurrent();

		// Iterate through in reverse, playing back each of the blocks.
		try
		{
			for (auto it = chain.rbegin(); it != chain.rend(); ++it)
			{
				auto b = _bc.block(*it);
				enact(&b, _bc, _ir);
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
#endif
	return ret;
}

u256 State::enactOn(VerifiedBlockRef const& _block, BlockChain const& _bc)
{
#if ETH_TIMED_ENACTMENTS
	Timer t;
	double populateVerify;
	double populateGrand;
	double syncReset;
	double enactment;
#endif

	// Check family:
	BlockInfo biParent = _bc.info(_block.info.parentHash());
	_block.info.verifyParent(biParent);

#if ETH_TIMED_ENACTMENTS
	populateVerify = t.elapsed();
	t.restart();
#endif

	BlockInfo biGrandParent;
	if (biParent.number())
		biGrandParent = _bc.info(biParent.parentHash());

#if ETH_TIMED_ENACTMENTS
	populateGrand = t.elapsed();
	t.restart();
#endif

	sync(_bc, _block.info.parentHash(), BlockInfo());
	resetCurrent();

#if ETH_TIMED_ENACTMENTS
	syncReset = t.elapsed();
	t.restart();
#endif

	m_previousBlock = biParent;
	auto ret = enact(_block, _bc);

#if ETH_TIMED_ENACTMENTS
	enactment = t.elapsed();
	if (populateVerify + populateGrand + syncReset + enactment > 0.5)
		clog(StateChat) << "popVer/popGrand/syncReset/enactment = " << populateVerify << "/" << populateGrand << "/" << syncReset << "/" << enactment;
#endif
	return ret;
}

unordered_map<Address, u256> State::addresses() const
{
#if ETH_FATDB
	unordered_map<Address, u256> ret;
	for (auto i: m_cache)
		if (i.second.isAlive())
			ret[i.first] = i.second.balance();
	for (auto const& i: m_state)
		if (m_cache.find(i.first) == m_cache.end())
			ret[i.first] = RLP(i.second)[1].toInt<u256>();
	return ret;
#else
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("State::addresses()"));
#endif
}

void State::resetCurrent()
{
	m_transactions.clear();
	m_receipts.clear();
	m_transactionSet.clear();
	m_cache.clear();
	m_touched.clear();
	m_currentBlock = BlockInfo();
	m_currentBlock.setCoinbaseAddress(m_ourAddress);
	m_currentBlock.setTimestamp(max(m_previousBlock.timestamp() + 1, (u256)time(0)));
	m_currentBlock.populateFromParent(m_previousBlock);

	// TODO: check.

	m_lastTx = m_db;
	m_state.setRoot(m_previousBlock.stateRoot());

	m_committedToMine = false;

	paranoia("begin resetCurrent", true);
}

pair<TransactionReceipts, bool> State::sync(BlockChain const& _bc, TransactionQueue& _tq, GasPricer const& _gp, unsigned msTimeout)
{
	// TRANSACTIONS
	pair<TransactionReceipts, bool> ret;
	ret.second = false;

	auto ts = _tq.topTransactions(c_maxSyncTransactions);

	LastHashes lh;

	auto deadline =  chrono::steady_clock::now() + chrono::milliseconds(msTimeout);

	for (int goodTxs = 1; goodTxs; )
	{
		goodTxs = 0;
		for (auto const& t: ts)
			if (!m_transactionSet.count(t.sha3()))
			{
				try
				{
					if (t.gasPrice() >= _gp.ask(*this))
					{
//						Timer t;
						if (lh.empty())
							lh = _bc.lastHashes();
						execute(lh, t);
						ret.first.push_back(m_receipts.back());
						++goodTxs;
//						cnote << "TX took:" << t.elapsed() * 1000;
					}
					else if (t.gasPrice() < _gp.ask(*this) * 9 / 10)
					{
						clog(StateTrace) << t.sha3() << "Dropping El Cheapo transaction (<90% of ask price)";
						_tq.drop(t.sha3());
					}
				}
				catch (InvalidNonce const& in)
				{
					bigint const& req = *boost::get_error_info<errinfo_required>(in);
					bigint const& got = *boost::get_error_info<errinfo_got>(in);

					if (req > got)
					{
						// too old
						clog(StateTrace) << t.sha3() << "Dropping old transaction (nonce too low)";
						_tq.drop(t.sha3());
					}
					else if (got > req + _tq.waiting(t.sender()))
					{
						// too new
						clog(StateTrace) << t.sha3() << "Dropping new transaction (too many nonces ahead)";
						_tq.drop(t.sha3());
					}
					else
						_tq.setFuture(t.sha3());
				}
				catch (BlockGasLimitReached const& e)
				{
					bigint const& got = *boost::get_error_info<errinfo_got>(e);
					if (got > m_currentBlock.gasLimit())
					{
						clog(StateTrace) << t.sha3() << "Dropping over-gassy transaction (gas > block's gas limit)";
						_tq.drop(t.sha3());
					}
					else
					{
						// Temporarily no gas left in current block.
						// OPTIMISE: could note this and then we don't evaluate until a block that does have the gas left.
						// for now, just leave alone.
					}
				}
				catch (Exception const& _e)
				{
					// Something else went wrong - drop it.
					clog(StateTrace) << t.sha3() << "Dropping invalid transaction:" << diagnostic_information(_e);
					_tq.drop(t.sha3());
				}
				catch (std::exception const&)
				{
					// Something else went wrong - drop it.
					_tq.drop(t.sha3());
					cwarn << t.sha3() << "Transaction caused low-level exception :(";
				}
			}
		if (chrono::steady_clock::now() > deadline)
		{
			ret.second = true;
			break;
		}
	}
	return ret;
}

string State::vmTrace(bytesConstRef _block, BlockChain const& _bc, ImportRequirements::value _ir)
{
	RLP rlp(_block);

	cleanup(false);
	BlockInfo bi(_block, (_ir & ImportRequirements::ValidSeal) ? CheckEverything : IgnoreSeal);
	m_currentBlock = bi;
	m_currentBlock.verifyInternals(_block);
	m_currentBlock.noteDirty();

	LastHashes lh = _bc.lastHashes((unsigned)m_previousBlock.number());

	string ret;
	unsigned i = 0;
	for (auto const& tr: rlp[1])
	{
		StandardTrace st;
		st.setShowMnemonics();
		execute(lh, Transaction(tr.data(), CheckTransaction::Everything), Permanence::Committed, st.onOp());
		ret += (ret.empty() ? "[" : ",") + st.json();
		++i;
	}
	return ret.empty() ? "[]" : (ret + "]");
}

u256 State::enact(VerifiedBlockRef const& _block, BlockChain const& _bc)
{
	DEV_TIMED_FUNCTION_ABOVE(500);

	// m_currentBlock is assumed to be prepopulated and reset.
#if !ETH_RELEASE
	assert(m_previousBlock.hash() == _block.info.parentHash());
	assert(m_currentBlock.parentHash() == _block.info.parentHash());
	assert(rootHash() == m_previousBlock.stateRoot());
#endif

	if (m_currentBlock.parentHash() != m_previousBlock.hash())
		// Internal client error.
		BOOST_THROW_EXCEPTION(InvalidParentHash());

	// Populate m_currentBlock with the correct values.
	m_currentBlock.noteDirty();
	m_currentBlock = _block.info;

//	cnote << "playback begins:" << m_state.root();
//	cnote << m_state;

	LastHashes lh;
	DEV_TIMED_ABOVE("lastHashes", 500)
		lh = _bc.lastHashes((unsigned)m_previousBlock.number());

	RLP rlp(_block.block);

	vector<bytes> receipts;

	// All ok with the block generally. Play back the transactions now...
	unsigned i = 0;
	DEV_TIMED_ABOVE("txExec", 500)
		for (auto const& tr: _block.transactions)
		{
			try
			{
				LogOverride<ExecutiveWarnChannel> o(false);
				execute(lh, tr);
			}
			catch (Exception& ex)
			{
				ex << errinfo_transactionIndex(i);
				throw;
			}

			RLPStream receiptRLP;
			m_receipts.back().streamRLP(receiptRLP);
			receipts.push_back(receiptRLP.out());
			++i;
		}

	h256 receiptsRoot;
	DEV_TIMED_ABOVE(".receiptsRoot()", 500)
		receiptsRoot = orderedTrieRoot(receipts);

	if (receiptsRoot != m_currentBlock.receiptsRoot())
	{
		InvalidReceiptsStateRoot ex;
		ex << Hash256RequirementError(receiptsRoot, m_currentBlock.receiptsRoot());
		ex << errinfo_receipts(receipts);
//		ex << errinfo_vmtrace(vmTrace(_block.block, _bc, ImportRequirements::None));
		BOOST_THROW_EXCEPTION(ex);
	}

	if (m_currentBlock.logBloom() != logBloom())
	{
		InvalidLogBloom ex;
		ex << LogBloomRequirementError(logBloom(), m_currentBlock.logBloom());
		ex << errinfo_receipts(receipts);
		BOOST_THROW_EXCEPTION(ex);
	}

	// Initialise total difficulty calculation.
	u256 tdIncrease = m_currentBlock.difficulty();

	// Check uncles & apply their rewards to state.
	if (rlp[2].itemCount() > 2)
	{
		TooManyUncles ex;
		ex << errinfo_max(2);
		ex << errinfo_got(rlp[2].itemCount());
		BOOST_THROW_EXCEPTION(ex);
	}

	vector<BlockInfo> rewarded;
	h256Hash excluded;
	DEV_TIMED_ABOVE("allKin", 500)
		excluded = _bc.allKinFrom(m_currentBlock.parentHash(), 6);
	excluded.insert(m_currentBlock.hash());

	unsigned ii = 0;
	DEV_TIMED_ABOVE("uncleCheck", 500)
		for (auto const& i: rlp[2])
		{
			try
			{
				auto h = sha3(i.data());
				if (excluded.count(h))
				{
					UncleInChain ex;
					ex << errinfo_comment("Uncle in block already mentioned");
					ex << errinfo_unclesExcluded(excluded);
					ex << errinfo_hash256(sha3(i.data()));
					BOOST_THROW_EXCEPTION(ex);
				}
				excluded.insert(h);

				// IgnoreSeal since it's a VerifiedBlock.
				BlockInfo uncle(i.data(), IgnoreSeal, h, HeaderData);

				BlockInfo uncleParent;
				if (!_bc.isKnown(uncle.parentHash()))
					BOOST_THROW_EXCEPTION(UnknownParent());
				uncleParent = BlockInfo(_bc.block(uncle.parentHash()));

				if ((bigint)uncleParent.number() < (bigint)m_currentBlock.number() - 7)
				{
					UncleTooOld ex;
					ex << errinfo_uncleNumber(uncle.number());
					ex << errinfo_currentNumber(m_currentBlock.number());
					BOOST_THROW_EXCEPTION(ex);
				}
				else if (uncle.number() == m_currentBlock.number())
				{
					UncleIsBrother ex;
					ex << errinfo_uncleNumber(uncle.number());
					ex << errinfo_currentNumber(m_currentBlock.number());
					BOOST_THROW_EXCEPTION(ex);
				}
				uncle.verifyParent(uncleParent);

				rewarded.push_back(uncle);
				++ii;
			}
			catch (Exception& ex)
			{
				ex << errinfo_uncleIndex(ii);
				throw;
			}
		}

	DEV_TIMED_ABOVE("applyRewards", 500)
		applyRewards(rewarded);

	// Commit all cached state changes to the state trie.
	DEV_TIMED_ABOVE("commit", 500)
		commit();

	// Hash the state trie and check against the state_root hash in m_currentBlock.
	if (m_currentBlock.stateRoot() != m_previousBlock.stateRoot() && m_currentBlock.stateRoot() != rootHash())
	{
		auto r = rootHash();
		m_db.rollback();
		BOOST_THROW_EXCEPTION(InvalidStateRoot() << Hash256RequirementError(r, m_currentBlock.stateRoot()));
	}

	if (m_currentBlock.gasUsed() != gasUsed())
	{
		// Rollback the trie.
		m_db.rollback();
		BOOST_THROW_EXCEPTION(InvalidGasUsed() << RequirementError(bigint(gasUsed()), bigint(m_currentBlock.gasUsed())));
	}

	return tdIncrease;
}

void State::cleanup(bool _fullCommit)
{
	if (_fullCommit)
	{
		paranoia("immediately before database commit", true);

		// Commit the new trie to disk.
		if (isChannelVisible<StateTrace>()) // Avoid calling toHex if not needed
			clog(StateTrace) << "Committing to disk: stateRoot" << m_currentBlock.stateRoot() << "=" << rootHash() << "=" << toHex(asBytes(m_db.lookup(rootHash())));

		try
		{
			EnforceRefs er(m_db, true);
			rootHash();
		}
		catch (BadRoot const&)
		{
			clog(StateChat) << "Trie corrupt! :-(";
			throw;
		}

		m_db.commit();
		if (isChannelVisible<StateTrace>()) // Avoid calling toHex if not needed
			clog(StateTrace) << "Committed: stateRoot" << m_currentBlock.stateRoot() << "=" << rootHash() << "=" << toHex(asBytes(m_db.lookup(rootHash())));

		paranoia("immediately after database commit", true);
		m_previousBlock = m_currentBlock;
		m_currentBlock.populateFromParent(m_previousBlock);

		clog(StateTrace) << "finalising enactment. current -> previous, hash is" << m_previousBlock.hash();
	}
	else
		m_db.rollback();

	resetCurrent();
}

void State::uncommitToMine()
{
	if (m_committedToMine)
	{
		m_cache.clear();
		if (!m_transactions.size())
			m_state.setRoot(m_previousBlock.stateRoot());
		else
			m_state.setRoot(m_receipts.back().stateRoot());
		m_db = m_lastTx;
		paranoia("Uncommited to mine", true);
		m_committedToMine = false;
	}
}

LogBloom State::logBloom() const
{
	LogBloom ret;
	for (TransactionReceipt const& i: m_receipts)
		ret |= i.bloom();
	return ret;
}

void State::commitToMine(BlockChain const& _bc, bytes const& _extraData)
{
	uncommitToMine();

	m_lastTx = m_db;

	vector<BlockInfo> uncleBlockHeaders;

	RLPStream unclesData;
	unsigned unclesCount = 0;
	if (m_previousBlock.number() != 0)
	{
		// Find great-uncles (or second-cousins or whatever they are) - children of great-grandparents, great-great-grandparents... that were not already uncles in previous generations.
		clog(StateDetail) << "Checking " << m_previousBlock.hash() << ", parent=" << m_previousBlock.parentHash();
		h256Hash excluded = _bc.allKinFrom(m_currentBlock.parentHash(), 6);
		auto p = m_previousBlock.parentHash();
		for (unsigned gen = 0; gen < 6 && p != _bc.genesisHash() && unclesCount < 2; ++gen, p = _bc.details(p).parent)
		{
			auto us = _bc.details(p).children;
			assert(us.size() >= 1);	// must be at least 1 child of our grandparent - it's our own parent!
			for (auto const& u: us)
				if (!excluded.count(u))	// ignore any uncles/mainline blocks that we know about.
				{
					uncleBlockHeaders.push_back(_bc.info(u));
					unclesData.appendRaw(_bc.headerData(u));
					++unclesCount;
					if (unclesCount == 2)
						break;
				}
		}
	}

	BytesMap transactionsMap;
	BytesMap receiptsMap;

	RLPStream txs;
	txs.appendList(m_transactions.size());

	for (unsigned i = 0; i < m_transactions.size(); ++i)
	{
		RLPStream k;
		k << i;

		RLPStream receiptrlp;
		m_receipts[i].streamRLP(receiptrlp);
		receiptsMap.insert(std::make_pair(k.out(), receiptrlp.out()));

		RLPStream txrlp;
		m_transactions[i].streamRLP(txrlp);
		transactionsMap.insert(std::make_pair(k.out(), txrlp.out()));

		txs.appendRaw(txrlp.out());
	}

	txs.swapOut(m_currentTxs);

	RLPStream(unclesCount).appendRaw(unclesData.out(), unclesCount).swapOut(m_currentUncles);

	// Apply rewards last of all.
	applyRewards(uncleBlockHeaders);

	// Commit any and all changes to the trie that are in the cache, then update the state root accordingly.
	commit();

	clog(StateDetail) << "Post-reward stateRoot:" << m_state.root();
	clog(StateDetail) << m_state;
	clog(StateDetail) << *this;

	m_currentBlock.setLogBloom(logBloom());
	m_currentBlock.setGasUsed(gasUsed());
	m_currentBlock.setRoots(hash256(transactionsMap), hash256(receiptsMap), sha3(m_currentUncles), m_state.root());

	m_currentBlock.setParentHash(m_previousBlock.hash());
	m_currentBlock.setExtraData(_extraData);
	if (m_currentBlock.extraData().size() > 32)
	{
		auto ed = m_currentBlock.extraData();
		ed.resize(32);
		m_currentBlock.setExtraData(ed);
	}

	m_committedToMine = true;
}

bool State::sealBlock(bytesConstRef _header)
{
	if (!m_committedToMine)
		return false;

	// Check that this header is indeed for this block.
	if (BlockInfo(_header, CheckNothing, h256{}, HeaderData).hashWithout() != m_currentBlock.hashWithout())
		return false;

	// Looks good!
	clog(StateDetail) << "Sealing block!";

	// Compile block:
	RLPStream ret;
	ret.appendList(3);
	ret.appendRaw(_header);
	ret.appendRaw(m_currentTxs);
	ret.appendRaw(m_currentUncles);
	ret.swapOut(m_currentBytes);
	m_currentBlock = BlockInfo(_header, CheckNothing, h256(), HeaderData);
	cnote << "Mined " << m_currentBlock.hash() << "(parent: " << m_currentBlock.parentHash() << ")";
	// TODO: move into Sealer
	StructuredLogger::minedNewBlock(
		m_currentBlock.hash().abridged(),
		"",	// Can't give the nonce here.
		"", //TODO: chain head hash here ??
		m_currentBlock.parentHash().abridged()
	);

	// Quickly reset the transactions.
	// TODO: Leave this in a better state than this limbo, or at least record that it's in limbo.
	m_transactions.clear();
	m_receipts.clear();
	m_transactionSet.clear();
	m_lastTx = m_db;

	return true;
}

bool State::addressInUse(Address const& _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return true;
}

bool State::addressHasCode(Address const& _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return false;
	return it->second.isFreshCode() || it->second.codeHash() != EmptySHA3;
}

u256 State::balance(Address const& _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return 0;
	return it->second.balance();
}

void State::noteSending(Address const& _id)
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

void State::addBalance(Address const& _id, u256 const& _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		m_cache[_id] = Account(_amount, Account::NormalCreation);
	else
		it->second.addBalance(_amount);
}

void State::subBalance(Address const& _id, bigint const& _amount)
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end() || (bigint)it->second.balance() < _amount)
		BOOST_THROW_EXCEPTION(NotEnoughCash());
	else
		it->second.addBalance(-_amount);
}

Address State::newContract(u256 const& _balance, bytes const& _code)
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
			m_cache[ret] = Account(0, _balance, EmptyTrie, h, Account::Changed);
			return ret;
		}
	}
}

u256 State::transactionsFrom(Address const& _id) const
{
	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it == m_cache.end())
		return 0;
	else
		return it->second.nonce();
}

u256 State::storage(Address const& _id, u256 const& _memory) const
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
	SecureTrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&m_db), it->second.baseRoot());			// promise we won't change the overlay! :)
	string payload = memdb.at(_memory);
	u256 ret = payload.size() ? RLP(payload).toInt<u256>() : 0;
	it->second.setStorage(_memory, ret);
	return ret;
}

unordered_map<u256, u256> State::storage(Address const& _id) const
{
	unordered_map<u256, u256> ret;

	ensureCached(_id, false, false);
	auto it = m_cache.find(_id);
	if (it != m_cache.end())
	{
		// Pull out all values from trie storage.
		if (it->second.baseRoot())
		{
			SecureTrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&m_db), it->second.baseRoot());		// promise we won't alter the overlay! :)
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

h256 State::storageRoot(Address const& _id) const
{
	string s = m_state.at(_id);
	if (s.size())
	{
		RLP r(s);
		return r[2].toHash<h256>();
	}
	return EmptyTrie;
}

bytes const& State::code(Address const& _contract) const
{
	if (!addressHasCode(_contract))
		return NullBytes;
	ensureCached(_contract, true, false);
	return m_cache[_contract].code();
}

h256 State::codeHash(Address const& _contract) const
{
	if (!addressHasCode(_contract))
		return EmptySHA3;
	if (m_cache[_contract].isFreshCode())
		return sha3(code(_contract));
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
/*			for (auto const& i: m_state)
			{
				RLP r(i.second);
				SecureTrieDB<h256, OverlayDB> storageDB(const_cast<OverlayDB*>(&m_db), r[2].toHash<h256>());	// promise not to alter OverlayDB.
				for (auto const& j: storageDB) { (void)j; }
				if (!e && r[3].toHash<h256>() != EmptySHA3 && m_db.lookup(r[3].toHash<h256>()).empty())
					return false;
			}*/
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

ExecutionResult State::execute(LastHashes const& _lh, Transaction const& _t, Permanence _p, OnOpFunc const& _onOp)
{
	auto onOp = _onOp;
#if ETH_VMTRACE
	if (isChannelVisible<VMTraceChannel>())
		onOp = Executive::simpleTrace(); // override tracer
#endif

#if ETH_PARANOIA
	paranoia("start of execution.", true);
	State old(*this);
	auto h = rootHash();
#endif

	// Create and initialize the executive. This will throw fairly cheaply and quickly if the
	// transaction is bad in any way.
	Executive e(*this, _lh, 0);
	ExecutionResult res;
	e.setResultRecipient(res);
	e.initialize(_t);

	// Uncommitting is a non-trivial operation - only do it once we've verified as much of the
	// transaction as possible.
	uncommitToMine();

	// OK - transaction looks valid - execute.
	u256 startGasUsed = gasUsed();
#if ETH_PARANOIA
	ctrace << "Executing" << e.t() << "on" << h;
	ctrace << toHex(e.t().rlp());
#endif
	if (!e.execute())
		e.go(onOp);
	e.finalize();

#if ETH_PARANOIA
	ctrace << "Ready for commit;";
	ctrace << old.diff(*this);
#endif

	if (_p == Permanence::Reverted)
		m_cache.clear();
	else
	{
		commit();

#if ETH_PARANOIA && !ETH_FATDB
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
	}

	return res;
}

State State::fromPending(unsigned _i) const
{
	State ret = *this;
	ret.m_cache.clear();
	_i = min<unsigned>(_i, m_transactions.size());
	if (!_i)
		ret.m_state.setRoot(m_previousBlock.stateRoot());
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

void State::applyRewards(vector<BlockInfo> const& _uncleBlockHeaders)
{
	u256 r = m_blockReward;
	for (auto const& i: _uncleBlockHeaders)
	{
		addBalance(i.coinbaseAddress(), m_blockReward * (8 + i.number() - m_currentBlock.number()) / 8);
		r += m_blockReward / 32;
	}
	addBalance(m_currentBlock.coinbaseAddress(), r);
}

std::ostream& dev::eth::operator<<(std::ostream& _out, State const& _s)
{
	_out << "--- " << _s.rootHash() << std::endl;
	std::set<Address> d;
	std::set<Address> dtr;
	auto trie = SecureTrieDB<Address, OverlayDB>(const_cast<OverlayDB*>(&_s.m_db), _s.rootHash());
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
					SecureTrieDB<h256, OverlayDB> memdb(const_cast<OverlayDB*>(&_s.m_db), r[2].toHash<h256>());		// promise we won't alter the overlay! :)
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
				if (!delta.empty())
					lead = (lead == " .   ") ? "*.*  " : "***  ";

				contout << " @:";
				if (!delta.empty())
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
