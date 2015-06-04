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
/** @file BlockChain.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "BlockChain.h"

#if ETH_PROFILING_GPERF
#include <gperftools/profiler.h>
#endif
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <test/JsonSpiritHeaders.h>
#include <libdevcore/Common.h>
#include <libdevcore/Assertions.h>
#include <libdevcore/RLP.h>
#include <libdevcore/StructuredLogger.h>
#include <libdevcore/FileSystem.h>
#include <libethcore/Exceptions.h>
#include <libethcore/ProofOfWork.h>
#include <libethcore/BlockInfo.h>
#include <libethcore/Params.h>
#include <liblll/Compiler.h>
#include "GenesisInfo.h"
#include "State.h"
#include "Defaults.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
namespace js = json_spirit;

#define ETH_CATCH 1
#define ETH_TIMED_IMPORTS 0

#ifdef _WIN32
const char* BlockChainDebug::name() { return EthBlue "8" EthWhite " <>"; }
const char* BlockChainWarn::name() { return EthBlue "8" EthOnRed EthBlackBold " X"; }
const char* BlockChainNote::name() { return EthBlue "8" EthBlue " i"; }
const char* BlockChainChat::name() { return EthBlue "8" EthWhite " o"; }
#else
const char* BlockChainDebug::name() { return EthBlue "☍" EthWhite " ◇"; }
const char* BlockChainWarn::name() { return EthBlue "☍" EthOnRed EthBlackBold " ✘"; }
const char* BlockChainNote::name() { return EthBlue "☍" EthBlue " ℹ"; }
const char* BlockChainChat::name() { return EthBlue "☍" EthWhite " ◌"; }
#endif

std::ostream& dev::eth::operator<<(std::ostream& _out, BlockChain const& _bc)
{
	string cmp = toBigEndianString(_bc.currentHash());
	auto it = _bc.m_blocksDB->NewIterator(_bc.m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().ToString() != "best")
		{
			try {
				BlockInfo d(bytesConstRef(it->value()));
				_out << toHex(it->key().ToString()) << ":   " << d.number << " @ " << d.parentHash << (cmp == it->key().ToString() ? "  BEST" : "") << std::endl;
			}
			catch (...) {
				cwarn << "Invalid DB entry:" << toHex(it->key().ToString()) << " -> " << toHex(bytesConstRef(it->value()));
			}
		}
	delete it;
	return _out;
}

ldb::Slice dev::eth::toSlice(h256 const& _h, unsigned _sub)
{
#if ALL_COMPILERS_ARE_CPP11_COMPLIANT
	static thread_local h256 h = _h ^ sha3(h256(u256(_sub)));
	return ldb::Slice((char const*)&h, 32);
#else
	static boost::thread_specific_ptr<FixedHash<33>> t_h;
	if (!t_h.get())
		t_h.reset(new FixedHash<33>);
	*t_h = FixedHash<33>(_h);
	(*t_h)[32] = (uint8_t)_sub;
	return (ldb::Slice)t_h->ref();//(char const*)t_h.get(), 32);
#endif
}

#if ETH_DEBUG&&0
static const chrono::system_clock::duration c_collectionDuration = chrono::seconds(15);
static const unsigned c_collectionQueueSize = 2;
static const unsigned c_maxCacheSize = 1024 * 1024 * 1;
static const unsigned c_minCacheSize = 1;
#else

/// Duration between flushes.
static const chrono::system_clock::duration c_collectionDuration = chrono::seconds(60);

/// Length of death row (total time in cache is multiple of this and collection duration).
static const unsigned c_collectionQueueSize = 20;

/// Max size, above which we start forcing cache reduction.
static const unsigned c_maxCacheSize = 1024 * 1024 * 64;

/// Min size, below which we don't bother flushing it.
static const unsigned c_minCacheSize = 1024 * 1024 * 32;

#endif

BlockChain::BlockChain(bytes const& _genesisBlock, std::string _path, WithExisting _we, ProgressCallback const& _p)
{
	// initialise deathrow.
	m_cacheUsage.resize(c_collectionQueueSize);
	m_lastCollection = chrono::system_clock::now();

	// Initialise with the genesis as the last block on the longest chain.
	m_genesisBlock = _genesisBlock;
	m_genesisHash = sha3(RLP(m_genesisBlock)[0].data());

	open(_path, _we);
	if (_we == WithExisting::Verify)
		rebuild(_path, _p);
}

BlockChain::~BlockChain()
{
	close();
}

void BlockChain::open(std::string const& _path, WithExisting _we)
{
	std::string path = _path.empty() ? Defaults::get()->m_dbPath : _path;
	boost::filesystem::create_directories(path);
	if (_we == WithExisting::Kill)
	{
		boost::filesystem::remove_all(path + "/blocks");
		boost::filesystem::remove_all(path + "/details");
	}

	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path + "/blocks", &m_blocksDB);
	ldb::DB::Open(o, path + "/details", &m_extrasDB);
	if (!m_blocksDB || !m_extrasDB)
	{
		if (boost::filesystem::space(path + "/blocks").available < 1024)
		{
			cwarn << "Not enough available space found on hard drive. Please free some up and then re-run. Bailing.";
			BOOST_THROW_EXCEPTION(NotEnoughAvailableSpace());
		}
		else
		{
			cwarn << "Database already open. You appear to have another instance of ethereum running. Bailing.";
			BOOST_THROW_EXCEPTION(DatabaseAlreadyOpen());
		}
	}

	if (_we != WithExisting::Verify && !details(m_genesisHash))
	{
		// Insert details of genesis block.
		m_details[m_genesisHash] = BlockDetails(0, c_genesisDifficulty, h256(), {});
		auto r = m_details[m_genesisHash].rlp();
		m_extrasDB->Put(m_writeOptions, toSlice(m_genesisHash, ExtraDetails), (ldb::Slice)dev::ref(r));
	}

#if ETH_PARANOIA
	checkConsistency();
#endif

	// TODO: Implement ability to rebuild details map from DB.
	std::string l;
	m_extrasDB->Get(m_readOptions, ldb::Slice("best"), &l);
	m_lastBlockHash = l.empty() ? m_genesisHash : *(h256*)l.data();
	m_lastBlockNumber = number(m_lastBlockHash);

	cnote << "Opened blockchain DB. Latest: " << currentHash();
}

void BlockChain::close()
{
	cnote << "Closing blockchain DB";
	delete m_extrasDB;
	delete m_blocksDB;
	m_lastBlockHash = m_genesisHash;
	m_lastBlockNumber = 0;
	m_details.clear();
	m_blocks.clear();
}

#define IGNORE_EXCEPTIONS(X) try { X; } catch (...) {}

void BlockChain::rebuild(std::string const& _path, std::function<void(unsigned, unsigned)> const& _progress, bool _prepPoW)
{
	std::string path = _path.empty() ? Defaults::get()->m_dbPath : _path;

#if ETH_PROFILING_GPERF
	ProfilerStart("BlockChain_rebuild.log");
#endif

	unsigned originalNumber = m_lastBlockNumber;

	// Keep extras DB around, but under a temp name
	delete m_extrasDB;
	m_extrasDB = nullptr;
	IGNORE_EXCEPTIONS(boost::filesystem::remove_all(path + "/details.old"));
	boost::filesystem::rename(path + "/details", path + "/details.old");
	ldb::DB* oldExtrasDB;
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path + "/details.old", &oldExtrasDB);
	ldb::DB::Open(o, path + "/details", &m_extrasDB);

	// Open a fresh state DB
	State s(State::openDB(path, WithExisting::Kill), BaseState::CanonGenesis);

	// Clear all memos ready for replay.
	m_details.clear();
	m_logBlooms.clear();
	m_receipts.clear();
	m_transactionAddresses.clear();
	m_blockHashes.clear();
	m_blocksBlooms.clear();
	m_lastLastHashes.clear();
	m_lastBlockHash = genesisHash();
	m_lastBlockNumber = 0;

	m_details[m_lastBlockHash].totalDifficulty = c_genesisDifficulty;

	m_extrasDB->Put(m_writeOptions, toSlice(m_lastBlockHash, ExtraDetails), (ldb::Slice)dev::ref(m_details[m_lastBlockHash].rlp()));

	h256 lastHash = m_lastBlockHash;
	boost::timer t;
	for (unsigned d = 1; d < originalNumber; ++d)
	{
		if (!(d % 1000))
		{
			cerr << "\n1000 blocks in " << t.elapsed() << "s = " << (1000.0 / t.elapsed()) << "b/s" << endl;
			t.restart();
		}
		try
		{
			bytes b = block(queryExtras<BlockHash, ExtraBlockHash>(h256(u256(d)), m_blockHashes, x_blockHashes, NullBlockHash, oldExtrasDB).value);

			BlockInfo bi(b);
			if (_prepPoW)
				ProofOfWork::prep(bi);

			if (bi.parentHash != lastHash)
			{
				cwarn << "DISJOINT CHAIN DETECTED; " << bi.hash() << "#" << d << " -> parent is" << bi.parentHash << "; expected" << lastHash << "#" << (d - 1);
				return;
			}
			lastHash = bi.hash();
			import(b, s.db(), 0);
		}
		catch (...)
		{
			// Failed to import - stop here.
			break;
		}

		if (_progress)
			_progress(d, originalNumber);
	}

#if ETH_PROFILING_GPERF
	ProfilerStop();
#endif

	delete oldExtrasDB;
	boost::filesystem::remove_all(path + "/details.old");
}

template <class T, class V>
bool contains(T const& _t, V const& _v)
{
	for (auto const& i: _t)
		if (i == _v)
			return true;
	return false;
}

LastHashes BlockChain::lastHashes(unsigned _n) const
{
	Guard l(x_lastLastHashes);
	if (m_lastLastHashesNumber != _n || m_lastLastHashes.empty())
	{
		m_lastLastHashes.resize(256);
		for (unsigned i = 0; i < 256; ++i)
			m_lastLastHashes[i] = _n >= i ? numberHash(_n - i) : h256();
		m_lastLastHashesNumber = _n;
	}
	return m_lastLastHashes;
}

tuple<h256s, h256s, bool> BlockChain::sync(BlockQueue& _bq, OverlayDB const& _stateDB, unsigned _max)
{
//	_bq.tick(*this);

	vector<pair<BlockInfo, bytes>> blocks;
	_bq.drain(blocks, _max);

	h256s fresh;
	h256s dead;
	h256s badBlocks;
	for (auto const& block: blocks)
	{
		try
		{
			// Nonce & uncle nonces already verified in verification thread at this point.
			ImportRoute r;
			DEV_TIMED_ABOVE(Block import, 500)
				r = import(block.first, block.second, _stateDB, ImportRequirements::Default & ~ImportRequirements::ValidNonce & ~ImportRequirements::CheckUncles);
			fresh += r.first;
			dead += r.second;
		}
		catch (dev::eth::UnknownParent)
		{
			cwarn << "ODD: Import queue contains block with unknown parent." << LogTag::Error << boost::current_exception_diagnostic_information();
			// NOTE: don't reimport since the queue should guarantee everything in the right order.
			// Can't continue - chain bad.
			badBlocks.push_back(block.first.hash());
		}
		catch (Exception const& _e)
		{
			cnote << "Exception while importing block. Someone (Jeff? That you?) seems to be giving us dodgy blocks!" << LogTag::Error << diagnostic_information(_e);
			// NOTE: don't reimport since the queue should guarantee everything in the right order.
			// Can't continue - chain  bad.
			badBlocks.push_back(block.first.hash());
		}
	}
	return make_tuple(fresh, dead, _bq.doneDrain(badBlocks));
}

pair<ImportResult, ImportRoute> BlockChain::attemptImport(bytes const& _block, OverlayDB const& _stateDB, ImportRequirements::value _ir) noexcept
{
	try
	{
		return make_pair(ImportResult::Success, import(_block, _stateDB, _ir));
	}
	catch (UnknownParent&)
	{
		return make_pair(ImportResult::UnknownParent, make_pair(h256s(), h256s()));
	}
	catch (AlreadyHaveBlock&)
	{
		return make_pair(ImportResult::AlreadyKnown, make_pair(h256s(), h256s()));
	}
	catch (FutureTime&)
	{
		return make_pair(ImportResult::FutureTime, make_pair(h256s(), h256s()));
	}
	catch (...)
	{
		return make_pair(ImportResult::Malformed, make_pair(h256s(), h256s()));
	}
}

ImportRoute BlockChain::import(bytes const& _block, OverlayDB const& _db, ImportRequirements::value _ir)
{
	// VERIFY: populates from the block and checks the block is internally coherent.
	BlockInfo bi;

#if ETH_CATCH
	try
#endif
	{
		bi.populate(&_block);
		bi.verifyInternals(&_block);
	}
#if ETH_CATCH
	catch (Exception const& _e)
	{
		clog(BlockChainNote) << "   Malformed block: " << diagnostic_information(_e);
		_e << errinfo_comment("Malformed block ");
		throw;
	}
#endif

	return import(bi, _block, _db, _ir);
}

ImportRoute BlockChain::import(BlockInfo const& _bi, bytes const& _block, OverlayDB const& _db, ImportRequirements::value _ir)
{
	//@tidy This is a behemoth of a method - could do to be split into a few smaller ones.

#if ETH_TIMED_IMPORTS
	boost::timer total;
	double preliminaryChecks;
	double enactment;
	double collation;
	double writing;
	double checkBest;
	boost::timer t;
#endif

	// Check block doesn't already exist first!
	if (isKnown(_bi.hash()) && (_ir & ImportRequirements::DontHave))
	{
		clog(BlockChainNote) << _bi.hash() << ": Not new.";
		BOOST_THROW_EXCEPTION(AlreadyHaveBlock());
	}

	// Work out its number as the parent's number + 1
	if (!isKnown(_bi.parentHash))
	{
		clog(BlockChainNote) << _bi.hash() << ": Unknown parent " << _bi.parentHash;
		// We don't know the parent (yet) - discard for now. It'll get resent to us if we find out about its ancestry later on.
		BOOST_THROW_EXCEPTION(UnknownParent());
	}

	auto pd = details(_bi.parentHash);
	if (!pd)
	{
		auto pdata = pd.rlp();
		clog(BlockChainDebug) << "Details is returning false despite block known:" << RLP(pdata);
		auto parentBlock = block(_bi.parentHash);
		clog(BlockChainDebug) << "isKnown:" << isKnown(_bi.parentHash);
		clog(BlockChainDebug) << "last/number:" << m_lastBlockNumber << m_lastBlockHash << _bi.number;
		clog(BlockChainDebug) << "Block:" << BlockInfo(parentBlock);
		clog(BlockChainDebug) << "RLP:" << RLP(parentBlock);
		clog(BlockChainDebug) << "DATABASE CORRUPTION: CRITICAL FAILURE";
		exit(-1);
	}

	// Check it's not crazy
	if (_bi.timestamp > (u256)time(0))
	{
		clog(BlockChainChat) << _bi.hash() << ": Future time " << _bi.timestamp << " (now at " << time(0) << ")";
		// Block has a timestamp in the future. This is no good.
		BOOST_THROW_EXCEPTION(FutureTime());
	}

	clog(BlockChainChat) << "Attempting import of " << _bi.hash() << "...";

#if ETH_TIMED_IMPORTS
	preliminaryChecks = t.elapsed();
	t.restart();
#endif

	ldb::WriteBatch blocksBatch;
	ldb::WriteBatch extrasBatch;
	h256 newLastBlockHash = currentHash();
	unsigned newLastBlockNumber = number();

	u256 td;
#if ETH_CATCH
	try
#endif
	{
		// Check transactions are valid and that they result in a state equivalent to our state_root.
		// Get total difficulty increase and update state, checking it.
		State s(_db);
		auto tdIncrease = s.enactOn(&_block, _bi, *this, _ir);

		BlockLogBlooms blb;
		BlockReceipts br;
		for (unsigned i = 0; i < s.pending().size(); ++i)
		{
			blb.blooms.push_back(s.receipt(i).bloom());
			br.receipts.push_back(s.receipt(i));
		}
		try {
			s.cleanup(true);
		}
		catch (BadRoot)
		{
			cwarn << "BadRoot error. Retrying import later.";
			BOOST_THROW_EXCEPTION(FutureTime());
		}

		td = pd.totalDifficulty + tdIncrease;

#if ETH_TIMED_IMPORTS
		enactment = t.elapsed();
		t.restart();
#endif

#if ETH_PARANOIA || !ETH_TRUE
		checkConsistency();
#endif

		// All ok - insert into DB

		// ensure parent is cached for later addition.
		// TODO: this is a bit horrible would be better refactored into an enveloping UpgradableGuard
		// together with an "ensureCachedWithUpdatableLock(l)" method.
		// This is safe in practice since the caches don't get flushed nearly often enough to be
		// done here.
		details(_bi.parentHash);
		DEV_WRITE_GUARDED(x_details)
			m_details[_bi.parentHash].children.push_back(_bi.hash());

#if ETH_TIMED_IMPORTS || !ETH_TRUE
		collation = t.elapsed();
		t.restart();
#endif

		blocksBatch.Put(toSlice(_bi.hash()), (ldb::Slice)ref(_block));
		DEV_READ_GUARDED(x_details)
			extrasBatch.Put(toSlice(_bi.parentHash, ExtraDetails), (ldb::Slice)dev::ref(m_details[_bi.parentHash].rlp()));

		extrasBatch.Put(toSlice(_bi.hash(), ExtraDetails), (ldb::Slice)dev::ref(BlockDetails((unsigned)pd.number + 1, td, _bi.parentHash, {}).rlp()));
		extrasBatch.Put(toSlice(_bi.hash(), ExtraLogBlooms), (ldb::Slice)dev::ref(blb.rlp()));
		extrasBatch.Put(toSlice(_bi.hash(), ExtraReceipts), (ldb::Slice)dev::ref(br.rlp()));

#if ETH_TIMED_IMPORTS || !ETH_TRUE
		writing = t.elapsed();
		t.restart();
#endif
	}
#if ETH_CATCH
	catch (InvalidNonce const& _e)
	{
		clog(BlockChainNote) << "   Malformed block: " << diagnostic_information(_e);
		_e << errinfo_comment("Malformed block ");
		throw;
	}
	catch (Exception const& _e)
	{
		clog(BlockChainWarn) << "   Malformed block: " << diagnostic_information(_e);
		_e << errinfo_comment("Malformed block ");
		clog(BlockChainWarn) << "Block: " << _bi.hash();
		clog(BlockChainWarn) << _bi;
		clog(BlockChainWarn) << "Block parent: " << _bi.parentHash;
		clog(BlockChainWarn) << BlockInfo(block(_bi.parentHash));
		throw;
	}
#endif

	StructuredLogger::chainReceivedNewBlock(
		_bi.headerHash(WithoutNonce).abridged(),
		_bi.nonce.abridged(),
		currentHash().abridged(),
		"", // TODO: remote id ??
		_bi.parentHash.abridged()
	);
	//	cnote << "Parent " << bi.parentHash << " has " << details(bi.parentHash).children.size() << " children.";

	h256s route;
	h256 common;
	// This might be the new best block...
	h256 last = currentHash();
	if (td > details(last).totalDifficulty)
	{
		// don't include bi.hash() in treeRoute, since it's not yet in details DB...
		// just tack it on afterwards.
		unsigned commonIndex;
		tie(route, common, commonIndex) = treeRoute(last, _bi.parentHash);
		route.push_back(_bi.hash());

		// Most of the time these two will be equal - only when we're doing a chain revert will they not be
		if (common != last)
			// If we are reverting previous blocks, we need to clear their blooms (in particular, to
			// rebuild any higher level blooms that they contributed to).
			clearBlockBlooms(number(common) + 1, number(last) + 1);

		// Go through ret backwards until hash != last.parent and update m_transactionAddresses, m_blockHashes
		for (auto i = route.rbegin(); i != route.rend() && *i != common; ++i)
		{
			BlockInfo tbi;
			if (*i == _bi.hash())
				tbi = _bi;
			else
				tbi = BlockInfo(block(*i));

			// Collate logs into blooms.
			h256s alteredBlooms;
			{
				LogBloom blockBloom = tbi.logBloom;
				blockBloom.shiftBloom<3>(sha3(tbi.coinbaseAddress.ref()));

				// Pre-memoize everything we need before locking x_blocksBlooms
				for (unsigned level = 0, index = (unsigned)tbi.number; level < c_bloomIndexLevels; level++, index /= c_bloomIndexSize)
					blocksBlooms(chunkId(level, index / c_bloomIndexSize));

				WriteGuard l(x_blocksBlooms);
				for (unsigned level = 0, index = (unsigned)tbi.number; level < c_bloomIndexLevels; level++, index /= c_bloomIndexSize)
				{
					unsigned i = index / c_bloomIndexSize;
					unsigned o = index % c_bloomIndexSize;
					alteredBlooms.push_back(chunkId(level, i));
					m_blocksBlooms[alteredBlooms.back()].blooms[o] |= blockBloom;
				}
			}
			// Collate transaction hashes and remember who they were.
			h256s newTransactionAddresses;
			{
				bytes blockBytes;
				RLP blockRLP(*i == _bi.hash() ? _block : (blockBytes = block(*i)));
				TransactionAddress ta;
				ta.blockHash = tbi.hash();
				for (ta.index = 0; ta.index < blockRLP[1].itemCount(); ++ta.index)
					extrasBatch.Put(toSlice(sha3(blockRLP[1][ta.index].data()), ExtraTransactionAddress), (ldb::Slice)dev::ref(ta.rlp()));
			}

			// Update database with them.
			ReadGuard l1(x_blocksBlooms);
			for (auto const& h: alteredBlooms)
				extrasBatch.Put(toSlice(h, ExtraBlocksBlooms), (ldb::Slice)dev::ref(m_blocksBlooms[h].rlp()));
			extrasBatch.Put(toSlice(h256(tbi.number), ExtraBlockHash), (ldb::Slice)dev::ref(BlockHash(tbi.hash()).rlp()));
		}

		// FINALLY! change our best hash.
		{
			newLastBlockHash = _bi.hash();
			newLastBlockNumber = (unsigned)_bi.number;
		}

		clog(BlockChainNote) << "   Imported and best" << td << " (#" << _bi.number << "). Has" << (details(_bi.parentHash).children.size() - 1) << "siblings. Route:" << route;

		StructuredLogger::chainNewHead(
			_bi.headerHash(WithoutNonce).abridged(),
			_bi.nonce.abridged(),
			currentHash().abridged(),
			_bi.parentHash.abridged()
		);
	}
	else
	{
		clog(BlockChainChat) << "   Imported but not best (oTD:" << details(last).totalDifficulty << " > TD:" << td << ")";
	}

	m_blocksDB->Write(m_writeOptions, &blocksBatch);
	m_extrasDB->Write(m_writeOptions, &extrasBatch);

	if (isKnown(_bi.hash()) && !details(_bi.hash()))
	{
		clog(BlockChainDebug) << "Known block just inserted has no details.";
		clog(BlockChainDebug) << "Block:" << _bi;
		clog(BlockChainDebug) << "DATABASE CORRUPTION: CRITICAL FAILURE";
		exit(-1);
	}

	try {
		State canary(_db, *this, _bi.hash(), ImportRequirements::DontHave);
	}
	catch (...)
	{
		clog(BlockChainDebug) << "Failed to initialise State object form imported block.";
		clog(BlockChainDebug) << "Block:" << _bi;
		clog(BlockChainDebug) << "DATABASE CORRUPTION: CRITICAL FAILURE";
		exit(-1);
	}

	if (m_lastBlockHash != newLastBlockHash)
		DEV_WRITE_GUARDED(x_lastBlockHash)
		{
			m_lastBlockHash = newLastBlockHash;
			m_lastBlockNumber = newLastBlockNumber;
			m_extrasDB->Put(m_writeOptions, ldb::Slice("best"), ldb::Slice((char const*)&m_lastBlockHash, 32));
		}

#if ETH_PARANOIA || !ETH_TRUE
	checkConsistency();
#endif

#if ETH_TIMED_IMPORTS
	checkBest = t.elapsed();
	cnote << "Import took:" << total.elapsed();
	cnote << "preliminaryChecks:" << preliminaryChecks;
	cnote << "enactment:" << enactment;
	cnote << "collation:" << collation;
	cnote << "writing:" << writing;
	cnote << "checkBest:" << checkBest;
#endif

	if (!route.empty())
		noteCanonChanged();

	h256s fresh;
	h256s dead;
	bool isOld = true;
	for (auto const& h: route)
		if (h == common)
			isOld = false;
		else if (isOld)
			dead.push_back(h);
		else
			fresh.push_back(h);
	return make_pair(fresh, dead);
}

void BlockChain::clearBlockBlooms(unsigned _begin, unsigned _end)
{
	//   ... c c c c c c c c c c C o o o o o o
	//   ...                               /=15        /=21
	// L0...| ' | ' | ' | ' | ' | ' | ' | 'b|x'x|x'x|x'e| /=11
	// L1...|   '   |   '   |   '   |   ' b | x ' x | x ' e |   /=6
	// L2...|       '       |       '   b   |   x   '   x   |   e   /=3
	// L3...|               '       b       |       x       '       e
	// model: c_bloomIndexLevels = 4, c_bloomIndexSize = 2

	//   ...                               /=15        /=21
	// L0...| ' ' ' | ' ' ' | ' ' ' | ' ' 'b|x'x'x'x|x'e' ' |
	// L1...|       '       '       '   b   |   x   '   x   '   e   '       |
	// L2...|               b               '               x               '                e              '                               |
	// model: c_bloomIndexLevels = 2, c_bloomIndexSize = 4

	// algorithm doesn't have the best memoisation coherence, but eh well...

	unsigned beginDirty = _begin;
	unsigned endDirty = _end;
	for (unsigned level = 0; level < c_bloomIndexLevels; level++, beginDirty /= c_bloomIndexSize, endDirty = (endDirty - 1) / c_bloomIndexSize + 1)
	{
		// compute earliest & latest index for each level, rebuild from previous levels.
		for (unsigned item = beginDirty; item != endDirty; ++item)
		{
			unsigned bunch = item / c_bloomIndexSize;
			unsigned offset = item % c_bloomIndexSize;
			auto id = chunkId(level, bunch);
			LogBloom acc;
			if (!!level)
			{
				// rebuild the bloom from the previous (lower) level (if there is one).
				auto lowerChunkId = chunkId(level - 1, item);
				for (auto const& bloom: blocksBlooms(lowerChunkId).blooms)
					acc |= bloom;
			}
			blocksBlooms(id);	// make sure it has been memoized.
			m_blocksBlooms[id].blooms[offset] = acc;
		}
	}
}

tuple<h256s, h256, unsigned> BlockChain::treeRoute(h256 const& _from, h256 const& _to, bool _common, bool _pre, bool _post) const
{
//	cdebug << "treeRoute" << _from << "..." << _to;
	if (!_from || !_to)
		return make_tuple(h256s(), h256(), 0);
	h256s ret;
	h256s back;
	unsigned fn = details(_from).number;
	unsigned tn = details(_to).number;
//	cdebug << "treeRoute" << fn << "..." << tn;
	h256 from = _from;
	while (fn > tn)
	{
		if (_pre)
			ret.push_back(from);
		from = details(from).parent;
		fn--;
//		cdebug << "from:" << fn << _from;
	}
	h256 to = _to;
	while (fn < tn)
	{
		if (_post)
			back.push_back(to);
		to = details(to).parent;
		tn--;
//		cdebug << "to:" << tn << _to;
	}
	for (;; from = details(from).parent, to = details(to).parent)
	{
		if (_pre && (from != to || _common))
			ret.push_back(from);
		if (_post && (from != to || (!_pre && _common)))
			back.push_back(to);
		fn--;
		tn--;
//		cdebug << "from:" << fn << _from << "; to:" << tn << _to;
		if (from == to)
			break;
		if (!from)
			assert(from);
		if (!to)
			assert(to);
	}
	ret.reserve(ret.size() + back.size());
	unsigned i = ret.size() - (int)(_common && !ret.empty() && !back.empty());
	for (auto it = back.rbegin(); it != back.rend(); ++it)
		ret.push_back(*it);
	return make_tuple(ret, from, i);
}

void BlockChain::noteUsed(h256 const& _h, unsigned _extra) const
{
	auto id = CacheID(_h, _extra);
	Guard l(x_cacheUsage);
	m_cacheUsage[0].insert(id);
	if (m_cacheUsage[1].count(id))
		m_cacheUsage[1].erase(id);
	else
		m_inUse.insert(id);
}

template <class T> static unsigned getHashSize(unordered_map<h256, T> const& _map)
{
	unsigned ret = 0;
	for (auto const& i: _map)
		ret += i.second.size + 64;
	return ret;
}

void BlockChain::updateStats() const
{
	{
		ReadGuard l(x_blocks);
		m_lastStats.memBlocks = 0;
		for (auto const& i: m_blocks)
			m_lastStats.memBlocks += i.second.size() + 64;
	}
	{
		ReadGuard l(x_details);
		m_lastStats.memDetails = getHashSize(m_details);
	}
	{
		ReadGuard l1(x_logBlooms);
		ReadGuard l2(x_blocksBlooms);
		m_lastStats.memLogBlooms = getHashSize(m_logBlooms) + getHashSize(m_blocksBlooms);
	}
	{
		ReadGuard l(x_receipts);
		m_lastStats.memReceipts = getHashSize(m_receipts);
	}
	{
		ReadGuard l(x_blockHashes);
		m_lastStats.memBlockHashes = getHashSize(m_blockHashes);
	}
	{
		ReadGuard l(x_transactionAddresses);
		m_lastStats.memTransactionAddresses = getHashSize(m_transactionAddresses);
	}
}

void BlockChain::garbageCollect(bool _force)
{
	updateStats();

	if (!_force && chrono::system_clock::now() < m_lastCollection + c_collectionDuration && m_lastStats.memTotal() < c_maxCacheSize)
		return;
	if (m_lastStats.memTotal() < c_minCacheSize)
		return;

	m_lastCollection = chrono::system_clock::now();

	Guard l(x_cacheUsage);
	WriteGuard l1(x_blocks);
	WriteGuard l2(x_details);
	WriteGuard l3(x_blockHashes);
	WriteGuard l4(x_receipts);
	WriteGuard l5(x_logBlooms);
	WriteGuard l6(x_transactionAddresses);
	WriteGuard l7(x_blocksBlooms);
	for (CacheID const& id: m_cacheUsage.back())
	{
		m_inUse.erase(id);
		// kill i from cache.
		switch (id.second)
		{
		case (unsigned)-1:
			m_blocks.erase(id.first);
			break;
		case ExtraDetails:
			m_details.erase(id.first);
			break;
		case ExtraBlockHash:
			m_blockHashes.erase(id.first);
			break;
		case ExtraReceipts:
			m_receipts.erase(id.first);
			break;
		case ExtraLogBlooms:
			m_logBlooms.erase(id.first);
			break;
		case ExtraTransactionAddress:
			m_transactionAddresses.erase(id.first);
			break;
		case ExtraBlocksBlooms:
			m_blocksBlooms.erase(id.first);
			break;
		}
	}
	m_cacheUsage.pop_back();
	m_cacheUsage.push_front(std::unordered_set<CacheID>{});
}

void BlockChain::checkConsistency()
{
	{
		WriteGuard l(x_details);
		m_details.clear();
	}
	ldb::Iterator* it = m_blocksDB->NewIterator(m_readOptions);
	for (it->SeekToFirst(); it->Valid(); it->Next())
		if (it->key().size() == 32)
		{
			h256 h((byte const*)it->key().data(), h256::ConstructFromPointer);
			auto dh = details(h);
			auto p = dh.parent;
			if (p != h256() && p != m_genesisHash)	// TODO: for some reason the genesis details with the children get squished. not sure why.
			{
				auto dp = details(p);
				if (asserts(contains(dp.children, h)))
				{
					cnote << "Apparently the database is corrupt. Not much we can do at this stage...";
				}
				if (assertsEqual(dp.number, dh.number - 1))
				{
					cnote << "Apparently the database is corrupt. Not much we can do at this stage...";
				}
			}
		}
	delete it;
}

static inline unsigned upow(unsigned a, unsigned b) { while (b-- > 0) a *= a; return a; }
static inline unsigned ceilDiv(unsigned n, unsigned d) { return n / (n + d - 1); }
//static inline unsigned floorDivPow(unsigned n, unsigned a, unsigned b) { return n / upow(a, b); }
//static inline unsigned ceilDivPow(unsigned n, unsigned a, unsigned b) { return ceilDiv(n, upow(a, b)); }

// Level 1
// [xxx.            ]

// Level 0
// [.x............F.]
// [........x.......]
// [T.............x.]
// [............    ]

// F = 14. T = 32

vector<unsigned> BlockChain::withBlockBloom(LogBloom const& _b, unsigned _earliest, unsigned _latest) const
{
	vector<unsigned> ret;

	// start from the top-level
	unsigned u = upow(c_bloomIndexSize, c_bloomIndexLevels);

	// run through each of the top-level blockbloom blocks
	for (unsigned index = _earliest / u; index <= ceilDiv(_latest, u); ++index)				// 0
		ret += withBlockBloom(_b, _earliest, _latest, c_bloomIndexLevels - 1, index);

	return ret;
}

vector<unsigned> BlockChain::withBlockBloom(LogBloom const& _b, unsigned _earliest, unsigned _latest, unsigned _level, unsigned _index) const
{
	// 14, 32, 1, 0
		// 14, 32, 0, 0
		// 14, 32, 0, 1
		// 14, 32, 0, 2

	vector<unsigned> ret;

	unsigned uCourse = upow(c_bloomIndexSize, _level + 1);
	// 256
		// 16
	unsigned uFine = upow(c_bloomIndexSize, _level);
	// 16
		// 1

	unsigned obegin = _index == _earliest / uCourse ? _earliest / uFine % c_bloomIndexSize : 0;
	// 0
		// 14
		// 0
		// 0
	unsigned oend = _index == _latest / uCourse ? (_latest / uFine) % c_bloomIndexSize + 1 : c_bloomIndexSize;
	// 3
		// 16
		// 16
		// 1

	BlocksBlooms bb = blocksBlooms(_level, _index);
	for (unsigned o = obegin; o < oend; ++o)
		if (bb.blooms[o].contains(_b))
		{
			// This level has something like what we want.
			if (_level > 0)
				ret += withBlockBloom(_b, _earliest, _latest, _level - 1, o + _index * c_bloomIndexSize);
			else
				ret.push_back(o + _index * c_bloomIndexSize);
		}
	return ret;
}

h256Hash BlockChain::allKinFrom(h256 const& _parent, unsigned _generations) const
{
	// Get all uncles cited given a parent (i.e. featured as uncles/main in parent, parent + 1, ... parent + 5).
	h256 p = _parent;
	h256Hash ret = { p };
	// p and (details(p).parent: i == 5) is likely to be overkill, but can't hurt to be cautious.
	for (unsigned i = 0; i < _generations && p != m_genesisHash; ++i, p = details(p).parent)
	{
		ret.insert(details(p).parent);
		auto b = block(p);
		for (auto i: RLP(b)[2])
			ret.insert(sha3(i.data()));
	}
	return ret;
}

bool BlockChain::isKnown(h256 const& _hash) const
{
	if (_hash == m_genesisHash)
		return true;

	DEV_READ_GUARDED(x_blocks)
		if (!m_blocks.count(_hash))
		{
			string d;
			m_blocksDB->Get(m_readOptions, toSlice(_hash), &d);
			if (d.empty())
				return false;
		}
	DEV_READ_GUARDED(x_details)
		if (!m_details.count(_hash))
		{
			string d;
			m_extrasDB->Get(m_readOptions, toSlice(_hash, ExtraDetails), &d);
			if (d.empty())
				return false;
		}
	return true;
}

bytes BlockChain::block(h256 const& _hash) const
{
	if (_hash == m_genesisHash)
		return m_genesisBlock;

	{
		ReadGuard l(x_blocks);
		auto it = m_blocks.find(_hash);
		if (it != m_blocks.end())
			return it->second;
	}

	string d;
	m_blocksDB->Get(m_readOptions, toSlice(_hash), &d);

	if (d.empty())
	{
		cwarn << "Couldn't find requested block:" << _hash;
		return bytes();
	}

	noteUsed(_hash);

	WriteGuard l(x_blocks);
	m_blocks[_hash].resize(d.size());
	memcpy(m_blocks[_hash].data(), d.data(), d.size());

	return m_blocks[_hash];
}
