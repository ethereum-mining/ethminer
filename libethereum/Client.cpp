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
/** @file Client.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "Client.h"

#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <libdevcore/Log.h>
#include <libdevcore/StructuredLogger.h>
#include <libp2p/Host.h>
#include "Defaults.h"
#include "Executive.h"
#include "EthereumHost.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

VersionChecker::VersionChecker(string const& _dbPath):
	m_path(_dbPath.size() ? _dbPath : Defaults::dbPath())
{
	auto protocolContents = contents(m_path + "/protocol");
	auto databaseContents = contents(m_path + "/database");
	m_ok = RLP(protocolContents).toInt<unsigned>(RLP::LaisezFaire) == eth::c_protocolVersion && RLP(databaseContents).toInt<unsigned>(RLP::LaisezFaire) == c_databaseVersion;
}

void VersionChecker::setOk()
{
	if (!m_ok)
	{
		try
		{
			boost::filesystem::create_directory(m_path);
		}
		catch (...)
		{
			cwarn << "Unhandled exception! Failed to create directory: " << m_path << "\n" << boost::current_exception_diagnostic_information();
		}
		writeFile(m_path + "/protocol", rlp(eth::c_protocolVersion));
		writeFile(m_path + "/database", rlp(c_databaseVersion));
	}
}

void BasicGasPricer::update(BlockChain const& _bc)
{
	unsigned c = 0;
	h256 p = _bc.currentHash();
	m_gasPerBlock = _bc.info(p).gasLimit;

	map<u256, unsigned> dist;
	unsigned total = 0;
	while (c < 1000 && p)
	{
		BlockInfo bi = _bc.info(p);
		if (bi.transactionsRoot != EmptyTrie)
		{
			auto bb = _bc.block(p);
			RLP r(bb);
			BlockReceipts brs(_bc.receipts(bi.hash));
			for (unsigned i = 0; i < r[1].size(); ++i)
			{
				auto gu = brs.receipts[i].gasUsed();
				dist[Transaction(r[1][i].data(), CheckSignature::None).gasPrice()] += (unsigned)brs.receipts[i].gasUsed();
				total += (unsigned)gu;
			}
		}
		p = bi.parentHash;
		++c;
	}
	if (total > 0)
	{
		unsigned t = 0;
		unsigned q = 1;
		m_octiles[0] = dist.begin()->first;
		for (auto const& i: dist)
		{
			for (; t <= total * q / 8 && t + i.second > total * q / 8; ++q)
				m_octiles[q] = i.first;
			if (q > 7)
				break;
		}
		m_octiles[8] = dist.rbegin()->first;
	}
}

Client::Client(p2p::Host* _extNet, std::string const& _dbPath, bool _forceClean, u256 _networkId, int _miners):
	Worker("eth"),
	m_vc(_dbPath),
	m_bc(_dbPath, !m_vc.ok() || _forceClean),
	m_gp(new TrivialGasPricer),
	m_stateDB(State::openDB(_dbPath, !m_vc.ok() || _forceClean)),
	m_preMine(Address(), m_stateDB),
	m_postMine(Address(), m_stateDB)
{
	m_gp->update(m_bc);

	m_host = _extNet->registerCapability(new EthereumHost(m_bc, m_tq, m_bq, _networkId));

	if (_miners > -1)
		setMiningThreads(_miners);
	else
		setMiningThreads();
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
	m_vc.setOk();
	doWork();

	startWorking();
}

Client::Client(p2p::Host* _extNet, std::shared_ptr<GasPricer> _gp, std::string const& _dbPath, bool _forceClean, u256 _networkId, int _miners):
	Worker("eth"),
	m_vc(_dbPath),
	m_bc(_dbPath, !m_vc.ok() || _forceClean),
	m_gp(_gp),
	m_stateDB(State::openDB(_dbPath, !m_vc.ok() || _forceClean)),
	m_preMine(Address(), m_stateDB),
	m_postMine(Address(), m_stateDB)
{
	m_gp->update(m_bc);

	m_host = _extNet->registerCapability(new EthereumHost(m_bc, m_tq, m_bq, _networkId));

	if (_miners > -1)
		setMiningThreads(_miners);
	else
		setMiningThreads();
	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
	m_vc.setOk();
	doWork();

	startWorking();
}

Client::~Client()
{
	stopWorking();
}

void Client::setNetworkId(u256 _n)
{
	if (auto h = m_host.lock())
		h->setNetworkId(_n);
}

DownloadMan const* Client::downloadMan() const
{
	if (auto h = m_host.lock())
		return &(h->downloadMan());
	return nullptr;
}

bool Client::isSyncing() const
{
	if (auto h = m_host.lock())
		return h->isSyncing();
	return false;
}

void Client::doneWorking()
{
	// Synchronise the state according to the head of the block chain.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	WriteGuard l(x_stateDB);
	m_preMine.sync(m_bc);
	m_postMine = m_preMine;
}

void Client::flushTransactions()
{
	doWork();
}

void Client::killChain()
{
	bool wasMining = isMining();
	if (wasMining)
		stopMining();
	stopWorking();

	m_tq.clear();
	m_bq.clear();
	m_localMiners.clear();
	m_preMine = State();
	m_postMine = State();

	{
		WriteGuard l(x_stateDB);
		m_stateDB = OverlayDB();
		m_stateDB = State::openDB(Defaults::dbPath(), true);
	}
	m_bc.reopen(Defaults::dbPath(), true);

	m_preMine = State(Address(), m_stateDB);
	m_postMine = State(Address(), m_stateDB);

	if (auto h = m_host.lock())
		h->reset();

	doWork();

	setMiningThreads(0);

	startWorking();
	if (wasMining)
		startMining();
}

void Client::clearPending()
{
	h256Set changeds;
	{
		WriteGuard l(x_stateDB);
		if (!m_postMine.pending().size())
			return;
//		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
//			appendFromNewPending(m_postMine.logBloom(i), changeds);
		changeds.insert(PendingChangedFilter);
		m_tq.clear();
		m_postMine = m_preMine;
	}

	{
		ReadGuard l(x_localMiners);
		for (auto& m: m_localMiners)
			m.noteStateChange();
	}

	noteChanged(changeds);
}

unsigned Client::installWatch(h256 _h, Reaping _r)
{
	unsigned ret;
	{
		Guard l(m_filterLock);
		ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
		m_watches[ret] = ClientWatch(_h, _r);
		cwatch << "+++" << ret << _h.abridged();
	}
	auto ch = logs(ret);
	if (ch.empty())
		ch.push_back(InitialChange);
	{
		Guard l(m_filterLock);
		swap(m_watches[ret].changes, ch);
	}
	return ret;
}

unsigned Client::installWatch(LogFilter const& _f, Reaping _r)
{
	h256 h = _f.sha3();
	{
		Guard l(m_filterLock);
		if (!m_filters.count(h))
		{
			cwatch << "FFF" << _f << h.abridged();
			m_filters.insert(make_pair(h, _f));
		}
	}
	return installWatch(h, _r);
}

bool Client::uninstallWatch(unsigned _i)
{
	cwatch << "XXX" << _i;

	Guard l(m_filterLock);

	auto it = m_watches.find(_i);
	if (it == m_watches.end())
		return false;
	auto id = it->second.id;
	m_watches.erase(it);

	auto fit = m_filters.find(id);
	if (fit != m_filters.end())
		if (!--fit->second.refCount)
		{
			cwatch << "*X*" << fit->first << ":" << fit->second.filter;
			m_filters.erase(fit);
		}
	return true;
}

void Client::noteChanged(h256Set const& _filters)
{
	Guard l(m_filterLock);
	if (_filters.size())
		cnote << "noteChanged(" << _filters << ")";
	// accrue all changes left in each filter into the watches.
	for (auto& i: m_watches)
		if (_filters.count(i.second.id))
		{
			cwatch << "!!!" << i.first << i.second.id;
			if (m_filters.count(i.second.id))
				i.second.changes += m_filters.at(i.second.id).changes;
			else
				i.second.changes.push_back(LocalisedLogEntry(SpecialLogEntry, 0));
		}
	// clear the filters now.
	for (auto& i: m_filters)
		i.second.changes.clear();
}

LocalisedLogEntries Client::peekWatch(unsigned _watchId) const
{
	Guard l(m_filterLock);

#if ETH_DEBUG
	cdebug << "peekWatch" << _watchId;
#endif
	auto& w = m_watches.at(_watchId);
#if ETH_DEBUG
	cdebug << "lastPoll updated to " << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch()).count();
#endif
	w.lastPoll = chrono::system_clock::now();
	return w.changes;
}

LocalisedLogEntries Client::checkWatch(unsigned _watchId)
{
	Guard l(m_filterLock);
	LocalisedLogEntries ret;

#if ETH_DEBUG && 0
	cdebug << "checkWatch" << _watchId;
#endif
	auto& w = m_watches.at(_watchId);
#if ETH_DEBUG && 0
	cdebug << "lastPoll updated to " << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch()).count();
#endif
	std::swap(ret, w.changes);
	w.lastPoll = chrono::system_clock::now();

	return ret;
}

void Client::appendFromNewPending(TransactionReceipt const& _receipt, h256Set& io_changed, h256 _transactionHash)
{
	Guard l(m_filterLock);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
		if (i.second.filter.envelops(RelativeBlock::Pending, m_bc.number() + 1))
		{
			// acceptable number.
			auto m = i.second.filter.matches(_receipt);
			if (m.size())
			{
				// filter catches them
				for (LogEntry const& l: m)
					i.second.changes.push_back(LocalisedLogEntry(l, m_bc.number() + 1, _transactionHash));
				io_changed.insert(i.first);
			}
		}
}

void Client::appendFromNewBlock(h256 const& _block, h256Set& io_changed)
{
	// TODO: more precise check on whether the txs match.
	auto d = m_bc.info(_block);
	auto br = m_bc.receipts(_block);

	Guard l(m_filterLock);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
		if (i.second.filter.envelops(RelativeBlock::Latest, d.number) && i.second.filter.matches(d.logBloom))
			// acceptable number & looks like block may contain a matching log entry.
			for (size_t j = 0; j < br.receipts.size(); j++)
			{
				auto tr = br.receipts[j];
				auto m = i.second.filter.matches(tr);
				if (m.size())
				{
					auto transactionHash = transaction(d.hash, j).sha3();
					// filter catches them
					for (LogEntry const& l: m)
						i.second.changes.push_back(LocalisedLogEntry(l, (unsigned)d.number, transactionHash));
					io_changed.insert(i.first);
				}
			}
}

void Client::setForceMining(bool _enable)
{
	 m_forceMining = _enable;
	 ReadGuard l(x_localMiners);
	 for (auto& m: m_localMiners)
		 m.noteStateChange();
}

void Client::setMiningThreads(unsigned _threads)
{
	stopMining();
#if ETH_ETHASHCL
	unsigned t = 1;
#else
	auto t = _threads ? _threads : thread::hardware_concurrency();
#endif
	WriteGuard l(x_localMiners);
	m_localMiners.clear();
	m_localMiners.resize(t);
	unsigned i = 0;
	for (auto& m: m_localMiners)
		m.setup(this, i++);
}

MineProgress Client::miningProgress() const
{
	MineProgress ret;
	ReadGuard l(x_localMiners);
	for (auto& m: m_localMiners)
		ret.combine(m.miningProgress());
	return ret;
}

std::list<MineInfo> Client::miningHistory()
{
	std::list<MineInfo> ret;

	ReadGuard l(x_localMiners);
	if (m_localMiners.empty())
		return ret;
	ret = m_localMiners[0].miningHistory();
	for (unsigned i = 1; i < m_localMiners.size(); ++i)
	{
		auto l = m_localMiners[i].miningHistory();
		auto ri = ret.begin();
		auto li = l.begin();
		for (; ri != ret.end() && li != l.end(); ++ri, ++li)
			ri->combine(*li);
	}
	return ret;
}

void Client::setupState(State& _s)
{
	{
		ReadGuard l(x_stateDB);
		cwork << "SETUP MINE";
		_s = m_postMine;
	}
	if (m_paranoia)
	{
		if (_s.amIJustParanoid(m_bc))
		{
			cnote << "I'm just paranoid. Block is fine.";
			_s.commitToMine(m_bc);
		}
		else
		{
			cwarn << "I'm not just paranoid. Cannot mine. Please file a bug report.";
		}
	}
	else
		_s.commitToMine(m_bc);
}

void Client::submitTransaction(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	startWorking();

	u256 n;
	{
		ReadGuard l(x_stateDB);
		n = m_postMine.transactionsFrom(toAddress(_secret));
	}
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
	StructuredLogger::transactionReceived(t.sha3().abridged(), t.sender().abridged());
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
}

ExecutionResult Client::call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber)
{
	ExecutionResult ret;
	try
	{
		u256 n;
		State temp;
	//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		{
			ReadGuard l(x_stateDB);
			temp = asOf(_blockNumber);
			n = temp.transactionsFrom(toAddress(_secret));
		}
		Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
		ret = temp.execute(m_bc, t.rlp(), Permanence::Reverted);
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return ret;
}

ExecutionResult Client::create(Secret _secret, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber)
{
	ExecutionResult ret;
	try
	{
		u256 n;
		State temp;
	//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		{
			ReadGuard l(x_stateDB);
			temp = asOf(_blockNumber);
			n = temp.transactionsFrom(toAddress(_secret));
		}
		Transaction t(_value, _gasPrice, _gas, _data, n, _secret);
		ret = temp.execute(m_bc, t.rlp(), Permanence::Reverted);
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return ret;
}

ExecutionResult Client::call(Address _dest, bytes const& _data, u256 _gas, u256 _value, u256 _gasPrice)
{
	ExecutionResult ret;
	try
	{
		State temp;
//		cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		{
			ReadGuard l(x_stateDB);
			temp = m_postMine;
		}
		Executive e(temp, LastHashes(), 0);
		if (!e.call(_dest, _dest, Address(), _value, _gasPrice, &_data, _gas, Address()))
			e.go();
		ret = e.executionResult();
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return ret;
}

Address Client::submitTransaction(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	startWorking();

	u256 n;
	{
		ReadGuard l(x_stateDB);
		n = m_postMine.transactionsFrom(toAddress(_secret));
	}
	Transaction t(_endowment, _gasPrice, _gas, _init, n, _secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
	return right160(sha3(rlpList(t.sender(), t.nonce())));
}

void Client::inject(bytesConstRef _rlp)
{
	startWorking();

	m_tq.attemptImport(_rlp);
}

pair<h256, u256> Client::getWork()
{
	Guard l(x_remoteMiner);
	{
		ReadGuard l(x_stateDB);
		m_remoteMiner.update(m_postMine, m_bc);
	}
	return make_pair(m_remoteMiner.workHash(), m_remoteMiner.difficulty());
}

bool Client::submitWork(ProofOfWork::Proof const& _proof)
{
	Guard l(x_remoteMiner);
	return m_remoteMiner.submitWork(_proof);
}

void Client::doWork()
{
	// TODO: Use condition variable rather than polling.

	cworkin << "WORK";
	h256Set changeds;

	auto maintainMiner = [&](Miner& m)
	{
		if (m.isComplete())
		{
			// TODO: enable a short-circuit option since we mined it. will need to get the end state from the miner.
			auto lm = dynamic_cast<LocalMiner*>(&m);
			h256s hs;
			if (false && lm && !m_verifyOwnBlocks)
			{
				// TODO: implement
				//m_bc.attemptImport(m_blockData(), m_stateDB, lm->state());
				// TODO: derive hs from lm->state()
			}
			else
			{
				cwork << "CHAIN <== postSTATE";
				WriteGuard l(x_stateDB);
				hs = m_bc.attemptImport(m.blockData(), m_stateDB);
			}
			if (hs.size())
			{
				for (auto const& h: hs)
					appendFromNewBlock(h, changeds);
				changeds.insert(ChainChangedFilter);
			}
			for (auto& m: m_localMiners)
				m.noteStateChange();
		}
	};
	{
		ReadGuard l(x_localMiners);
		for (auto& m: m_localMiners)
			maintainMiner(m);
	}
	{
		Guard l(x_remoteMiner);
		maintainMiner(m_remoteMiner);
	}

	// Synchronise state to block chain.
	// This should remove any transactions on our queue that are included within our state.
	// It also guarantees that the state reflects the longest (valid!) chain on the block chain.
	//   This might mean reverting to an earlier state and replaying some blocks, or, (worst-case:
	//   if there are no checkpoints before our fork) reverting to the genesis block and replaying
	//   all blocks.
	// Resynchronise state with block chain & trans
	bool resyncStateNeeded = false;
	{
		WriteGuard l(x_stateDB);
		cwork << "BQ ==> CHAIN ==> STATE";
		OverlayDB db = m_stateDB;
		x_stateDB.unlock();
		h256s newBlocks = m_bc.sync(m_bq, db, 100);	// TODO: remove transactions from m_tq nicely rather than relying on out of date nonce later on.
		if (newBlocks.size())
		{
			for (auto i: newBlocks)
				appendFromNewBlock(i, changeds);
			changeds.insert(ChainChangedFilter);
		}
		x_stateDB.lock();
		if (newBlocks.size())
			m_stateDB = db;

		cwork << "preSTATE <== CHAIN";
		if (m_preMine.sync(m_bc) || m_postMine.address() != m_preMine.address())
		{
			if (isMining())
				cnote << "New block on chain: Restarting mining operation.";
			m_postMine = m_preMine;
			resyncStateNeeded = true;
			changeds.insert(PendingChangedFilter);
			// TODO: Move transactions pending from m_postMine back to transaction queue.
		}

		// returns h256s as blooms, once for each transaction.
		cwork << "postSTATE <== TQ";
		TransactionReceipts newPendingReceipts = m_postMine.sync(m_bc, m_tq, *m_gp);
		if (newPendingReceipts.size())
		{
			for (size_t i = 0; i < newPendingReceipts.size(); i++)
				appendFromNewPending(newPendingReceipts[i], changeds, m_postMine.pending()[i].sha3());
			
			changeds.insert(PendingChangedFilter);

			if (isMining())
				cnote << "Additional transaction ready: Restarting mining operation.";
			resyncStateNeeded = true;
		}
	}
	if (resyncStateNeeded)
	{
		ReadGuard l(x_localMiners);
		for (auto& m: m_localMiners)
			m.noteStateChange();
	}

	cwork << "noteChanged" << changeds.size() << "items";
	noteChanged(changeds);
	cworkout << "WORK";

	this_thread::sleep_for(chrono::milliseconds(100));
	if (chrono::system_clock::now() - m_lastGarbageCollection > chrono::seconds(5))
	{
		// watches garbage collection
		vector<unsigned> toUninstall;
		{
			Guard l(m_filterLock);
			for (auto key: keysOf(m_watches))
				if (m_watches[key].lastPoll != chrono::system_clock::time_point::max() && chrono::system_clock::now() - m_watches[key].lastPoll > chrono::seconds(20))
				{
					toUninstall.push_back(key);
					cnote << "GC: Uninstall" << key << "(" << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now() - m_watches[key].lastPoll).count() << "s old)";
				}
		}
		for (auto i: toUninstall)
			uninstallWatch(i);

		// blockchain GC
		m_bc.garbageCollect();

		m_lastGarbageCollection = chrono::system_clock::now();
	}
}

State Client::asOf(unsigned _h) const
{
	ReadGuard l(x_stateDB);
	if (_h == PendingBlock)
		return m_postMine;
	else if (_h == LatestBlock)
		return m_preMine;
	else
		return State(m_stateDB, m_bc, m_bc.numberHash(_h));
}

State Client::state(unsigned _txi, h256 _block) const
{
	ReadGuard l(x_stateDB);
	return State(m_stateDB, m_bc, _block).fromPending(_txi);
}

eth::State Client::state(h256 _block) const
{
	ReadGuard l(x_stateDB);
	return State(m_stateDB, m_bc, _block);
}

eth::State Client::state(unsigned _txi) const
{
	ReadGuard l(x_stateDB);
	return m_postMine.fromPending(_txi);
}

StateDiff Client::diff(unsigned _txi, BlockNumber _block) const
{
	State st = asOf(_block);
	return st.fromPending(_txi).diff(st.fromPending(_txi + 1));
}

StateDiff Client::diff(unsigned _txi, h256 _block) const
{
	State st = state(_block);
	return st.fromPending(_txi).diff(st.fromPending(_txi + 1));
}

std::vector<Address> Client::addresses(BlockNumber _block) const
{
	vector<Address> ret;
	for (auto const& i: asOf(_block).addresses())
		ret.push_back(i.first);
	return ret;
}

u256 Client::balanceAt(Address _a, BlockNumber _block) const
{
	return asOf(_block).balance(_a);
}

std::map<u256, u256> Client::storageAt(Address _a, BlockNumber _block) const
{
	return asOf(_block).storage(_a);
}

u256 Client::countAt(Address _a, BlockNumber _block) const
{
	return asOf(_block).transactionsFrom(_a);
}

u256 Client::stateAt(Address _a, u256 _l, BlockNumber _block) const
{
	return asOf(_block).storage(_a, _l);
}

bytes Client::codeAt(Address _a, BlockNumber _block) const
{
	return asOf(_block).code(_a);
}

Transaction Client::transaction(h256 _transactionHash) const
{
	return Transaction(m_bc.transaction(_transactionHash), CheckSignature::Range);
}

Transaction Client::transaction(h256 _blockHash, unsigned _i) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	if (_i < b[1].itemCount())
		return Transaction(b[1][_i].data(), CheckSignature::Range);
	else
		return Transaction();
}

BlockInfo Client::uncle(h256 _blockHash, unsigned _i) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	if (_i < b[2].itemCount())
		return BlockInfo::fromHeader(b[2][_i].data());
	else
		return BlockInfo();
}

unsigned Client::transactionCount(h256 _blockHash) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	return b[1].itemCount();
}

unsigned Client::uncleCount(h256 _blockHash) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	return b[2].itemCount();
}

Transactions Client::transactions(h256 _blockHash) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	Transactions res;
	for (unsigned i = 0; i < b[1].itemCount(); i++)
		res.emplace_back(b[1][i].data(), CheckSignature::Range);
	return res;
}

TransactionHashes Client::transactionHashes(h256 _blockHash) const
{
	return m_bc.transactionHashes(_blockHash);
}

LocalisedLogEntries Client::logs(unsigned _watchId) const
{
	LogFilter f;
	try
	{
		Guard l(m_filterLock);
		f = m_filters.at(m_watches.at(_watchId).id).filter;
	}
	catch (...)
	{
		return LocalisedLogEntries();
	}
	return logs(f);
}

LocalisedLogEntries Client::logs(LogFilter const& _f) const
{
	LocalisedLogEntries ret;
	unsigned begin = min<unsigned>(m_bc.number() + 1, (unsigned)_f.latest());
	unsigned end = min(m_bc.number(), min(begin, (unsigned)_f.earliest()));

	// Handle pending transactions differently as they're not on the block chain.
	if (begin > m_bc.number())
	{
		ReadGuard l(x_stateDB);
		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
		{
			// Might have a transaction that contains a matching log.
			TransactionReceipt const& tr = m_postMine.receipt(i);
			auto sha3 = m_postMine.pending()[i].sha3();
			LogEntries le = _f.matches(tr);
			if (le.size())
				for (unsigned j = 0; j < le.size(); ++j)
					ret.insert(ret.begin(), LocalisedLogEntry(le[j], begin, sha3));
		}
		begin = m_bc.number();
	}

	set<unsigned> matchingBlocks;
	for (auto const& i: _f.bloomPossibilities())
		for (auto u: m_bc.withBlockBloom(i, end, begin))
			matchingBlocks.insert(u);

#if ETH_DEBUG
	unsigned falsePos = 0;
#endif
	for (auto n: matchingBlocks)
	{
#if ETH_DEBUG
		int total = 0;
#endif
		auto h = m_bc.numberHash(n);
		auto receipts = m_bc.receipts(h).receipts;
		for (size_t i = 0; i < receipts.size(); i++)
		{
			TransactionReceipt receipt = receipts[i];
			if (_f.matches(receipt.bloom()))
			{
				auto info = m_bc.info(h);
				auto h = transaction(info.hash, i).sha3();
				LogEntries le = _f.matches(receipt);
				if (le.size())
				{
#if ETH_DEBUG
					total += le.size();
#endif
					for (unsigned j = 0; j < le.size(); ++j)
						ret.insert(ret.begin(), LocalisedLogEntry(le[j], n, h));
				}
			}
#if ETH_DEBUG
			if (!total)
				falsePos++;
#endif
		}
	}
#if ETH_DEBUG
	cdebug << matchingBlocks.size() << "searched from" << (end - begin) << "skipped; " << falsePos << "false +ves";
#endif
	return ret;
}
