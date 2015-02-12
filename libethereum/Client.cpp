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
	m_ok = RLP(protocolContents).toInt<unsigned>(RLP::LaisezFaire) == c_protocolVersion && RLP(databaseContents).toInt<unsigned>(RLP::LaisezFaire) == c_databaseVersion;
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
		writeFile(m_path + "/protocol", rlp(c_protocolVersion));
		writeFile(m_path + "/database", rlp(c_databaseVersion));
	}
}

Client::Client(p2p::Host* _extNet, std::string const& _dbPath, bool _forceClean, u256 _networkId, int miners):
	Worker("eth"),
	m_vc(_dbPath),
	m_bc(_dbPath, !m_vc.ok() || _forceClean),
	m_stateDB(State::openDB(_dbPath, !m_vc.ok() || _forceClean)),
	m_preMine(Address(), m_stateDB),
	m_postMine(Address(), m_stateDB)
{
	m_host = _extNet->registerCapability(new EthereumHost(m_bc, m_tq, m_bq, _networkId));

	if (miners > -1)
		setMiningThreads(miners);
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

unsigned Client::installWatch(h256 _h)
{
	unsigned ret;
	{
		Guard l(m_filterLock);
		ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
		m_watches[ret] = ClientWatch(_h);
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

unsigned Client::installWatch(LogFilter const& _f)
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
	return installWatch(h);
}

void Client::uninstallWatch(unsigned _i)
{
	cwatch << "XXX" << _i;

	Guard l(m_filterLock);

	auto it = m_watches.find(_i);
	if (it == m_watches.end())
		return;
	auto id = it->second.id;
	m_watches.erase(it);

	auto fit = m_filters.find(id);
	if (fit != m_filters.end())
		if (!--fit->second.refCount)
		{
			cwatch << "*X*" << fit->first << ":" << fit->second.filter;
			m_filters.erase(fit);
		}
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

	try {
		auto& w = m_watches.at(_watchId);
		w.lastPoll = chrono::system_clock::now();
		return w.changes;
	} catch (...) {}

	return LocalisedLogEntries();
}

LocalisedLogEntries Client::checkWatch(unsigned _watchId)
{
	Guard l(m_filterLock);
	LocalisedLogEntries ret;

	try {
		auto& w = m_watches.at(_watchId);
		std::swap(ret, w.changes);
		w.lastPoll = chrono::system_clock::now();
	} catch (...) {}

	return ret;
}

void Client::appendFromNewPending(TransactionReceipt const& _receipt, h256Set& io_changed)
{
	Guard l(m_filterLock);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
		if ((unsigned)i.second.filter.latest() > m_bc.number())
		{
			// acceptable number.
			auto m = i.second.filter.matches(_receipt);
			if (m.size())
			{
				// filter catches them
				for (LogEntry const& l: m)
					i.second.changes.push_back(LocalisedLogEntry(l, m_bc.number() + 1));
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
		if ((unsigned)i.second.filter.latest() >= d.number && (unsigned)i.second.filter.earliest() <= d.number && i.second.filter.matches(d.logBloom))
			// acceptable number & looks like block may contain a matching log entry.
			for (TransactionReceipt const& tr: br.receipts)
			{
				auto m = i.second.filter.matches(tr);
				if (m.size())
				{
					// filter catches them
					for (LogEntry const& l: m)
						i.second.changes.push_back(LocalisedLogEntry(l, (unsigned)d.number));
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

	auto t = _threads ? _threads : thread::hardware_concurrency();
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

void Client::transact(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	startWorking();

	u256 n;
	{
		ReadGuard l(x_stateDB);
		n = m_postMine.transactionsFrom(toAddress(_secret));
	}
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
}

bytes Client::call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	bytes out;
	try
	{
		u256 n;
		State temp;
	//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		{
			ReadGuard l(x_stateDB);
			temp = m_postMine;
			n = temp.transactionsFrom(toAddress(_secret));
		}
		Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
		u256 gasUsed = temp.execute(m_bc, t.rlp(), &out, false);
		(void)gasUsed; // TODO: do something with gasused which it returns.
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return out;
}

bytes Client::call(Address _dest, bytes const& _data, u256 _gas, u256 _value, u256 _gasPrice)
{
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
		{
			e.go();
			return e.out().toBytes();
		}
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return bytes();
}

Address Client::transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
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

bool Client::submitNonce(h256  const&_nonce)
{
	Guard l(x_remoteMiner);
	return m_remoteMiner.submitWork(_nonce);
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
			cwork << "CHAIN <== postSTATE";
			h256s hs;
			{
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
		TransactionReceipts newPendingReceipts = m_postMine.sync(m_bc, m_tq);
		if (newPendingReceipts.size())
		{
			for (auto i: newPendingReceipts)
				appendFromNewPending(i, changeds);
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
		// garbage collect on watches
		vector<unsigned> toUninstall;
		{
			Guard l(m_filterLock);
			for (auto key: keysOf(m_watches))
				if (chrono::system_clock::now() - m_watches[key].lastPoll > chrono::seconds(20))
				{
					toUninstall.push_back(key);
					cnote << "GC: Uninstall" << key << "(" << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now() - m_watches[key].lastPoll).count() << "s old)";
				}
		}
		for (auto i: toUninstall)
			uninstallWatch(i);
		m_lastGarbageCollection = chrono::system_clock::now();
	}
}

unsigned Client::numberOf(int _n) const
{
	if (_n > 0)
		return _n;
	else if (_n == GenesisBlock)
		return 0;
	else
		return m_bc.details().number + max(-(int)m_bc.details().number, 1 + _n);
}

State Client::asOf(int _h) const
{
	ReadGuard l(x_stateDB);
	if (_h == 0)
		return m_postMine;
	else if (_h == -1)
		return m_preMine;
	else
		return State(m_stateDB, m_bc, m_bc.numberHash(numberOf(_h)));
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

StateDiff Client::diff(unsigned _txi, int _block) const
{
	State st = asOf(_block);
	return st.fromPending(_txi).diff(st.fromPending(_txi + 1));
}

StateDiff Client::diff(unsigned _txi, h256 _block) const
{
	State st = state(_block);
	return st.fromPending(_txi).diff(st.fromPending(_txi + 1));
}

std::vector<Address> Client::addresses(int _block) const
{
	vector<Address> ret;
	for (auto const& i: asOf(_block).addresses())
		ret.push_back(i.first);
	return ret;
}

u256 Client::balanceAt(Address _a, int _block) const
{
	return asOf(_block).balance(_a);
}

std::map<u256, u256> Client::storageAt(Address _a, int _block) const
{
	return asOf(_block).storage(_a);
}

u256 Client::countAt(Address _a, int _block) const
{
	return asOf(_block).transactionsFrom(_a);
}

u256 Client::stateAt(Address _a, u256 _l, int _block) const
{
	return asOf(_block).storage(_a, _l);
}

bytes Client::codeAt(Address _a, int _block) const
{
	return asOf(_block).code(_a);
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

LocalisedLogEntries Client::logs(LogFilter const& _f) const
{
	LocalisedLogEntries ret;
	unsigned begin = min<unsigned>(m_bc.number() + 1, (unsigned)_f.latest());
	unsigned end = min(m_bc.number(), min(begin, (unsigned)_f.earliest()));
	unsigned m = _f.max();
	unsigned s = _f.skip();

	// Handle pending transactions differently as they're not on the block chain.
	if (begin > m_bc.number())
	{
		ReadGuard l(x_stateDB);
		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
		{
			// Might have a transaction that contains a matching log.
			TransactionReceipt const& tr = m_postMine.receipt(i);
			LogEntries le = _f.matches(tr);
			if (le.size())
			{
				for (unsigned j = 0; j < le.size() && ret.size() != m; ++j)
					if (s)
						s--;
					else
						ret.insert(ret.begin(), LocalisedLogEntry(le[j], begin));
			}
		}
		begin = m_bc.number();
	}

#if ETH_DEBUG
	// fill these params
	unsigned skipped = 0;
	unsigned falsePos = 0;
#endif
	auto h = m_bc.numberHash(begin);
	unsigned n = begin;
	for (; ret.size() != m && n != end; n--, h = m_bc.details(h).parent)
	{
#if ETH_DEBUG
		int total = 0;
#endif
		// check block bloom
		if (_f.matches(m_bc.info(h).logBloom))
			for (TransactionReceipt receipt: m_bc.receipts(h).receipts)
			{
				if (_f.matches(receipt.bloom()))
				{
					LogEntries le = _f.matches(receipt);
					if (le.size())
					{
#if ETH_DEBUG
						total += le.size();
#endif
						for (unsigned j = 0; j < le.size() && ret.size() != m; ++j)
						{
							if (s)
								s--;
							else
								ret.insert(ret.begin(), LocalisedLogEntry(le[j], n));
						}
					}
				}
#if ETH_DEBUG
				if (!total)
					falsePos++;
#endif
			}
#if ETH_DEBUG
		else
			skipped++;
#endif
		if (n == end)
			break;
	}
#if ETH_DEBUG
	cdebug << (begin - n) << "searched; " << skipped << "skipped; " << falsePos << "false +ves";
#endif
	return ret;
}
