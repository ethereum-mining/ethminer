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

Client::Client(p2p::Host* _extNet, std::string const& _dbPath, bool _forceClean, u256 _networkId):
	Worker("eth"),
	m_vc(_dbPath),
	m_bc(_dbPath, !m_vc.ok() || _forceClean),
	m_stateDB(State::openDB(_dbPath, !m_vc.ok() || _forceClean)),
	m_preMine(Address(), m_stateDB),
	m_postMine(Address(), m_stateDB)
{
	m_host = _extNet->registerCapability(new EthereumHost(m_bc, m_tq, m_bq, _networkId));

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

DownloadMan const* Client::downloadMan() const { if (auto h = m_host.lock()) return &(h->downloadMan()); return nullptr; }
bool Client::isSyncing() const { if (auto h = m_host.lock()) return h->isSyncing(); return false; }

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
	m_miners.clear();
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
		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
			appendFromNewPending(m_postMine.bloom(i), changeds);
		changeds.insert(PendingChangedFilter);
		m_postMine = m_preMine;
	}

	{
		ReadGuard l(x_miners);
		for (auto& m: m_miners)
			m.noteStateChange();
	}

	noteChanged(changeds);
}

unsigned Client::installWatch(h256 _h)
{
	auto ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
	m_watches[ret] = ClientWatch(_h);
	cwatch << "+++" << ret << _h;
	return ret;
}

unsigned Client::installWatch(MessageFilter const& _f)
{
	lock_guard<mutex> l(m_filterLock);

	h256 h = _f.sha3();

	if (!m_filters.count(h))
		m_filters.insert(make_pair(h, _f));

	return installWatch(h);
}

void Client::uninstallWatch(unsigned _i)
{
	cwatch << "XXX" << _i;

	lock_guard<mutex> l(m_filterLock);

	auto it = m_watches.find(_i);
	if (it == m_watches.end())
		return;
	auto id = it->second.id;
	m_watches.erase(it);

	auto fit = m_filters.find(id);
	if (fit != m_filters.end())
		if (!--fit->second.refCount)
			m_filters.erase(fit);
}

void Client::noteChanged(h256Set const& _filters)
{
	lock_guard<mutex> l(m_filterLock);
	for (auto& i: m_watches)
		if (_filters.count(i.second.id))
		{
			cwatch << "!!!" << i.first << i.second.id;
			i.second.changes++;
		}
}

void Client::appendFromNewPending(h256 _bloom, h256Set& o_changed) const
{
	lock_guard<mutex> l(m_filterLock);
	for (pair<h256, InstalledFilter> const& i: m_filters)
		if ((unsigned)i.second.filter.latest() > m_bc.number() && i.second.filter.matches(_bloom))
			o_changed.insert(i.first);
}

void Client::appendFromNewBlock(h256 _block, h256Set& o_changed) const
{
	auto d = m_bc.details(_block);

	lock_guard<mutex> l(m_filterLock);
	for (pair<h256, InstalledFilter> const& i: m_filters)
		if ((unsigned)i.second.filter.latest() >= d.number && (unsigned)i.second.filter.earliest() <= d.number && i.second.filter.matches(d.bloom))
			o_changed.insert(i.first);
}

void Client::setForceMining(bool _enable)
{
	 m_forceMining = _enable;
	 ReadGuard l(x_miners);
	 for (auto& m: m_miners)
		 m.noteStateChange();
}

void Client::setMiningThreads(unsigned _threads)
{
	stopMining();

	auto t = _threads ? _threads : thread::hardware_concurrency();
	WriteGuard l(x_miners);
	m_miners.clear();
	m_miners.resize(t);
	unsigned i = 0;
	for (auto& m: m_miners)
		m.setup(this, i++);
}

MineProgress Client::miningProgress() const
{
	MineProgress ret;
	ReadGuard l(x_miners);
	for (auto& m: m_miners)
		ret.combine(m.miningProgress());
	return ret;
}

std::list<MineInfo> Client::miningHistory()
{
	std::list<MineInfo> ret;

	ReadGuard l(x_miners);
	if (m_miners.empty())
		return ret;
	ret = m_miners[0].miningHistory();
	for (unsigned i = 1; i < m_miners.size(); ++i)
	{
		auto l = m_miners[i].miningHistory();
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

	Transaction t;
//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
	{
		ReadGuard l(x_stateDB);
		t.nonce = m_postMine.transactionsFrom(toAddress(_secret));
	}
	t.value = _value;
	t.gasPrice = _gasPrice;
	t.gas = _gas;
	t.receiveAddress = _dest;
	t.data = _data;
	t.sign(_secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
}

bytes Client::call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	bytes out;
	try
	{
		State temp;
		Transaction t;
	//	cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		{
			ReadGuard l(x_stateDB);
			temp = m_postMine;
			t.nonce = temp.transactionsFrom(toAddress(_secret));
		}
		t.value = _value;
		t.gasPrice = _gasPrice;
		t.gas = _gas;
		t.receiveAddress = _dest;
		t.data = _data;
		t.sign(_secret);
		u256 gasUsed = temp.execute(t.data, &out, false);
		(void)gasUsed; // TODO: do something with gasused which it returns.
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return out;
}

Address Client::transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	startWorking();

	Transaction t;
	{
		ReadGuard l(x_stateDB);
		t.nonce = m_postMine.transactionsFrom(toAddress(_secret));
	}
	t.value = _endowment;
	t.gasPrice = _gasPrice;
	t.gas = _gas;
	t.receiveAddress = Address();
	t.data = _init;
	t.sign(_secret);
	cnote << "New transaction " << t;
	m_tq.attemptImport(t.rlp());
	return right160(sha3(rlpList(t.sender(), t.nonce)));
}

void Client::inject(bytesConstRef _rlp)
{
	startWorking();

	m_tq.attemptImport(_rlp);
}

void Client::doWork()
{
	// TODO: Use condition variable rather than polling.

	cworkin << "WORK";
	h256Set changeds;

	{
		ReadGuard l(x_miners);
		for (auto& m: m_miners)
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
					for (auto h: hs)
						appendFromNewBlock(h, changeds);
					changeds.insert(ChainChangedFilter);
					//changeds.insert(PendingChangedFilter);	// if we mined the new block, then we've probably reset the pending transactions.
				}
				for (auto& m: m_miners)
					m.noteStateChange();
			}
	}

	// Synchronise state to block chain.
	// This should remove any transactions on our queue that are included within our state.
	// It also guarantees that the state reflects the longest (valid!) chain on the block chain.
	//   This might mean reverting to an earlier state and replaying some blocks, or, (worst-case:
	//   if there are no checkpoints before our fork) reverting to the genesis block and replaying
	//   all blocks.
	// Resynchronise state with block chain & trans
	bool rsm = false;
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
			rsm = true;
			changeds.insert(PendingChangedFilter);
			// TODO: Move transactions pending from m_postMine back to transaction queue.
		}

		// returns h256s as blooms, once for each transaction.
		cwork << "postSTATE <== TQ";
		h256s newPendingBlooms = m_postMine.sync(m_tq);
		if (newPendingBlooms.size())
		{
			for (auto i: newPendingBlooms)
				appendFromNewPending(i, changeds);
			changeds.insert(PendingChangedFilter);

			if (isMining())
				cnote << "Additional transaction ready: Restarting mining operation.";
			rsm = true;
		}
	}
	if (rsm)
	{
		ReadGuard l(x_miners);
		for (auto& m: m_miners)
			m.noteStateChange();
	}

	cwork << "noteChanged" << changeds.size() << "items";
	noteChanged(changeds);
	cworkout << "WORK";

	this_thread::sleep_for(chrono::milliseconds(100));
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
	State st = state(_block);
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
	return Transaction(b[1][_i].data());
}

BlockInfo Client::uncle(h256 _blockHash, unsigned _i) const
{
	auto bl = m_bc.block(_blockHash);
	RLP b(bl);
	return BlockInfo::fromHeader(b[2][_i].data());
}

PastMessages Client::messages(MessageFilter const& _f) const
{
	PastMessages ret;
	unsigned begin = min<unsigned>(m_bc.number(), (unsigned)_f.latest());
	unsigned end = min(begin, (unsigned)_f.earliest());
	unsigned m = _f.max();
	unsigned s = _f.skip();

	// Handle pending transactions differently as they're not on the block chain.
	if (begin == m_bc.number())
	{
		ReadGuard l(x_stateDB);
		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
		{
			// Might have a transaction that contains a matching message.
			Manifest const& ms = m_postMine.changesFromPending(i);
			PastMessages pm = _f.matches(ms, i);
			if (pm.size())
			{
				auto ts = time(0);
				for (unsigned j = 0; j < pm.size() && ret.size() != m; ++j)
					if (s)
						s--;
					else
						// Have a transaction that contains a matching message.
						ret.insert(ret.begin(), pm[j].polish(h256(), ts, m_bc.number() + 1, m_postMine.address()));
			}
		}
	}

#if ETH_DEBUG
	unsigned skipped = 0;
	unsigned falsePos = 0;
#endif
	auto h = m_bc.numberHash(begin);
	unsigned n = begin;
	for (; ret.size() != m && n != end; n--, h = m_bc.details(h).parent)
	{
		auto d = m_bc.details(h);
#if ETH_DEBUG
		int total = 0;
#endif
		if (_f.matches(d.bloom))
		{
			// Might have a block that contains a transaction that contains a matching message.
			auto bs = m_bc.blooms(h).blooms;
			Manifests ms;
			BlockInfo bi;
			for (unsigned i = 0; i < bs.size(); ++i)
				if (_f.matches(bs[i]))
				{
					// Might have a transaction that contains a matching message.
					if (ms.empty())
						ms = m_bc.traces(h).traces;
					Manifest const& changes = ms[i];
					PastMessages pm = _f.matches(changes, i);
					if (pm.size())
					{
#if ETH_DEBUG
						total += pm.size();
#endif
						if (!bi)
							bi.populate(m_bc.block(h));
						auto ts = bi.timestamp;
						auto cb = bi.coinbaseAddress;
						for (unsigned j = 0; j < pm.size() && ret.size() != m; ++j)
							if (s)
								s--;
							else
								// Have a transaction that contains a matching message.
								ret.push_back(pm[j].polish(h, ts, n, cb));
					}
				}
#if ETH_DEBUG
			if (!total)
				falsePos++;
		}
		else
			skipped++;
#else
		}
#endif
		if (n == end)
			break;
	}
#if ETH_DEBUG
//	cdebug << (begin - n) << "searched; " << skipped << "skipped; " << falsePos << "false +ves";
#endif
	return ret;
}
