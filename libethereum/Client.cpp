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
#include <boost/math/distributions/normal.hpp>
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
	bytes statusBytes = contents(m_path + "/status");
	RLP status(statusBytes);
	try
	{
		auto protocolVersion = (unsigned)status[0];
		auto minorProtocolVersion = (unsigned)status[1];
		auto databaseVersion = (unsigned)status[2];
		m_action =
			protocolVersion != eth::c_protocolVersion || databaseVersion != c_databaseVersion ?
				WithExisting::Kill
			: minorProtocolVersion != eth::c_minorProtocolVersion ?
				WithExisting::Verify
			:
				WithExisting::Trust;
	}
	catch (...)
	{
		m_action = WithExisting::Kill;
	}
}

void VersionChecker::setOk()
{
	if (m_action != WithExisting::Trust)
	{
		try
		{
			boost::filesystem::create_directory(m_path);
		}
		catch (...)
		{
			cwarn << "Unhandled exception! Failed to create directory: " << m_path << "\n" << boost::current_exception_diagnostic_information();
		}
		writeFile(m_path + "/status", rlpList(eth::c_protocolVersion, eth::c_minorProtocolVersion, c_databaseVersion));
	}
}

void BasicGasPricer::update(BlockChain const& _bc)
{
	unsigned c = 0;
	h256 p = _bc.currentHash();
	m_gasPerBlock = _bc.info(p).gasLimit;

	map<u256, u256> dist;
	u256 total = 0;

	// make gasPrice versus gasUsed distribution for the last 1000 blocks
	while (c < 1000 && p)
	{
		BlockInfo bi = _bc.info(p);
		if (bi.transactionsRoot != EmptyTrie)
		{
			auto bb = _bc.block(p);
			RLP r(bb);
			BlockReceipts brs(_bc.receipts(bi.hash()));
			size_t i = 0;
			for (auto const& tr: r[1])
			{
				Transaction tx(tr.data(), CheckTransaction::None);
				u256 gu = brs.receipts[i].gasUsed();
				dist[tx.gasPrice()] += gu;
				total += gu;
				i++;
			}
		}
		p = bi.parentHash;
		++c;
	}

	// fill m_octiles with weighted gasPrices
	if (total > 0)
	{
		m_octiles[0] = dist.begin()->first;

		// calc mean
		u256 mean = 0;
		for (auto const& i: dist)
			mean += i.first * i.second;
		mean /= total;

		// calc standard deviation
		u256 sdSquared = 0;
		for (auto const& i: dist)
			sdSquared += i.second * (i.first - mean) * (i.first - mean);
		sdSquared /= total;

		if (sdSquared)
		{
			long double sd = sqrt(sdSquared.convert_to<long double>());
			long double normalizedSd = sd / mean.convert_to<long double>();

			// calc octiles normalized to gaussian distribution
			boost::math::normal gauss(1.0, (normalizedSd > 0.01) ? normalizedSd : 0.01);
			for (size_t i = 1; i < 8; i++)
				m_octiles[i] = u256(mean.convert_to<long double>() * boost::math::quantile(gauss, i / 8.0));
			m_octiles[8] = dist.rbegin()->first;
		}
		else
		{
			for (size_t i = 0; i < 9; i++)
				m_octiles[i] = (i + 1) * mean / 5;
		}
	}
}

std::ostream& dev::eth::operator<<(std::ostream& _out, ActivityReport const& _r)
{
	_out << "Since " << toString(_r.since) << " (" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - _r.since).count();
	_out << "): " << _r.ticks << "ticks";
	return _out;
}

#ifdef _WIN32
const char* ClientNote::name() { return EthTeal "^" EthBlue " i"; }
const char* ClientChat::name() { return EthTeal "^" EthWhite " o"; }
const char* ClientTrace::name() { return EthTeal "^" EthGray " O"; }
const char* ClientDetail::name() { return EthTeal "^" EthCoal " 0"; }
#else
const char* ClientNote::name() { return EthTeal "⧫" EthBlue " ℹ"; }
const char* ClientChat::name() { return EthTeal "⧫" EthWhite " ◌"; }
const char* ClientTrace::name() { return EthTeal "⧫" EthGray " ◎"; }
const char* ClientDetail::name() { return EthTeal "⧫" EthCoal " ●"; }
#endif

Client::Client(p2p::Host* _extNet, std::string const& _dbPath, WithExisting _forceAction, u256 _networkId):
	Client(_extNet, make_shared<TrivialGasPricer>(), _dbPath, _forceAction, _networkId)
{
	startWorking();
}

Client::Client(p2p::Host* _extNet, std::shared_ptr<GasPricer> _gp, std::string const& _dbPath, WithExisting _forceAction, u256 _networkId):
	Worker("eth"),
	m_vc(_dbPath),
	m_bc(_dbPath, max(m_vc.action(), _forceAction), [](unsigned d, unsigned t){ cerr << "REVISING BLOCKCHAIN: Processed " << d << " of " << t << "...\r"; }),
	m_gp(_gp),
	m_stateDB(State::openDB(_dbPath, max(m_vc.action(), _forceAction))),
	m_preMine(m_stateDB, BaseState::CanonGenesis),
	m_postMine(m_stateDB)
{
	m_lastGetWork = std::chrono::system_clock::now() - chrono::seconds(30);
	m_tqReady = m_tq.onReady([=](){ this->onTransactionQueueReady(); });	// TODO: should read m_tq->onReady(thisThread, syncTransactionQueue);
	m_bqReady = m_bq.onReady([=](){ this->onBlockQueueReady(); });			// TODO: should read m_bq->onReady(thisThread, syncBlockQueue);
	m_farm.onSolutionFound([=](ProofOfWork::Solution const& s){ return this->submitWork(s); });

	m_gp->update(m_bc);


	auto host = _extNet->registerCapability(new EthereumHost(m_bc, m_tq, m_bq, _networkId));
	m_host = host;
	_extNet->addCapability(host, EthereumHost::staticName(), EthereumHost::staticVersion() - 1);

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

void Client::startedWorking()
{
	// Synchronise the state according to the head of the block chain.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	cdebug << "startedWorking()";

	DEV_WRITE_GUARDED(x_preMine)
		m_preMine.sync(m_bc);
	DEV_READ_GUARDED(x_preMine)
	{
		DEV_WRITE_GUARDED(x_working)
			m_working = m_preMine;
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_preMine;
	}
}

void Client::doneWorking()
{
	// Synchronise the state according to the head of the block chain.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	DEV_WRITE_GUARDED(x_preMine)
		m_preMine.sync(m_bc);
	DEV_READ_GUARDED(x_preMine)
	{
		DEV_WRITE_GUARDED(x_working)
			m_working = m_preMine;
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_preMine;
	}
}

void Client::killChain()
{
	bool wasMining = isMining();
	if (wasMining)
		stopMining();
	stopWorking();

	m_tq.clear();
	m_bq.clear();
	m_farm.stop();

	{
		WriteGuard l(x_postMine);
		WriteGuard l2(x_preMine);

		m_preMine = State();
		m_postMine = State();

		m_stateDB = OverlayDB();
		m_stateDB = State::openDB(Defaults::dbPath(), WithExisting::Kill);
		m_bc.reopen(Defaults::dbPath(), WithExisting::Kill);

		m_preMine = State(m_stateDB, BaseState::CanonGenesis);
		m_postMine = State(m_stateDB);
	}

	if (auto h = m_host.lock())
		h->reset();

	doWork();

	startWorking();
	if (wasMining)
		startMining();
}

void Client::clearPending()
{
	h256Hash changeds;
	DEV_WRITE_GUARDED(x_postMine)
	{
		if (!m_postMine.pending().size())
			return;
//		for (unsigned i = 0; i < m_postMine.pending().size(); ++i)
//			appendFromNewPending(m_postMine.logBloom(i), changeds);
		changeds.insert(PendingChangedFilter);
		m_tq.clear();
		DEV_READ_GUARDED(x_preMine)
			m_postMine = m_preMine;
	}

	startMining();

	noteChanged(changeds);
}

template <class S, class T>
static S& filtersStreamOut(S& _out, T const& _fs)
{
	_out << "{";
	unsigned i = 0;
	for (h256 const& f: _fs)
	{
		_out << (i++ ? ", " : "");
		if (f == PendingChangedFilter)
			_out << LogTag::Special << "pending";
		else if (f == ChainChangedFilter)
			_out << LogTag::Special << "chain";
		else
			_out << f;
	}
	_out << "}";
	return _out;
}

void Client::appendFromNewPending(TransactionReceipt const& _receipt, h256Hash& io_changed, h256 _transactionHash)
{
	Guard l(x_filtersWatches);
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

void Client::appendFromNewBlock(h256 const& _block, h256Hash& io_changed)
{
	// TODO: more precise check on whether the txs match.
	auto d = m_bc.info(_block);
	auto br = m_bc.receipts(_block);

	Guard l(x_filtersWatches);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
		if (i.second.filter.envelops(RelativeBlock::Latest, d.number) && i.second.filter.matches(d.logBloom))
			// acceptable number & looks like block may contain a matching log entry.
			for (size_t j = 0; j < br.receipts.size(); j++)
			{
				auto tr = br.receipts[j];
				auto m = i.second.filter.matches(tr);
				if (m.size())
				{
					auto transactionHash = transaction(d.hash(), j).sha3();
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
	 if (isMining())
		startMining();
}

MiningProgress Client::miningProgress() const
{
	if (m_farm.isMining())
		return m_farm.miningProgress();
	return MiningProgress();
}

uint64_t Client::hashrate() const
{
	if (m_farm.isMining())
		return m_farm.miningProgress().rate();
	return 0;
}

std::list<MineInfo> Client::miningHistory()
{
	std::list<MineInfo> ret;
/*	ReadGuard l(x_localMiners);
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
	}*/
	return ret;
}

ExecutionResult Client::call(Address _dest, bytes const& _data, u256 _gas, u256 _value, u256 _gasPrice, Address const& _from)
{
	ExecutionResult ret;
	try
	{
		State temp;
//		cdebug << "Nonce at " << toAddress(_secret) << " pre:" << m_preMine.transactionsFrom(toAddress(_secret)) << " post:" << m_postMine.transactionsFrom(toAddress(_secret));
		DEV_READ_GUARDED(x_postMine)
			temp = m_postMine;
		temp.addBalance(_from, _value + _gasPrice * _gas);
		Executive e(temp, LastHashes(), 0);
		if (!e.call(_dest, _dest, _from, _value, _gasPrice, &_data, _gas, _from))
			e.go();
		ret = e.executionResult();
	}
	catch (...)
	{
		// TODO: Some sort of notification of failure.
	}
	return ret;
}

ProofOfWork::WorkPackage Client::getWork()
{
	// lock the work so a later submission isn't invalidated by processing a transaction elsewhere.
	// this will be reset as soon as a new block arrives, allowing more transactions to be processed.
	bool oldShould = shouldServeWork();
	m_lastGetWork = chrono::system_clock::now();

	// if this request has made us bother to serve work, prep it now.
	if (!oldShould && shouldServeWork())
		onPostStateChanged();
	else
		// otherwise, set this to true so that it gets prepped next time.
		m_remoteWorking = true;
	return ProofOfWork::package(m_miningInfo);
}

bool Client::submitWork(ProofOfWork::Solution const& _solution)
{
	bytes newBlock;
	DEV_WRITE_GUARDED(x_working)
		if (!m_working.completeMine<ProofOfWork>(_solution))
			return false;

	DEV_READ_GUARDED(x_working)
	{
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_working;
		newBlock = m_working.blockData();
	}

	// OPTIMISE: very inefficient to not utilise the existing OverlayDB in m_postMine that contains all trie changes.
	m_bq.import(&newBlock, m_bc, true);

	return true;
}

void Client::syncBlockQueue()
{
	ImportRoute ir;

	cwork << "BQ ==> CHAIN ==> STATE";
	{
		tie(ir.first, ir.second, m_syncBlockQueue) = m_bc.sync(m_bq, m_stateDB, rand() % 90 + 10);
		if (ir.first.empty())
			return;
	}
	onChainChanged(ir);
}

void Client::syncTransactionQueue()
{
	// returns TransactionReceipts, once for each transaction.
	cwork << "postSTATE <== TQ";

	h256Hash changeds;
	TransactionReceipts newPendingReceipts;

	DEV_WRITE_GUARDED(x_working)
		tie(newPendingReceipts, m_syncTransactionQueue) = m_working.sync(m_bc, m_tq, *m_gp);

	if (newPendingReceipts.empty())
		return;

	DEV_READ_GUARDED(x_working)
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_working;

	DEV_READ_GUARDED(x_postMine)
		for (size_t i = 0; i < newPendingReceipts.size(); i++)
			appendFromNewPending(newPendingReceipts[i], changeds, m_postMine.pending()[i].sha3());
	changeds.insert(PendingChangedFilter);

	// Tell farm about new transaction (i.e. restartProofOfWork mining).
	onPostStateChanged();

	// Tell watches about the new transactions.
	noteChanged(changeds);

	// Tell network about the new transactions.
	if (auto h = m_host.lock())
		h->noteNewTransactions();
}

void Client::onChainChanged(ImportRoute const& _ir)
{
	// insert transactions that we are declaring the dead part of the chain
	for (auto const& h: _ir.second)
	{
		clog(ClientNote) << "Dead block:" << h;
		for (auto const& t: m_bc.transactions(h))
		{
			clog(ClientNote) << "Resubmitting dead-block transaction " << Transaction(t, CheckTransaction::None);
			m_tq.import(t, TransactionQueue::ImportCallback(), IfDropped::Retry);
		}
	}

	// remove transactions from m_tq nicely rather than relying on out of date nonce later on.
	for (auto const& h: _ir.first)
	{
		clog(ClientChat) << "Live block:" << h;
		for (auto const& th: m_bc.transactionHashes(h))
		{
			clog(ClientNote) << "Safely dropping transaction " << th;
			m_tq.drop(th);
		}
	}

	if (auto h = m_host.lock())
		h->noteNewBlocks();

	h256Hash changeds;
	for (auto const& h: _ir.first)
		appendFromNewBlock(h, changeds);
	changeds.insert(ChainChangedFilter);

	// RESTART MINING

	bool preChanged = false;
	State newPreMine;
	DEV_READ_GUARDED(x_preMine)
		newPreMine = m_preMine;

	// TODO: use m_postMine to avoid re-evaluating our own blocks.
	preChanged = newPreMine.sync(m_bc);

	if (preChanged || m_postMine.address() != m_preMine.address())
	{
		if (isMining())
			cnote << "New block on chain.";

		DEV_WRITE_GUARDED(x_preMine)
			m_preMine = newPreMine;
		DEV_WRITE_GUARDED(x_working)
			m_working = newPreMine;
		DEV_READ_GUARDED(x_postMine)
			for (auto const& t: m_postMine.pending())
			{
				clog(ClientNote) << "Resubmitting post-mine transaction " << t;
				auto ir = m_tq.import(t, TransactionQueue::ImportCallback(), IfDropped::Retry);
				if (ir != ImportResult::Success)
					onTransactionQueueReady();
			}
		DEV_READ_GUARDED(x_working) DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_working;

		changeds.insert(PendingChangedFilter);

		onPostStateChanged();
	}

	// Quick hack for now - the TQ at this point already has the prior pending transactions in it;
	// we should resync with it manually until we are stricter about what constitutes "knowing".
	onTransactionQueueReady();

	noteChanged(changeds);
}

bool Client::remoteActive() const
{
	return chrono::system_clock::now() - m_lastGetWork < chrono::seconds(30);
}

void Client::onPostStateChanged()
{
	cnote << "Post state changed";

	if (m_bq.items().first == 0 && (isMining() || remoteActive()))
	{
		cnote << "Restarting mining...";
		DEV_WRITE_GUARDED(x_working)
			m_working.commitToMine(m_bc);
		DEV_READ_GUARDED(x_working)
		{
			DEV_WRITE_GUARDED(x_postMine)
				m_postMine = m_working;
			m_miningInfo = m_postMine.info();
		}
		m_farm.setWork(m_miningInfo);

		Ethash::ensurePrecomputed(m_bc.number());
	}
	m_remoteWorking = false;
}

void Client::startMining()
{
	if (m_turboMining)
		m_farm.startGPU();
	else
		m_farm.startCPU();
	onPostStateChanged();
}

void Client::noteChanged(h256Hash const& _filters)
{
	Guard l(x_filtersWatches);
	if (_filters.size())
		filtersStreamOut(cwatch << "noteChanged:", _filters);
	// accrue all changes left in each filter into the watches.
	for (auto& w: m_watches)
		if (_filters.count(w.second.id))
		{
			if (m_filters.count(w.second.id))
			{
				cwatch << "!!!" << w.first << w.second.id.abridged();
				w.second.changes += m_filters.at(w.second.id).changes;
			}
			else
			{
				cwatch << "!!!" << w.first << LogTag::Special << (w.second.id == PendingChangedFilter ? "pending" : w.second.id == ChainChangedFilter ? "chain" : "???");
				w.second.changes.push_back(LocalisedLogEntry(SpecialLogEntry, 0));
			}
		}
	// clear the filters now.
	for (auto& i: m_filters)
		i.second.changes.clear();
}

void Client::doWork()
{
	bool t = true;
	if (m_syncBlockQueue.compare_exchange_strong(t, false))
		syncBlockQueue();

	t = true;
	if (m_syncTransactionQueue.compare_exchange_strong(t, false) && !m_remoteWorking)
		syncTransactionQueue();

	tick();

	if (!m_syncBlockQueue && !m_syncTransactionQueue)
	{
		std::unique_lock<std::mutex> l(x_signalled);
		m_signalled.wait_for(l, chrono::seconds(1));
	}
}

void Client::tick()
{
	if (chrono::system_clock::now() - m_lastTick > chrono::seconds(1))
	{
		m_report.ticks++;
		checkWatchGarbage();
		m_bq.tick(m_bc);
		m_lastTick = chrono::system_clock::now();
		if (m_report.ticks == 15)
			clog(ClientTrace) << activityReport();
	}
}

void Client::checkWatchGarbage()
{
	if (chrono::system_clock::now() - m_lastGarbageCollection > chrono::seconds(5))
	{
		// watches garbage collection
		vector<unsigned> toUninstall;
		DEV_GUARDED(x_filtersWatches)
			for (auto key: keysOf(m_watches))
				if (m_watches[key].lastPoll != chrono::system_clock::time_point::max() && chrono::system_clock::now() - m_watches[key].lastPoll > chrono::seconds(20))
				{
					toUninstall.push_back(key);
					cnote << "GC: Uninstall" << key << "(" << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now() - m_watches[key].lastPoll).count() << "s old)";
				}
		for (auto i: toUninstall)
			uninstallWatch(i);

		// blockchain GC
		m_bc.garbageCollect();

		m_lastGarbageCollection = chrono::system_clock::now();
	}
}

State Client::asOf(h256 const& _block) const
{
	return State(m_stateDB, bc(), _block);
}

void Client::prepareForTransaction()
{
	startWorking();
}

State Client::state(unsigned _txi, h256 _block) const
{
	return State(m_stateDB, m_bc, _block).fromPending(_txi);
}

eth::State Client::state(h256 _block) const
{
	return State(m_stateDB, m_bc, _block);
}

eth::State Client::state(unsigned _txi) const
{
	DEV_READ_GUARDED(x_postMine)
		return m_postMine.fromPending(_txi);
	assert(false);
	return State();
}

void Client::flushTransactions()
{
	doWork();
}
