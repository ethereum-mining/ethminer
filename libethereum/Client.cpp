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
#include <memory>
#include <thread>
#include <boost/filesystem.hpp>
#if ETH_JSONRPC || !ETH_TRUE
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#endif
#include <libdevcore/Log.h>
#include <libdevcore/StructuredLogger.h>
#include <libp2p/Host.h>
#include <libethcore/Ethash.h>
#if ETH_JSONRPC || !ETH_TRUE
#include "Sentinel.h"
#endif
#include "Defaults.h"
#include "Executive.h"
#include "EthereumHost.h"
#include "Utility.h"
#include "TransactionQueue.h"

using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace p2p;

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

static const Addresses c_canaries =
{
	Address("4bb7e8ae99b645c2b7860b8f3a2328aae28bd80a"),		// gav
	Address("1baf27b88c48dd02b744999cf3522766929d2b2a"),		// vitalik
	Address("a8edb1ac2c86d3d9d78f96cd18001f60df29e52c"),		// jeff
	Address("ace7813896a84d3f5f80223916a5353ab16e46e6")			// christoph
};

Client::Client(std::shared_ptr<GasPricer> _gp):
	Worker("eth", 0),
	m_gp(_gp ? _gp : make_shared<TrivialGasPricer>())
{
}

void Client::init(p2p::Host* _extNet, std::string const& _dbPath, WithExisting _forceAction, u256 _networkId)
{
	// Cannot be opened until after blockchain is open, since BlockChain may upgrade the database.
	// TODO: consider returning the upgrade mechanism here. will delaying the opening of the blockchain database
	// until after the construction.
	m_stateDB = State::openDB(_dbPath, bc().genesisHash(), _forceAction);
	// LAZY. TODO: move genesis state construction/commiting to stateDB openning and have this just take the root from the genesis block.
	m_preMine = bc().genesisState(m_stateDB);
	m_postMine = m_preMine;

	m_bq.setChain(bc());

	m_lastGetWork = std::chrono::system_clock::now() - chrono::seconds(30);
	m_tqReady = m_tq.onReady([=](){ this->onTransactionQueueReady(); });	// TODO: should read m_tq->onReady(thisThread, syncTransactionQueue);
	m_tqReplaced = m_tq.onReplaced([=](h256 const&){ m_needStateReset = true; });
	m_bqReady = m_bq.onReady([=](){ this->onBlockQueueReady(); });			// TODO: should read m_bq->onReady(thisThread, syncBlockQueue);
	m_bq.setOnBad([=](Exception& ex){ this->onBadBlock(ex); });
	bc().setOnBad([=](Exception& ex){ this->onBadBlock(ex); });

	if (_forceAction == WithExisting::Rescue)
		bc().rescue(m_stateDB);

	m_gp->update(bc());

	auto host = _extNet->registerCapability(new EthereumHost(bc(), m_tq, m_bq, _networkId));
	m_host = host;
	_extNet->addCapability(host, EthereumHost::staticName(), EthereumHost::c_oldProtocolVersion); //TODO: remove this one v61+ protocol is common

	if (_dbPath.size())
		Defaults::setDBPath(_dbPath);
	doWork();
	startWorking();
}

Client::~Client()
{
	stopWorking();
}

ImportResult Client::queueBlock(bytes const& _block, bool _isSafe)
{
	if (m_bq.status().verified + m_bq.status().verifying + m_bq.status().unverified > 10000)
		this_thread::sleep_for(std::chrono::milliseconds(500));
	return m_bq.import(&_block, _isSafe);
}

tuple<ImportRoute, bool, unsigned> Client::syncQueue(unsigned _max)
{
	stopWorking();
	return bc().sync(m_bq, m_stateDB, _max);
}

void Client::onBadBlock(Exception& _ex) const
{
	// BAD BLOCK!!!
	bytes const* block = boost::get_error_info<errinfo_block>(_ex);
	if (!block)
	{
		cwarn << "ODD: onBadBlock called but exception (" << _ex.what() << ") has no block in it.";
		cwarn << boost::diagnostic_information(_ex, true);
		return;
	}

	badBlock(*block, _ex.what());

#if ETH_JSONRPC || !ETH_TRUE
	Json::Value report;

	report["client"] = "cpp";
	report["version"] = Version;
	report["protocolVersion"] = c_protocolVersion;
	report["databaseVersion"] = c_databaseVersion;
	report["errortype"] = _ex.what();
	report["block"] = toHex(*block);

	// add the various hints.
	if (unsigned const* uncleIndex = boost::get_error_info<errinfo_uncleIndex>(_ex))
	{
		// uncle that failed.
		report["hints"]["uncleIndex"] = *uncleIndex;
	}
	else if (unsigned const* txIndex = boost::get_error_info<errinfo_transactionIndex>(_ex))
	{
		// transaction that failed.
		report["hints"]["transactionIndex"] = *txIndex;
	}
	else
	{
		// general block failure.
	}

	if (string const* vmtraceJson = boost::get_error_info<errinfo_vmtrace>(_ex))
		Json::Reader().parse(*vmtraceJson, report["hints"]["vmtrace"]);

	if (vector<bytes> const* receipts = boost::get_error_info<errinfo_receipts>(_ex))
	{
		report["hints"]["receipts"] = Json::arrayValue;
		for (auto const& r: *receipts)
			report["hints"]["receipts"].append(toHex(r));
	}
	if (h256Hash const* excluded = boost::get_error_info<errinfo_unclesExcluded>(_ex))
	{
		report["hints"]["unclesExcluded"] = Json::arrayValue;
		for (auto const& r: h256Set() + *excluded)
			report["hints"]["unclesExcluded"].append(Json::Value(r.hex()));
	}

#define DEV_HINT_ERRINFO(X) \
		if (auto const* n = boost::get_error_info<errinfo_ ## X>(_ex)) \
			report["hints"][#X] = toString(*n)
#define DEV_HINT_ERRINFO_HASH(X) \
		if (auto const* n = boost::get_error_info<errinfo_ ## X>(_ex)) \
			report["hints"][#X] = n->hex()

	DEV_HINT_ERRINFO_HASH(hash256);
	DEV_HINT_ERRINFO(uncleNumber);
	DEV_HINT_ERRINFO(currentNumber);
	DEV_HINT_ERRINFO(now);
	DEV_HINT_ERRINFO(invalidSymbol);
	DEV_HINT_ERRINFO(wrongAddress);
	DEV_HINT_ERRINFO(comment);
	DEV_HINT_ERRINFO(min);
	DEV_HINT_ERRINFO(max);
	DEV_HINT_ERRINFO(name);
	DEV_HINT_ERRINFO(field);
	DEV_HINT_ERRINFO(transaction);
	DEV_HINT_ERRINFO(data);
	DEV_HINT_ERRINFO(phase);
	DEV_HINT_ERRINFO_HASH(nonce);
	DEV_HINT_ERRINFO(difficulty);
	DEV_HINT_ERRINFO(target);
	DEV_HINT_ERRINFO_HASH(seedHash);
	DEV_HINT_ERRINFO_HASH(mixHash);
	if (tuple<h256, h256> const* r = boost::get_error_info<errinfo_ethashResult>(_ex))
	{
		report["hints"]["ethashResult"]["value"] = get<0>(*r).hex();
		report["hints"]["ethashResult"]["mixHash"] = get<1>(*r).hex();
	}
	if (bytes const* ed = boost::get_error_info<errinfo_extraData>(_ex))
	{
		report["hints"]["extraData"] = toHex(*ed);
		try
		{
			RLP r(*ed);
			if (r[0].toInt<int>() == 0)
				report["hints"]["minerVersion"] = r[1].toString();
		}
		catch (...) {}
	}
	DEV_HINT_ERRINFO(required);
	DEV_HINT_ERRINFO(got);
	DEV_HINT_ERRINFO_HASH(required_LogBloom);
	DEV_HINT_ERRINFO_HASH(got_LogBloom);
	DEV_HINT_ERRINFO_HASH(required_h256);
	DEV_HINT_ERRINFO_HASH(got_h256);

	cwarn << ("Report: \n" + Json::StyledWriter().write(report));

	if (!m_sentinel.empty())
	{
		jsonrpc::HttpClient client(m_sentinel);
		Sentinel rpc(client);
		try
		{
			rpc.eth_badBlock(report);
		}
		catch (...)
		{
			cwarn << "Error reporting to sentinel. Sure the address" << m_sentinel << "is correct?";
		}
	}
#endif
}

bool Client::isChainBad() const
{
	unsigned numberBad = 0;
	for (auto const& a: c_canaries)
		if (!!stateAt(a, 0))
			numberBad++;
	return numberBad >= 2;
}

bool Client::isUpgradeNeeded() const
{
	return stateAt(c_canaries[0], 0) == 2;
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

bool Client::isMajorSyncing() const
{
	// TODO: only return true if it is actually doing a proper chain sync.
	if (auto h = m_host.lock())
		return h->isSyncing();
	return false;
}

void Client::startedWorking()
{
	// Synchronise the state according to the head of the block chain.
	// TODO: currently it contains keys for *all* blocks. Make it remove old ones.
	clog(ClientTrace) << "startedWorking()";

	DEV_WRITE_GUARDED(x_preMine)
		m_preMine.sync(bc());
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
		m_preMine.sync(bc());
	DEV_READ_GUARDED(x_preMine)
	{
		DEV_WRITE_GUARDED(x_working)
			m_working = m_preMine;
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_preMine;
	}
}

void Client::reopenChain(WithExisting _we)
{
	bool wasMining = isMining();
	if (wasMining)
		stopMining();
	stopWorking();

	m_tq.clear();
	m_bq.clear();
	m_sealEngine->cancelGeneration();

	{
		WriteGuard l(x_postMine);
		WriteGuard l2(x_preMine);
		WriteGuard l3(x_working);

		m_preMine = State();
		m_postMine = State();
		m_working = State();

		m_stateDB = OverlayDB();
		bc().reopen(_we);
		m_stateDB = State::openDB(Defaults::dbPath(), bc().genesisHash(), _we);

		m_preMine = bc().genesisState(m_stateDB);
		m_postMine = m_preMine;
	}

	if (auto h = m_host.lock())
		h->reset();

	startedWorking();
	doWork();

	startWorking();
	if (wasMining)
		startMining();
}

void Client::clearPending()
{
	DEV_WRITE_GUARDED(x_postMine)
	{
		if (!m_postMine.pending().size())
			return;
		m_tq.clear();
		DEV_READ_GUARDED(x_preMine)
			m_postMine = m_preMine;
	}

	startMining();
	h256Hash changeds;
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

void Client::appendFromNewPending(TransactionReceipt const& _receipt, h256Hash& io_changed, h256 _sha3)
{
	Guard l(x_filtersWatches);
	io_changed.insert(PendingChangedFilter);
	m_specialFilters.at(PendingChangedFilter).push_back(_sha3);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
	{
		// acceptable number.
		auto m = i.second.filter.matches(_receipt);
		if (m.size())
		{
			// filter catches them
			for (LogEntry const& l: m)
				i.second.changes.push_back(LocalisedLogEntry(l));
			io_changed.insert(i.first);
		}
	}
}

void Client::appendFromBlock(h256 const& _block, BlockPolarity _polarity, h256Hash& io_changed)
{
	// TODO: more precise check on whether the txs match.
	auto receipts = bc().receipts(_block).receipts;

	Guard l(x_filtersWatches);
	io_changed.insert(ChainChangedFilter);
	m_specialFilters.at(ChainChangedFilter).push_back(_block);
	for (pair<h256 const, InstalledFilter>& i: m_filters)
	{
		// acceptable number & looks like block may contain a matching log entry.
		for (size_t j = 0; j < receipts.size(); j++)
		{
			auto tr = receipts[j];
			auto m = i.second.filter.matches(tr);
			if (m.size())
			{
				auto transactionHash = transaction(_block, j).sha3();
				// filter catches them
				for (LogEntry const& l: m)
					i.second.changes.push_back(LocalisedLogEntry(l, _block, (BlockNumber)bc().number(_block), transactionHash, j, 0, _polarity));
				io_changed.insert(i.first);
			}
		}
	}
}

void Client::setForceMining(bool _enable)
{
	 m_forceMining = _enable;
	 if (isMining())
		startMining();
}

void Client::setShouldPrecomputeDAG(bool _precompute)
{
	bytes trueBytes {1};
	bytes falseBytes {0};
	sealEngine()->setOption("precomputeDAG", _precompute ? trueBytes: falseBytes);
}

void Client::setTurboMining(bool _enable)
{
	m_turboMining = _enable;
#if ETH_ETHASHCL || !ETH_TRUE
	sealEngine()->setSealer(_enable ? "opencl" : "cpu");
#endif
	if (isMining())
		startMining();
}

bool Client::isMining() const
{
	return Ethash::isWorking(m_sealEngine.get());
}

WorkingProgress Client::miningProgress() const
{
	if (Ethash::isWorking(m_sealEngine.get()))
		return Ethash::workingProgress(m_sealEngine.get());
	return WorkingProgress();
}

uint64_t Client::hashrate() const
{
	if (Ethash::isWorking(m_sealEngine.get()))
		return Ethash::workingProgress(m_sealEngine.get()).rate();
	return 0;
}

std::list<MineInfo> Client::miningHistory()
{
	//TODO: reimplement for CPU/GPU miner
	return std::list<MineInfo> {};
}

ExecutionResult Client::call(Address _dest, bytes const& _data, u256 _gas, u256 _value, u256 _gasPrice, Address const& _from)
{
	ExecutionResult ret;
	try
	{
		State temp;
		clog(ClientDetail) << "Nonce at " << _dest << " pre:" << m_preMine.transactionsFrom(_dest) << " post:" << m_postMine.transactionsFrom(_dest);
		DEV_READ_GUARDED(x_postMine)
			temp = m_postMine;
		temp.addBalance(_from, _value + _gasPrice * _gas);
		Executive e(temp, LastHashes(), 0);
		e.setResultRecipient(ret);
		if (!e.call(_dest, _from, _value, _gasPrice, &_data, _gas))
			e.go();
		e.finalize();
	}
	catch (...)
	{
		cwarn << "Client::call failed: " << boost::current_exception_diagnostic_information();
	}
	return ret;
}

unsigned static const c_syncMin = 1;
unsigned static const c_syncMax = 1000;
double static const c_targetDuration = 1;

void Client::syncBlockQueue()
{
	cwork << "BQ ==> CHAIN ==> STATE";
	ImportRoute ir;
	unsigned count;
	Timer t;
	tie(ir, m_syncBlockQueue, count) = bc().sync(m_bq, m_stateDB, m_syncAmount);
	double elapsed = t.elapsed();

	if (count)
		clog(ClientNote) << count << "blocks imported in" << unsigned(elapsed * 1000) << "ms (" << (count / elapsed) << "blocks/s)";

	if (elapsed > c_targetDuration * 1.1 && count > c_syncMin)
		m_syncAmount = max(c_syncMin, count * 9 / 10);
	else if (count == m_syncAmount && elapsed < c_targetDuration * 0.9 && m_syncAmount < c_syncMax)
		m_syncAmount = min(c_syncMax, m_syncAmount * 11 / 10 + 1);
	if (ir.liveBlocks.empty())
		return;
	onChainChanged(ir);
}

void Client::syncTransactionQueue()
{
	// returns TransactionReceipts, once for each transaction.
	cwork << "postSTATE <== TQ";

	h256Hash changeds;
	TransactionReceipts newPendingReceipts;

	DEV_WRITE_GUARDED(x_working)
		tie(newPendingReceipts, m_syncTransactionQueue) = m_working.sync(bc(), m_tq, *m_gp);

	if (newPendingReceipts.empty())
		return;

	DEV_READ_GUARDED(x_working)
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_working;

	DEV_READ_GUARDED(x_postMine)
		for (size_t i = 0; i < newPendingReceipts.size(); i++)
			appendFromNewPending(newPendingReceipts[i], changeds, m_postMine.pending()[i].sha3());

	// Tell farm about new transaction (i.e. restart mining).
	onPostStateChanged();

	// Tell watches about the new transactions.
	noteChanged(changeds);

	// Tell network about the new transactions.
	if (auto h = m_host.lock())
		h->noteNewTransactions();
}

void Client::onDeadBlocks(h256s const& _blocks, h256Hash& io_changed)
{
	// insert transactions that we are declaring the dead part of the chain
	for (auto const& h: _blocks)
	{
		clog(ClientTrace) << "Dead block:" << h;
		for (auto const& t: bc().transactions(h))
		{
			clog(ClientTrace) << "Resubmitting dead-block transaction " << Transaction(t, CheckTransaction::None);
			m_tq.import(t, IfDropped::Retry);
		}
	}

	for (auto const& h: _blocks)
		appendFromBlock(h, BlockPolarity::Dead, io_changed);
}

void Client::onNewBlocks(h256s const& _blocks, h256Hash& io_changed)
{
	// remove transactions from m_tq nicely rather than relying on out of date nonce later on.
	for (auto const& h: _blocks)
		clog(ClientTrace) << "Live block:" << h;

	if (auto h = m_host.lock())
		h->noteNewBlocks();

	for (auto const& h: _blocks)
		appendFromBlock(h, BlockPolarity::Live, io_changed);
}

void Client::resyncStateFromChain()
{
	// RESTART MINING

	if (!isMajorSyncing())
	{
		bool preChanged = false;
		State newPreMine;
		DEV_READ_GUARDED(x_preMine)
			newPreMine = m_preMine;

		// TODO: use m_postMine to avoid re-evaluating our own blocks.
		preChanged = newPreMine.sync(bc());

		if (preChanged || m_postMine.address() != m_preMine.address())
		{
			if (isMining())
				clog(ClientTrace) << "New block on chain.";

			DEV_WRITE_GUARDED(x_preMine)
				m_preMine = newPreMine;
			DEV_WRITE_GUARDED(x_working)
				m_working = newPreMine;
			DEV_READ_GUARDED(x_postMine)
				for (auto const& t: m_postMine.pending())
				{
					clog(ClientTrace) << "Resubmitting post-mine transaction " << t;
					auto ir = m_tq.import(t, IfDropped::Retry);
					if (ir != ImportResult::Success)
						onTransactionQueueReady();
				}
			DEV_READ_GUARDED(x_working) DEV_WRITE_GUARDED(x_postMine)
				m_postMine = m_working;

			onPostStateChanged();
		}

		// Quick hack for now - the TQ at this point already has the prior pending transactions in it;
		// we should resync with it manually until we are stricter about what constitutes "knowing".
		onTransactionQueueReady();
	}
}

void Client::resetState()
{
	State newPreMine;
	DEV_READ_GUARDED(x_preMine)
		newPreMine = m_preMine;

	DEV_WRITE_GUARDED(x_working)
		m_working = newPreMine;
	DEV_READ_GUARDED(x_working) DEV_WRITE_GUARDED(x_postMine)
		m_postMine = m_working;

	onPostStateChanged();
	onTransactionQueueReady();
}

void Client::onChainChanged(ImportRoute const& _ir)
{
	h256Hash changeds;
	onDeadBlocks(_ir.deadBlocks, changeds);
	for (auto const& t: _ir.goodTranactions)
	{
		clog(ClientTrace) << "Safely dropping transaction " << t.sha3();
		m_tq.dropGood(t);
	}
	onNewBlocks(_ir.liveBlocks, changeds);
	resyncStateFromChain();
	noteChanged(changeds);
}

bool Client::remoteActive() const
{
	return chrono::system_clock::now() - m_lastGetWork < chrono::seconds(30);
}

void Client::onPostStateChanged()
{
	clog(ClientTrace) << "Post state changed.";
	rejigMining();
	m_remoteWorking = false;
}

void Client::startMining()
{
	clog(ClientNote) << "MiningBenefactor: " << address();
	if (address() != Address())
	{
		m_wouldMine = true;
		rejigMining();
	}
	else
		clog(ClientNote) << "You need to set a MiningBenefactor in order to mine!";
}

void Client::rejigMining()
{
	if ((wouldMine() || remoteActive()) && !isMajorSyncing() && (!isChainBad() || mineOnBadChain()) /*&& (forceMining() || transactionsWaiting())*/)
	{
		clog(ClientTrace) << "Rejigging mining...";
		DEV_WRITE_GUARDED(x_working)
			m_working.commitToMine(bc(), m_extraData);
		DEV_READ_GUARDED(x_working)
		{
			DEV_WRITE_GUARDED(x_postMine)
				m_postMine = m_working;
			m_miningInfo = m_postMine.info();
		}

		if (m_wouldMine)
			m_sealEngine->generateSeal(m_miningInfo);
	}
	if (!m_wouldMine)
		m_sealEngine->cancelGeneration();
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
			else if (m_specialFilters.count(w.second.id))
				for (h256 const& hash: m_specialFilters.at(w.second.id))
				{
					cwatch << "!!!" << w.first << LogTag::Special << (w.second.id == PendingChangedFilter ? "pending" : w.second.id == ChainChangedFilter ? "chain" : "???");
					w.second.changes.push_back(LocalisedLogEntry(SpecialLogEntry, hash));
				}
		}
	// clear the filters now.
	for (auto& i: m_filters)
		i.second.changes.clear();
	for (auto& i: m_specialFilters)
		i.second.clear();
}

void Client::doWork()
{
	bool t = true;
	if (m_syncBlockQueue.compare_exchange_strong(t, false))
		syncBlockQueue();

	if (m_needStateReset)
	{
		resetState();
		m_needStateReset = false;
	}

	t = true;
	if (m_syncTransactionQueue.compare_exchange_strong(t, false) && !m_remoteWorking && !isSyncing())
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
		m_bq.tick();
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
					clog(ClientTrace) << "GC: Uninstall" << key << "(" << chrono::duration_cast<chrono::seconds>(chrono::system_clock::now() - m_watches[key].lastPoll).count() << "s old)";
				}
		for (auto i: toUninstall)
			uninstallWatch(i);

		// blockchain GC
		bc().garbageCollect();

		m_lastGarbageCollection = chrono::system_clock::now();
	}
}

State Client::asOf(h256 const& _block) const
{
	try
	{
		State ret(m_stateDB);
		ret.populateFromChain(bc(), _block);
		return ret;
	}
	catch (Exception& ex)
	{
		ex << errinfo_block(bc().block(_block));
		onBadBlock(ex);
		return State();
	}
}

void Client::prepareForTransaction()
{
	startWorking();
}

State Client::state(unsigned _txi, h256 _block) const
{
	try
	{
		State ret(m_stateDB);
		ret.populateFromChain(bc(), _block);
		return ret.fromPending(_txi);
	}
	catch (Exception& ex)
	{
		ex << errinfo_block(bc().block(_block));
		onBadBlock(ex);
		return State();
	}
}

State Client::state(h256 const& _block, PopulationStatistics* o_stats) const
{
	try
	{
		State ret(m_stateDB);
		PopulationStatistics s = ret.populateFromChain(bc(), _block);
		if (o_stats)
			swap(s, *o_stats);
		return ret;
	}
	catch (Exception& ex)
	{
		ex << errinfo_block(bc().block(_block));
		onBadBlock(ex);
		return State();
	}
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

SyncStatus Client::syncStatus() const
{
	auto h = m_host.lock();
	return h ? h->status() : SyncStatus();
}

bool Client::submitSealed(bytes const& _header)
{
	DEV_WRITE_GUARDED(x_working)
		if (!m_working.sealBlock(_header))
			return false;

	bytes newBlock;
	DEV_READ_GUARDED(x_working)
	{
		DEV_WRITE_GUARDED(x_postMine)
			m_postMine = m_working;
		newBlock = m_working.blockData();
	}

	// OPTIMISE: very inefficient to not utilise the existing OverlayDB in m_postMine that contains all trie changes.
	return m_bq.import(&newBlock, true) == ImportResult::Success;
}














std::tuple<h256, h256, h256> EthashClient::getEthashWork()
{
	// lock the work so a later submission isn't invalidated by processing a transaction elsewhere.
	// this will be reset as soon as a new block arrives, allowing more transactions to be processed.
	bool oldShould = shouldServeWork();
	m_lastGetWork = chrono::system_clock::now();

	if (!m_mineOnBadChain && isChainBad())
		return std::tuple<h256, h256, h256>();

	// if this request has made us bother to serve work, prep it now.
	if (!oldShould && shouldServeWork())
		onPostStateChanged();
	else
		// otherwise, set this to true so that it gets prepped next time.
		m_remoteWorking = true;
	Ethash::BlockHeader bh = Ethash::BlockHeader(m_miningInfo);
	Ethash::manuallySetWork(m_sealEngine.get(), bh);
	return std::tuple<h256, h256, h256>(bh.hashWithout(), bh.seedHash(), bh.boundary());
}

bool EthashClient::submitEthashWork(h256 const& _mixHash, h64 const& _nonce)
{
	Ethash::manuallySubmitWork(m_sealEngine.get(), _mixHash, _nonce);
	return true;
}

