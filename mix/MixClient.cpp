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
/** @file MixClient.cpp
 * @author Yann yann@ethdev.com
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <vector>
#include <libdevcore/Exceptions.h>
#include <libethereum/BlockChain.h>
#include <libethereum/Transaction.h>
#include <libethereum/Executive.h>
#include <libethereum/ExtVM.h>
#include <libevm/VM.h>

#include "Exceptions.h"
#include "MixClient.h"

using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

const Secret c_stdSecret = Secret("cb73d9408c4720e230387d956eb0f829d8a4dd2c1055f96257167e14e7169074");

MixClient::MixClient():
	m_userAccount(c_stdSecret)
{
	resetState(10000000 * ether);
}

void MixClient::resetState(u256 _balance)
{
	WriteGuard l(x_state);
	Guard fl(m_filterLock);
	m_filters.clear();
	m_watches.clear();
	m_state = eth::State(m_userAccount.address(), m_stateDB, BaseState::Empty);
	m_state.addBalance(m_userAccount.address(), _balance);
	Block genesis;
	genesis.state = m_state;
	Block open;
	m_blocks = Blocks { genesis, open }; //last block contains a list of pending transactions to be finalized
}

void MixClient::executeTransaction(Transaction const& _t, State& _state)
{
	bytes rlp = _t.rlp();
	Executive execution(_state, LastHashes(), 0);
	execution.setup(&rlp);
	bytes code;
	bytesConstRef data;
	bool firstIteration = true;
	std::vector<MachineState> machineStates;
	std::vector<MachineState const*> levels;
	auto onOp = [&](uint64_t steps, Instruction inst, dev::bigint newMemSize, dev::bigint gasCost, void* voidVM, void const* voidExt)
	{
		VM& vm = *(VM*)voidVM;
		ExtVM const& ext = *(ExtVM const*)voidExt;

		if (firstIteration)
		{
			code = ext.code;
			data = ext.data;
			firstIteration = false;
		}

		if (levels.size() < ext.depth)
			levels.push_back(&machineStates.back());
		else
			levels.resize(ext.depth);

		machineStates.push_back(MachineState({steps, ext.myAddress, vm.curPC(), inst, newMemSize, vm.gas(),
									  vm.stack(), vm.memory(), gasCost, ext.state().storage(ext.myAddress), levels}));
	};

	execution.go(onOp);
	execution.finalize();

	ExecutionResult d;
	d.returnValue = execution.out().toVector();
	d.machineStates = machineStates;
	d.executionCode = code;
	d.executionData = data;
	d.transactionData = _t.data();
	d.address = _t.receiveAddress();
	d.sender = _t.sender();
	d.value = _t.value();
	if (_t.isCreation())
		d.contractAddress = right160(sha3(rlpList(_t.sender(), _t.nonce())));
	d.receipt = TransactionReceipt(m_state.rootHash(), execution.gasUsed(), execution.logs()); //TODO: track gas usage
	m_blocks.back().transactions.emplace_back(d);

	h256Set changed;
	Guard l(m_filterLock);
	for (std::pair<h256 const, eth::InstalledFilter>& i: m_filters)
	{
		if ((unsigned)i.second.filter.latest() > m_blocks.size() - 1)
		{
			// acceptable number.
			auto m = i.second.filter.matches(d.receipt);
			if (m.size())
			{
				// filter catches them
				for (LogEntry const& l: m)
					i.second.changes.push_back(LocalisedLogEntry(l, m_blocks.size()));
				changed.insert(i.first);
			}
		}
	}
	changed.insert(dev::eth::PendingChangedFilter);
	noteChanged(changed);
}

void MixClient::validateBlock(int _block) const
{
	if (_block != -1 && _block != 0 && (unsigned)_block >= m_blocks.size() - 1)
		BOOST_THROW_EXCEPTION(InvalidBlockException() << BlockIndex(_block));
}

void MixClient::mine()
{
	WriteGuard l(x_state);
	Block& block = m_blocks.back();
	m_state.completeMine();
	block.state = m_state;
	block.info = m_state.info();
	block.hash = block.info.hash;
	m_blocks.push_back(Block());
	h256Set changed { dev::eth::PendingChangedFilter, dev::eth::ChainChangedFilter };
	noteChanged(changed);
}

State const& MixClient::asOf(int _block) const
{
	validateBlock(_block);
	if (_block == 0)
		return m_blocks[m_blocks.size() - 2].state;
	else if (_block == -1)
		return m_state;
	else
		return m_blocks[_block].state;
}


void MixClient::transact(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	WriteGuard l(x_state);
	u256 n = m_state.transactionsFrom(toAddress(_secret));
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
	executeTransaction(t, m_state);
}

Address MixClient::transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	WriteGuard l(x_state);
	u256 n = m_state.transactionsFrom(toAddress(_secret));
	eth::Transaction t(_endowment, _gasPrice, _gas, _init, n, _secret);
	executeTransaction(t, m_state);
	Address address = right160(sha3(rlpList(t.sender(), t.nonce())));
	return address;
}

void MixClient::inject(bytesConstRef _rlp)
{
	WriteGuard l(x_state);
	eth::Transaction t(_rlp, CheckSignature::None);
	executeTransaction(t, m_state);
}

void MixClient::flushTransactions()
{
}

bytes MixClient::call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	u256 n;
	State temp;
	{
		ReadGuard lr(x_state);
		temp = m_state;
		n = temp.transactionsFrom(toAddress(_secret));
	}
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
	bytes rlp = t.rlp();
	WriteGuard lw(x_state); //TODO: lock is required only for last execution state
	executeTransaction(t, temp);
	return m_blocks.back().transactions.back().returnValue;
}

u256 MixClient::balanceAt(Address _a, int _block) const
{
	ReadGuard l(x_state);
	return asOf(_block).balance(_a);
}

u256 MixClient::countAt(Address _a, int _block) const
{
	ReadGuard l(x_state);
	return asOf(_block).transactionsFrom(_a);
}

u256 MixClient::stateAt(Address _a, u256 _l, int _block) const
{
	ReadGuard l(x_state);
	return asOf(_block).storage(_a, _l);
}

bytes MixClient::codeAt(Address _a, int _block) const
{
	ReadGuard l(x_state);
	return asOf(_block).code(_a);
}

std::map<u256, u256> MixClient::storageAt(Address _a, int _block) const
{
	ReadGuard l(x_state);
	return asOf(_block).storage(_a);
}

eth::LocalisedLogEntries MixClient::logs(unsigned _watchId) const
{
	Guard l(m_filterLock);
	h256 h = m_watches.at(_watchId).id;
	auto filterIter = m_filters.find(h);
	if (filterIter != m_filters.end())
		return logs(filterIter->second.filter);
	return eth::LocalisedLogEntries();
}

eth::LocalisedLogEntries MixClient::logs(eth::LogFilter const& _f) const
{
	LocalisedLogEntries ret;
	unsigned lastBlock = m_blocks.size() - 1; //last block contains pending transactions
	unsigned block = std::min<unsigned>(lastBlock, (unsigned)_f.latest());
	unsigned end = std::min(lastBlock, std::min(block, (unsigned)_f.earliest()));
	for (; ret.size() != _f.max() && block != end; block--)
	{
		bool pendingBlock = (block == lastBlock);
		if (pendingBlock || _f.matches(m_blocks[block].info.logBloom))
			for (ExecutionResult const& t: m_blocks[block].transactions)
				if (pendingBlock || _f.matches(t.receipt.bloom()))
				{
					LogEntries logEntries = _f.matches(t.receipt);
					if (logEntries.size())
					{
						for (unsigned entry = _f.skip(); entry < logEntries.size() && ret.size() != _f.max(); ++entry)
							ret.insert(ret.begin(), LocalisedLogEntry(logEntries[entry], block));
					}
				}
	}
	return ret;
}

unsigned MixClient::installWatch(h256 _h)
{
	unsigned ret;
	{
		Guard l(m_filterLock);
		ret = m_watches.size() ? m_watches.rbegin()->first + 1 : 0;
		m_watches[ret] = ClientWatch(_h);
	}
	auto ch = logs(ret);
	if (ch.empty())
		ch.push_back(eth::InitialChange);
	{
		Guard l(m_filterLock);
		swap(m_watches[ret].changes, ch);
	}
	return ret;
}

unsigned MixClient::installWatch(eth::LogFilter const& _f)
{
	h256 h = _f.sha3();
	{
		Guard l(m_filterLock);
		if (!m_filters.count(h))
			m_filters.insert(std::make_pair(h, _f));
	}
	return installWatch(h);
}

void MixClient::uninstallWatch(unsigned _i)
{
	Guard l(m_filterLock);

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

void MixClient::noteChanged(h256Set const& _filters)
{
	for (auto& i: m_watches)
		if (_filters.count(i.second.id))
		{
			if (m_filters.count(i.second.id))
				i.second.changes += m_filters.at(i.second.id).changes;
			else
				i.second.changes.push_back(LocalisedLogEntry(SpecialLogEntry, 0));
		}
	for (auto& i: m_filters)
		i.second.changes.clear();
}

LocalisedLogEntries MixClient::peekWatch(unsigned _watchId) const
{
	Guard l(m_filterLock);
	if (_watchId < m_watches.size())
		return m_watches.at(_watchId).changes;
	return LocalisedLogEntries();
}

LocalisedLogEntries MixClient::checkWatch(unsigned _watchId)
{
	Guard l(m_filterLock);
	LocalisedLogEntries ret;
	if (_watchId < m_watches.size())
		std::swap(ret, m_watches.at(_watchId).changes);
	return ret;
}

h256 MixClient::hashFromNumber(unsigned _number) const
{
	validateBlock(_number);
	return m_blocks[_number].hash;
}

eth::BlockInfo MixClient::blockInfo(h256 _hash) const
{
	(void)_hash;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::blockInfo"));
}

eth::BlockDetails MixClient::blockDetails(h256 _hash) const
{
	(void)_hash;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::blockDetails"));
}

eth::Transaction MixClient::transaction(h256 _blockHash, unsigned _i) const
{
	(void)_blockHash;
	(void)_i;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::transaction"));
}

eth::BlockInfo MixClient::uncle(h256 _blockHash, unsigned _i) const
{
	(void)_blockHash;
	(void)_i;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::uncle"));
}

unsigned MixClient::number() const
{
	return m_blocks.size() - 1;
}

eth::Transactions MixClient::pending() const
{
	return eth::Transactions();
}

eth::StateDiff MixClient::diff(unsigned _txi, h256 _block) const
{
	(void)_txi;
	(void)_block;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::diff"));
}

eth::StateDiff MixClient::diff(unsigned _txi, int _block) const
{
	(void)_txi;
	(void)_block;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::diff"));
}

Addresses MixClient::addresses(int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	Addresses ret;
	for (auto const& i: m_state.addresses())
		ret.push_back(i.first);
	return ret;
}

u256 MixClient::gasLimitRemaining() const
{
	ReadGuard l(x_state);
	return m_state.gasLimitRemaining();
}

void MixClient::setAddress(Address _us)
{
	WriteGuard l(x_state);
	m_state.setAddress(_us);
}

Address MixClient::address() const
{
	ReadGuard l(x_state);
	return m_state.address();
}

void MixClient::setMiningThreads(unsigned _threads)
{
	(void)_threads;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::setMiningThreads"));
}

unsigned MixClient::miningThreads() const
{
	return 0;
}

void MixClient::startMining()
{
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::startMining"));
}

void MixClient::stopMining()
{
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::stopMining"));
}

bool MixClient::isMining()
{
	return false;
}

eth::MineProgress MixClient::miningProgress() const
{
	return eth::MineProgress();
}
