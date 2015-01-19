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

#include "MixClient.h"

using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

MixClient::MixClient():
	m_userAccount(KeyPair::create())
{
	resetState(10000000 * ether);
}

void MixClient::resetState(u256 _balance)
{
	WriteGuard l(x_state);
	m_state = eth::State(Address(), m_stateDB, BaseState::Empty);
	m_state.addBalance(m_userAccount.address(), _balance);
}

void MixClient::executeTransaction(bytesConstRef _rlp, State& _state)
{
	Executive execution(_state, LastHashes(), 0);
	execution.setup(_rlp);
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
	d.contentAvailable = true;
	d.message = "ok";
	m_lastExecutionResult = d;
}

void MixClient::validateBlock(int _block) const
{
	//TODO: throw exception here if _block != 0
	(void)_block;
}

void MixClient::transact(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	WriteGuard l(x_state);
	u256 n = m_state.transactionsFrom(toAddress(_secret));
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
	bytes rlp = t.rlp();
	executeTransaction(&rlp, m_state);
}

Address MixClient::transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice)
{
	WriteGuard l(x_state);
	u256 n = m_state.transactionsFrom(toAddress(_secret));
	eth::Transaction t(_endowment, _gasPrice, _gas, _init, n, _secret);
	bytes rlp = t.rlp();
	executeTransaction(&rlp, m_state);
	return right160(sha3(rlpList(t.sender(), t.nonce())));
}

void MixClient::inject(bytesConstRef _rlp)
{
	WriteGuard l(x_state);
	executeTransaction(_rlp, m_state);
}

void MixClient::flushTransactions()
{
}

bytes MixClient::call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice)
{
	bytes out;
	u256 n;
	State temp;
	{
		ReadGuard lr(x_state);
		temp = m_state;
		n = temp.transactionsFrom(toAddress(_secret));
	}
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n, _secret);
	bytes rlp = t.rlp();
	WriteGuard lw(x_state); //TODO: lock is required only for last executoin state
	executeTransaction(&rlp, temp);
	return m_lastExecutionResult.returnValue;
}

u256 MixClient::balanceAt(Address _a, int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	return m_state.balance(_a);
}

u256 MixClient::countAt(Address _a, int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	return m_state.transactionsFrom(_a);
}

u256 MixClient::stateAt(Address _a, u256 _l, int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	return m_state.storage(_a, _l);
}

bytes MixClient::codeAt(Address _a, int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	return m_state.code(_a);
}

std::map<u256, u256> MixClient::storageAt(Address _a, int _block) const
{
	validateBlock(_block);
	ReadGuard l(x_state);
	return m_state.storage(_a);
}

eth::LocalisedLogEntries MixClient::logs(unsigned _watchId) const
{
	(void)_watchId;
	return LocalisedLogEntries();
}

eth::LocalisedLogEntries MixClient::logs(eth::LogFilter const& _filter) const
{
	(void)_filter;
	return LocalisedLogEntries();
}

unsigned MixClient::installWatch(eth::LogFilter const& _filter)
{
	(void)_filter;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::installWatch"));
}

unsigned MixClient::installWatch(h256 _filterId)
{
	(void)_filterId;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::installWatch"));
}

void MixClient::uninstallWatch(unsigned _watchId)
{
	(void)_watchId;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::uninstallWatch"));
}

eth::LocalisedLogEntries MixClient::peekWatch(unsigned _watchId) const
{
	(void)_watchId;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::peekWatch"));
}

eth::LocalisedLogEntries MixClient::checkWatch(unsigned _watchId)
{
	(void)_watchId;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::checkWatch"));
}

h256 MixClient::hashFromNumber(unsigned _number) const
{
	(void)_number;
	BOOST_THROW_EXCEPTION(InterfaceNotSupported("dev::eth::Interface::hashFromNumber"));
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
	return 0;
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
