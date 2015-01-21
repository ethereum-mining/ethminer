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
/** @file MixClient.h
 * @author Yann yann@ethdev.com
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <libethereum/Interface.h>

namespace dev
{
namespace mix
{

/**
 * @brief Store information about a machine state.
 */
struct MachineState
{
	uint64_t steps;
	dev::Address cur;
	dev::u256 curPC;
	dev::eth::Instruction inst;
	dev::bigint newMemSize;
	dev::u256 gas;
	dev::u256s stack;
	dev::bytes memory;
	dev::bigint gasCost;
	std::map<dev::u256, dev::u256> storage;
	std::vector<MachineState const*> levels;
};

/**
 * @brief Store information about a machine states.
 */
struct ExecutionResult
{
	std::vector<MachineState> machineStates;
	bytes executionCode;
	bytesConstRef executionData;
	Address contractAddress;
	bool contentAvailable;
	std::string message;
	bytes returnValue;
};

class MixClient: public dev::eth::Interface
{
public:
	MixClient();
	/// Reset state to the empty state with given balance.
	void resetState(u256 _balance);
	const KeyPair& userAccount() const { return m_userAccount; }
	const ExecutionResult lastExecutionResult() const { ReadGuard l(x_state); return m_lastExecutionResult; }
	const Address lastContractAddress() const { ReadGuard l(x_state); return m_lastExecutionResult.contractAddress; }

	//dev::eth::Interface
	void transact(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice) override;
	Address transact(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice) override;
	void inject(bytesConstRef _rlp) override;
	void flushTransactions() override;
	bytes call(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice) override;
	u256 balanceAt(Address _a, int _block) const override;
	u256 countAt(Address _a, int _block) const override;
	u256 stateAt(Address _a, u256 _l, int _block) const override;
	bytes codeAt(Address _a, int _block) const override;
	std::map<u256, u256> storageAt(Address _a, int _block) const override;
	eth::LocalisedLogEntries logs(unsigned _watchId) const override;
	eth::LocalisedLogEntries logs(eth::LogFilter const& _filter) const override;
	unsigned installWatch(eth::LogFilter const& _filter) override;
	unsigned installWatch(h256 _filterId) override;
	void uninstallWatch(unsigned _watchId) override;
	eth::LocalisedLogEntries peekWatch(unsigned _watchId) const override;
	eth::LocalisedLogEntries checkWatch(unsigned _watchId) override;
	h256 hashFromNumber(unsigned _number) const override;
	eth::BlockInfo blockInfo(h256 _hash) const override;
	eth::BlockDetails blockDetails(h256 _hash) const override;
	eth::Transaction transaction(h256 _blockHash, unsigned _i) const override;
	eth::BlockInfo uncle(h256 _blockHash, unsigned _i) const override;
	unsigned number() const override;
	eth::Transactions pending() const override;
	eth::StateDiff diff(unsigned _txi, h256 _block) const override;
	eth::StateDiff diff(unsigned _txi, int _block) const override;
	Addresses addresses(int _block) const override;
	u256 gasLimitRemaining() const override;
	void setAddress(Address _us) override;
	Address address() const override;
	void setMiningThreads(unsigned _threads) override;
	unsigned miningThreads() const override;
	void startMining() override;
	void stopMining() override;
	bool isMining() override;
	eth::MineProgress miningProgress() const override;

private:
	void executeTransaction(bytesConstRef _rlp, eth::State& _state);
	void validateBlock(int _block) const;

	KeyPair m_userAccount;
	eth::State m_state;
	OverlayDB m_stateDB;
	mutable boost::shared_mutex x_state;
	ExecutionResult m_lastExecutionResult;
};

}

}
