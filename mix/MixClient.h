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

#include <vector>
#include <libethereum/Interface.h>
#include <libethereum/Client.h>

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
	dev::Address address;
	dev::u256 curPC;
	dev::eth::Instruction inst;
	dev::bigint newMemSize;
	dev::u256 gas;
	dev::u256s stack;
	dev::bytes memory;
	dev::bigint gasCost;
	std::map<dev::u256, dev::u256> storage;
	std::vector<unsigned> levels;
	unsigned codeIndex;
	unsigned dataIndex;
};

/**
 * @brief Store information about a machine states.
 */
struct ExecutionResult
{
	ExecutionResult(): receipt(dev::h256(), dev::h256(), dev::eth::LogEntries()) {}

	std::vector<MachineState> machineStates;
	std::vector<bytes> transactionData;
	std::vector<bytes> executionCode;
	bytes returnValue;
	dev::Address address;
	dev::Address sender;
	dev::Address contractAddress;
	dev::u256 value;
	dev::eth::TransactionReceipt receipt;
};

using ExecutionResults = std::vector<ExecutionResult>;

struct Block
{
	ExecutionResults transactions;
	h256 hash;
	dev::eth::State state;
	dev::eth::BlockInfo info;
};

using Blocks = std::vector<Block>;


class MixClient: public dev::eth::Interface
{
public:
	MixClient();
	/// Reset state to the empty state with given balance.
	void resetState(u256 _balance);
	KeyPair const& userAccount() const { return m_userAccount; }
	void mine();
	Blocks const& record() const { return m_blocks; }

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
	void executeTransaction(dev::eth::Transaction const& _t, eth::State& _state);
	void validateBlock(int _block) const;
	void noteChanged(h256Set const& _filters);
	dev::eth::State const& asOf(int _block) const;

	KeyPair m_userAccount;
	eth::State m_state;
	OverlayDB m_stateDB;
	mutable boost::shared_mutex x_state;
	mutable std::mutex m_filterLock;
	std::map<h256, dev::eth::InstalledFilter> m_filters;
	std::map<unsigned, dev::eth::ClientWatch> m_watches;
	Blocks m_blocks;
};

}

}
