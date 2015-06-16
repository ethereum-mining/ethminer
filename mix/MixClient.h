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
#include <string>
#include <libethereum/ExtVM.h>
#include <libethereum/ClientBase.h>
#include <libethereum/Client.h>
#include "MachineStates.h"

namespace dev
{
namespace mix
{

class MixBlockChain: public dev::eth::BlockChain
{
public:
	MixBlockChain(std::string const& _path, h256 _stateRoot): BlockChain(createGenesisBlock(_stateRoot), _path, WithExisting::Kill) {}

	static bytes createGenesisBlock(h256 _stateRoot);
};

class MixClient: public dev::eth::ClientBase
{
public:
	MixClient(std::string const& _dbPath);
	virtual ~MixClient();
	/// Reset state to the empty state with given balance.
	void resetState(std::unordered_map<dev::Address, dev::eth::Account> const& _accounts,  Secret const& _miner = Secret());
	void mine();
	ExecutionResult lastExecution() const;
	ExecutionResult execution(unsigned _index) const;

	void submitTransaction(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice) override;
	Address submitTransaction(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice) override;
	dev::eth::ExecutionResult call(Address const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, eth::BlockNumber _blockNumber = eth::PendingBlock, eth::FudgeFactor _ff = eth::FudgeFactor::Strict) override;
	dev::eth::ExecutionResult create(Address const& _secret, u256 _value, bytes const& _data = bytes(), u256 _gas = 10000, u256 _gasPrice = 10 * eth::szabo, eth::BlockNumber _blockNumber = eth::PendingBlock, eth::FudgeFactor _ff = eth::FudgeFactor::Strict) override;

	void submitTransaction(Secret _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, bool _gasAuto);
	Address submitTransaction(Secret _secret, u256 _endowment, bytes const& _init, u256 _gas, u256 _gasPrice, bool _gasAuto);
	dev::eth::ExecutionResult call(Address const& _secret, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, eth::BlockNumber _blockNumber, bool _gasAuto, eth::FudgeFactor _ff = eth::FudgeFactor::Strict);

	void setAddress(Address _us) override;
	void startMining() override;
	void stopMining() override;
	bool isMining() const override;
	uint64_t hashrate() const override;
	eth::MiningProgress miningProgress() const override;
	eth::ProofOfWork::WorkPackage getWork() override { return eth::ProofOfWork::WorkPackage(); }
	bool submitWork(eth::ProofOfWork::Solution const&) override { return false; }
	virtual void flushTransactions() override {}

	/// @returns the last mined block information
	using Interface::blockInfo; // to remove warning about hiding virtual function
	eth::BlockInfo blockInfo() const;

protected:
	/// ClientBase methods
	using ClientBase::asOf;
	virtual dev::eth::State asOf(h256 const& _block) const override;
	virtual dev::eth::BlockChain& bc() override { return *m_bc; }
	virtual dev::eth::BlockChain const& bc() const override { return *m_bc; }
	virtual dev::eth::State preMine() const override { ReadGuard l(x_state);  return m_startState; }
	virtual dev::eth::State postMine() const override { ReadGuard l(x_state); return m_state; }
	virtual void prepareForTransaction() override {}

private:
	void executeTransaction(dev::eth::Transaction const& _t, eth::State& _state, bool _call, bool _gasAuto, dev::Secret const& _secret = dev::Secret());
	dev::eth::Transaction replaceGas(dev::eth::Transaction const& _t, dev::u256 const& _gas, dev::Secret const& _secret = dev::Secret());

	eth::State m_state;
	eth::State m_startState;
	OverlayDB m_stateDB;
	std::unique_ptr<MixBlockChain> m_bc;
	mutable boost::shared_mutex x_state;
	mutable boost::shared_mutex x_executions;
	ExecutionResults m_executions;
	std::string m_dbPath;
};

}

}
