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
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include "MixClient.h"
#include <vector>
#include <utility>
#include <libdevcore/Exceptions.h>
#include <libethcore/Params.h>
#include <libethcore/BasicAuthority.h>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/Transaction.h>
#include <libethereum/Executive.h>
#include <libethereum/ExtVM.h>
#include <libethereum/BlockChain.h>
#include <libevm/VM.h>
#include "Exceptions.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace mix
{

u256 const c_mixGenesisDifficulty = 131072; //TODO: make it lower for Mix somehow

namespace
{
}

MixBlockChain::MixBlockChain(std::string const& _path, h256 _stateRoot):
	FullBlockChain<NoProof>(createGenesisBlock(_stateRoot), std::unordered_map<Address, Account>(), _path, WithExisting::Kill)
{
}

bytes MixBlockChain::createGenesisBlock(h256 _stateRoot)
{
	RLPStream block(3);
	block.appendList(13)
			<< h256() << EmptyListSHA3 << h160() << _stateRoot << EmptyTrie << EmptyTrie
			<< LogBloom() << c_mixGenesisDifficulty << 0 << 3141592 << 0 << (unsigned)0
			<< std::string();
	block.appendRaw(RLPEmptyList);
	block.appendRaw(RLPEmptyList);
	return block.out();
}

MixClient::MixClient(std::string const& _dbPath):
	m_dbPath(_dbPath)
{
	resetState(std::unordered_map<Address, Account>());
}

MixClient::~MixClient()
{
}

void MixClient::resetState(std::unordered_map<Address, Account> const& _accounts,  Secret const& _miner)
{
	WriteGuard l(x_state);
	Guard fl(x_filtersWatches);

	m_filters.clear();
	for (auto& i: m_specialFilters)
		i.second.clear();
	m_watches.clear();

	m_stateDB = OverlayDB();
	SecureTrieDB<Address, MemoryDB> accountState(&m_stateDB);
	accountState.init();

	dev::eth::commit(_accounts, accountState);
	h256 stateRoot = accountState.root();
	m_bc.reset();
	m_bc.reset(new MixBlockChain(m_dbPath, stateRoot));
	Block b(m_stateDB, BaseState::PreExisting, KeyPair(_miner).address());
	b.sync(bc());
	m_preMine = b;
	m_postMine = b;
	WriteGuard lx(x_executions);
	m_executions.clear();
}

Transaction MixClient::replaceGas(Transaction const& _t, u256 const& _gas, Secret const& _secret)
{
	Transaction ret;
	if (_secret)
	{
		if (_t.isCreation())
			ret = Transaction(_t.value(), _t.gasPrice(), _gas, _t.data(), _t.nonce(), _secret);
		else
			ret = Transaction(_t.value(), _t.gasPrice(), _gas, _t.receiveAddress(), _t.data(), _t.nonce(), _secret);
	}
	else
	{
		if (_t.isCreation())
			ret = Transaction(_t.value(), _t.gasPrice(), _gas, _t.data(), _t.nonce());
		else
			ret = Transaction(_t.value(), _t.gasPrice(), _gas, _t.receiveAddress(), _t.data(), _t.nonce());
		ret.forceSender(_t.safeSender());
	}
	return ret;
}

// TODO: prototype changed - will need rejigging.
ExecutionResult MixClient::debugTransaction(Transaction const& _t, State const& _state, EnvInfo const& _envInfo, bool _call)
{
	State execState = _state;
	execState.addBalance(_t.sender(), _t.gas() * _t.gasPrice()); //give it enough balance for gas estimation
	eth::ExecutionResult er;
	Executive execution(execState, _envInfo);
	execution.setResultRecipient(er);
	execution.initialize(_t);
	execution.execute();

	std::vector<MachineState> machineStates;
	std::vector<unsigned> levels;
	std::vector<MachineCode> codes;
	std::map<bytes const*, unsigned> codeIndexes;
	std::vector<bytes> data;
	std::map<bytesConstRef const*, unsigned> dataIndexes;
	bytes const* lastCode = nullptr;
	bytesConstRef const* lastData = nullptr;
	unsigned codeIndex = 0;
	unsigned dataIndex = 0;
	auto onOp = [&](uint64_t steps, Instruction inst, bigint newMemSize, bigint gasCost, bigint gas, void* voidVM, void const* voidExt)
	{
		VM& vm = *static_cast<VM*>(voidVM);
		ExtVM const& ext = *static_cast<ExtVM const*>(voidExt);
		if (lastCode == nullptr || lastCode != &ext.code)
		{
			auto const& iter = codeIndexes.find(&ext.code);
			if (iter != codeIndexes.end())
				codeIndex = iter->second;
			else
			{
				codeIndex = codes.size();
				codes.push_back(MachineCode({ext.myAddress, ext.code}));
				codeIndexes[&ext.code] = codeIndex;
			}
			lastCode = &ext.code;
		}

		if (lastData == nullptr || lastData != &ext.data)
		{
			auto const& iter = dataIndexes.find(&ext.data);
			if (iter != dataIndexes.end())
				dataIndex = iter->second;
			else
			{
				dataIndex = data.size();
				data.push_back(ext.data.toBytes());
				dataIndexes[&ext.data] = dataIndex;
			}
			lastData = &ext.data;
		}

		if (levels.size() < ext.depth)
			levels.push_back(machineStates.size() - 1);
		else
			levels.resize(ext.depth);

		machineStates.push_back(MachineState{
									steps,
									vm.curPC(),
									inst,
									newMemSize,
									static_cast<u256>(gas),
									vm.stack(),
									vm.memory(),
									gasCost,
									ext.state().storage(ext.myAddress),
									std::move(levels),
									codeIndex,
									dataIndex
								});
	};

	execution.go(onOp);
	execution.finalize();

	switch (er.excepted)
	{
	case TransactionException::None:
		break;
	case TransactionException::NotEnoughCash:
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Insufficient balance for contract deployment"));
	case TransactionException::OutOfGasIntrinsic:
	case TransactionException::OutOfGasBase:
	case TransactionException::OutOfGas:
		BOOST_THROW_EXCEPTION(OutOfGas() << errinfo_comment("Not enough gas"));
	case TransactionException::BlockGasLimitReached:
		BOOST_THROW_EXCEPTION(OutOfGas() << errinfo_comment("Block gas limit reached"));
	case TransactionException::BadJumpDestination:
		BOOST_THROW_EXCEPTION(OutOfGas() << errinfo_comment("Solidity exception (bad jump)"));
	case TransactionException::OutOfStack:
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Out of stack"));
	case TransactionException::StackUnderflow:
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Stack underflow"));
		//these should not happen in mix
	case TransactionException::Unknown:
	case TransactionException::BadInstruction:
	case TransactionException::InvalidSignature:
	case TransactionException::InvalidNonce:
	case TransactionException::InvalidFormat:
	case TransactionException::BadRLP:
		BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("Internal execution error"));
	}

	ExecutionResult d;
	d.inputParameters = _t.data();
	d.result = er;
	d.machineStates = machineStates;
	d.executionCode = std::move(codes);
	d.transactionData = std::move(data);
	d.address = _t.receiveAddress();
	d.sender = _t.sender();
	d.value = _t.value();
	d.gasUsed = er.gasUsed + er.gasRefunded + c_callStipend;
	if (_t.isCreation())
		d.contractAddress = right160(sha3(rlpList(_t.sender(), _t.nonce())));
	if (!_call)
		d.transactionIndex = m_postMine.pending().size();
	d.executonIndex = m_executions.size();
	return d;
}


void MixClient::executeTransaction(Transaction const& _t, Block& _block, bool _call, bool _gasAuto, Secret const& _secret)
{
	Transaction t = _gasAuto ? replaceGas(_t, m_postMine.gasLimitRemaining()) : _t;

	// do debugging run first
	EnvInfo envInfo(bc().info(), bc().lastHashes());
	ExecutionResult d = debugTransaction(t, _block.state(), envInfo, _call);

	// execute on a state
	if (!_call)
	{
		t = _gasAuto ? replaceGas(_t, d.gasUsed, _secret) : _t;
		eth::ExecutionResult const& er = _block.execute(envInfo.lastHashes(), t);
		if (t.isCreation() && _block.state().code(d.contractAddress).empty())
			BOOST_THROW_EXCEPTION(OutOfGas() << errinfo_comment("Not enough gas for contract deployment"));
		d.gasUsed = er.gasUsed + er.gasRefunded + er.gasForDeposit + c_callStipend;
		LocalisedLogEntries logs;
		TransactionReceipt const& tr = _block.receipt(_block.pending().size() - 1);

		LogEntries le = tr.log();
		if (le.size())
			for (unsigned j = 0; j < le.size(); ++j)
				logs.insert(logs.begin(), LocalisedLogEntry(le[j]));
		d.logs =  logs;
	}
	WriteGuard l(x_executions);
	m_executions.emplace_back(std::move(d));
}

void MixClient::mine()
{
	WriteGuard l(x_state);
	m_postMine.commitToSeal(bc());

	NoProof::BlockHeader h(m_postMine.info());
	RLPStream header;
	h.streamRLP(header);
	m_postMine.sealBlock(header.out());
	bc().import(m_postMine.blockData(), m_stateDB, ImportRequirements::Everything & ~ImportRequirements::ValidSeal);
	m_postMine.sync(bc());
	m_preMine = m_postMine;
}

ExecutionResult MixClient::lastExecution() const
{
	ReadGuard l(x_executions);
	return m_executions.empty() ? ExecutionResult() : m_executions.back();
}

ExecutionResult MixClient::execution(unsigned _index) const
{
	ReadGuard l(x_executions);
	return m_executions.at(_index);
}

Block MixClient::asOf(h256 const& _block) const
{
	ReadGuard l(x_state);
	Block ret(m_stateDB);
	ret.populateFromChain(bc(), _block);
	return ret;
}

pair<h256, Address> MixClient::submitTransaction(eth::TransactionSkeleton const& _ts, Secret const& _secret, bool _gasAuto)
{
	WriteGuard l(x_state);
	TransactionSkeleton ts = _ts;
	ts.from = toAddress(_secret);
	ts.nonce = m_postMine.transactionsFrom(ts.from);
	eth::Transaction t(ts, _secret);
	executeTransaction(t, m_postMine, false, _gasAuto, _secret);
	return make_pair(t.sha3(), toAddress(ts.from, ts.nonce));
}

dev::eth::ExecutionResult MixClient::call(Address const& _from, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, bool _gasAuto, FudgeFactor _ff)
{
	(void)_blockNumber;
	Block block = asOf(eth::PendingBlock);
	u256 n = block.transactionsFrom(_from);
	Transaction t(_value, _gasPrice, _gas, _dest, _data, n);
	t.forceSender(_from);
	if (_ff == FudgeFactor::Lenient)
		block.mutableState().addBalance(_from, (u256)(t.gasRequired() * t.gasPrice() + t.value()));
	WriteGuard lw(x_state); //TODO: lock is required only for last execution state
	executeTransaction(t, block, true, _gasAuto);
	return lastExecution().result;
}

dev::eth::ExecutionResult MixClient::call(Address const& _from, u256 _value, Address _dest, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, eth::FudgeFactor _ff)
{
	return call(_from, _value, _dest, _data, _gas, _gasPrice, _blockNumber, false, _ff);
}

dev::eth::ExecutionResult MixClient::create(Address const& _from, u256 _value, bytes const& _data, u256 _gas, u256 _gasPrice, BlockNumber _blockNumber, eth::FudgeFactor _ff)
{
	(void)_blockNumber;
	u256 n;
	Block temp;
	{
		ReadGuard lr(x_state);
		temp = asOf(eth::PendingBlock);
		n = temp.transactionsFrom(_from);
	}
	Transaction t(_value, _gasPrice, _gas, _data, n);
	t.forceSender(_from);
	if (_ff == FudgeFactor::Lenient)
		temp.mutableState().addBalance(_from, (u256)(t.gasRequired() * t.gasPrice() + t.value()));
	WriteGuard lw(x_state); //TODO: lock is required only for last execution state
	executeTransaction(t, temp, true, false);
	return lastExecution().result;
}

eth::BlockInfo MixClient::blockInfo() const
{
	ReadGuard l(x_state);
	return BlockInfo(bc().block());
}

void MixClient::setBeneficiary(Address _us)
{
	WriteGuard l(x_state);
	m_postMine.setBeneficiary(_us);
}

void MixClient::startMining()
{
	//no-op
}

void MixClient::stopMining()
{
	//no-op
}

bool MixClient::isMining() const
{
	return false;
}

uint64_t MixClient::hashrate() const
{
	return 0;
}

eth::WorkingProgress MixClient::miningProgress() const
{
	return eth::WorkingProgress();
}

}
}
