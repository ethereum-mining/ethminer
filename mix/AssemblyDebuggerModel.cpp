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
/** @file AssemblyDebuggerModel.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * used as a model to debug contract assembly code.
 */

#include <QApplication>
#include <libdevcore/Common.h>
#include <libevm/VM.h>
#include <libethereum/Executive.h>
#include <libethereum/Transaction.h>
#include <libethereum/ExtVM.h>
#include "AppContext.h"
#include "TransactionListModel.h"
#include "AssemblyDebuggerModel.h"
#include "ConstantCompilationModel.h"
#include "DebuggingStateWrapper.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

AssemblyDebuggerModel::AssemblyDebuggerModel():
	m_userAccount(KeyPair::create()),
	m_baseState(Address(), m_overlayDB, BaseState::Empty)
{
	m_baseState.addBalance(m_userAccount.address(), 10000000 * ether);
	m_executiveState = m_baseState;
	m_currentExecution = std::unique_ptr<Executive>(new Executive(m_executiveState, LastHashes(), 0));
}

DebuggingContent AssemblyDebuggerModel::executeTransaction(bytesConstRef const& _rawTransaction)
{
	QList<DebuggingState> machineStates;
	m_currentExecution.reset(new Executive(m_executiveState, LastHashes(), 0));
	m_currentExecution->setup(_rawTransaction);
	std::vector<DebuggingState const*> levels;
	bytes code;
	bytesConstRef data;
	bool firstIteration = true;
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

		machineStates.append(DebuggingState({steps, ext.myAddress, vm.curPC(), inst, newMemSize, vm.gas(),
									  vm.stack(), vm.memory(), gasCost, ext.state().storage(ext.myAddress), levels}));
	};

	m_currentExecution->go(onOp);
	m_currentExecution->finalize(onOp);
	m_executiveState.completeMine();

	DebuggingContent d;
	d.returnValue = m_currentExecution->out().toVector();
	d.machineStates = machineStates;
	d.executionCode = code;
	d.executionData = data;
	d.contentAvailable = true;
	d.message = "ok";
	return d;
}

DebuggingContent AssemblyDebuggerModel::deployContract(bytes const& _code)
{
	u256 gasPrice = 10000000000000;
	u256 gas = 1000000;
	u256 amount = 100;
	Transaction _tr(amount, gasPrice, min(gas, m_baseState.gasLimitRemaining()), _code, m_executiveState.transactionsFrom(dev::toAddress(m_userAccount.secret())), m_userAccount.secret());
	bytes b = _tr.rlp();
	dev::bytesConstRef bytesRef = &b;
	DebuggingContent d = executeTransaction(bytesRef);
	h256 th = sha3(rlpList(_tr.sender(), _tr.nonce()));
	d.contractAddress = right160(th);
	return d;
}

DebuggingContent AssemblyDebuggerModel::callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr)
{
	Transaction tr = Transaction(_tr.value, _tr.gasPrice, min(_tr.gas, m_baseState.gasLimitRemaining()), _contract, _data, m_executiveState.transactionsFrom(dev::toAddress(m_userAccount.secret())), m_userAccount.secret());
	bytes b = tr.rlp();
	dev::bytesConstRef bytesRef = &b;
	DebuggingContent d = executeTransaction(bytesRef);
	d.contractAddress = tr.receiveAddress();
	return d;
}

void AssemblyDebuggerModel::resetState()
{
	// Reset the state back to our clean premine.
	m_executiveState = m_baseState;
}
