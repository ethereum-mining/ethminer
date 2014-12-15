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
/** @file AssemblyDebuggerModel.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * used as a model to debug contract assembly code.
 */

#include "libethereum/Executive.h"
#include "libethereum/Transaction.h"
#include "libethereum/ExtVM.h"
#include "libevm/VM.h"
#include "libdevcore/Common.h"
#include "AppContext.h"
#include "TransactionBuilder.h"
#include "TransactionListModel.h"
#include "AssemblyDebuggerModel.h"
#include "ConstantCompilationModel.h"
#include "DebuggingStateWrapper.h"
using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

AssemblyDebuggerModel::AssemblyDebuggerModel()
{
	m_currentExecution = std::unique_ptr<Executive>(new Executive(m_executiveState));
}

void AssemblyDebuggerModel::addBalance(KeyPair address, u256 amount)
{
	//m_currentExecution = std::unique_ptr<Executive>(new Executive(m_executiveState));
	m_executiveState.addBalance(dev::toAddress(address.secret()), amount);
	//m_currentExecution.reset();
}

DebuggingContent AssemblyDebuggerModel::executeTransaction()
{
	QList<DebuggingState> machineStates;
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

	m_currentExecution.get()->go(onOp);
	m_currentExecution.get()->finalize(onOp);
	m_executiveState.completeMine();

	DebuggingContent d;
	d.returnValue = m_currentExecution.get()->out().toVector();
	d.machineStates = machineStates;
	d.executionCode = code;
	d.executionData = data;
	d.contentAvailable = true;
	d.message = "ok";
	return d;
}

DebuggingContent AssemblyDebuggerModel::getContractInitiationDebugStates(bytes _code, KeyPair _sender)
{
	TransactionBuilder trBuilder;
	dev::eth::Transaction _tr = trBuilder.getDefaultCreationTransaction(_code, _sender,
																		m_executiveState.transactionsFrom(dev::toAddress(_sender.secret())));
	bytes b = _tr.rlp();
	dev::bytesConstRef bytesRef = &b;
	m_currentExecution.get()->forceSetup(bytesRef);
	DebuggingContent d = executeTransaction();

	h256 th = sha3(rlpList(_tr.sender(), _tr.nonce()));
	d.contractAddress = right160(th);
	m_currentExecution.reset();
	return d;
}

DebuggingContent AssemblyDebuggerModel::getContractCallDebugStates(Address _contract, bytes _data,
																   KeyPair _sender, dev::mix::TransactionSettings _tr)
{

	TransactionBuilder trBuilder;
	dev::eth::Transaction tr = trBuilder.getBasicTransaction(_tr.value,_tr.gasPrice,_tr.gas, _contract, _data,
												   m_executiveState.transactionsFrom(dev::toAddress(_sender.secret())), _sender.secret());

	m_currentExecution = std::unique_ptr<Executive>(new Executive(m_executiveState));
	bytes b = tr.rlp();
	dev::bytesConstRef bytesRef = &b;
	m_currentExecution.get()->forceSetup(bytesRef);
	DebuggingContent d = executeTransaction();

	d.contractAddress = tr.receiveAddress();
	m_currentExecution.reset();
	return d;
}

void AssemblyDebuggerModel::resetState()
{
	m_executiveState = State();
}

