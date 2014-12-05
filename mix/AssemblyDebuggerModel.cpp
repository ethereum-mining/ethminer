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
#include "ApplicationCtx.h"
#include "TransactionBuilder.h"
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

DebuggingContent AssemblyDebuggerModel::getContractInitiationDebugStates(dev::bytesConstRef _rawTransaction)
{
	QList<DebuggingState> states;
	Transaction tr(_rawTransaction);
	m_currentExecution.get()->create(tr.sender(), tr.value(), tr.gasPrice(), tr.gas(), &tr.data(), tr.sender());
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
			levels.push_back(&states.back());
		else
			levels.resize(ext.depth);

		states.append(DebuggingState({steps, ext.myAddress, vm.curPC(), inst, newMemSize, vm.gas(),
									  vm.stack(), vm.memory(), gasCost, ext.state().storage(ext.myAddress), levels}));
	};

	m_currentExecution.get()->go(onOp);
	m_currentExecution.get()->finalize(onOp);

	DebuggingContent d;
	d.states = states;
	d.executionCode = code;
	d.executionData = data;
	d.contentAvailable = true;
	d.message = "ok";
	return d;
}


DebuggingContent AssemblyDebuggerModel::getContractInitiationDebugStates(dev::u256 _value,
																			   dev::u256 _gasPrice,
																			   dev::u256 _gas,
																			   QString code,
																			   KeyPair _key)
{
	ConstantCompilationModel compiler;
	CompilerResult res = compiler.compile(code);
	if (!res.success)
	{
		DebuggingContent r;
		r.contentAvailable = false;
		r.message = "compile failed";
		return r;
	}

	TransactionBuilder trBuild;
	Transaction tr = trBuild.getCreationTransaction(_value, _gasPrice, _gas, res.bytes,
													m_executiveState.transactionsFrom(dev::toAddress(_key.secret())), _key.secret());
	bytes b = tr.rlp();
	dev::bytesConstRef bytesRef = &b;
	return getContractInitiationDebugStates(bytesRef);
}

bool AssemblyDebuggerModel::compile(QString code)
{
	ConstantCompilationModel compiler;
	CompilerResult res = compiler.compile(code);
	return res.success;
}
