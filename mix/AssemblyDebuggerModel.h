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
 * serves as a model to debug contract assembly code.
 */

#pragma once

#include <QObject>
#include <QList>
#include <libdevcore/Common.h>
#include <libdevcrypto/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include "DebuggingStateWrapper.h"
#include "TransactionListModel.h"

namespace dev
{
namespace mix
{

/**
 * @brief Long-life object for managing all executions.
 */
class AssemblyDebuggerModel
{
public:
	AssemblyDebuggerModel();
	DebuggingContent getContractCallDebugStates(Address _contract, bytes _data, dev::mix::TransactionSettings _tr);
	DebuggingContent getContractInitiationDebugStates(bytes _code);
	bool compile(QString);
	void resetState();

private:
	KeyPair m_userAccount;
	OverlayDB m_overlayDB;
	eth::State m_baseState;
	eth::State m_executiveState;
	std::unique_ptr<eth::Executive> m_currentExecution;
	DebuggingContent executeTransaction(dev::bytesConstRef _rawTransaction);
};

}

}
