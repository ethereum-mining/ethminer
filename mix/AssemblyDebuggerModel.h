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
 * Used as a model to debug contract assembly code.
 */

#pragma once

#include <QObject>
#include <QList>
#include <libdevcore/Common.h>
#include <libdevcrypto/Common.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include "DebuggingStateWrapper.h"

namespace dev
{
namespace mix
{

/// Backend transaction config class
struct TransactionSettings
{
	TransactionSettings(QString const& _functionId, u256 _value, u256 _gas, u256 _gasPrice):
		functionId(_functionId), value(_value), gas(_gas), gasPrice(_gasPrice) {}

	/// Contract function name
	QString functionId;
	/// Transaction value
	u256 value;
	/// Gas
	u256 gas;
	/// Gas price
	u256 gasPrice;
	/// Mapping from contract function parameter name to value
	std::map<QString, u256> parameterValues;
};


/**
 * @brief Long-life object for managing all executions.
 */
class AssemblyDebuggerModel
{
public:
	AssemblyDebuggerModel();
	/// Call function in a already deployed contract.
	DebuggingContent callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr);
	/// Deploy the contract described by _code.
	DebuggingContent deployContract(bytes const& _code);
	/// Reset state to the empty state with given balance.
	void resetState(u256 _balance);

private:
	KeyPair m_userAccount;
	OverlayDB m_overlayDB;
	eth::State m_executiveState;
	DebuggingContent executeTransaction(dev::bytesConstRef const& _rawTransaction);
};

}
}
