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
/** @file ClientModel.h
 * @author Yann yann@ethdev.com
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#pragma once

#include <atomic>
#include "DebuggingStateWrapper.h"
#include "MixClient.h"

using AssemblyDebuggerData = std::tuple<QList<QObject*>, dev::mix::QQMLMap*>;

Q_DECLARE_METATYPE(AssemblyDebuggerData)
Q_DECLARE_METATYPE(dev::mix::ExecutionResult)

namespace dev
{
namespace mix
{

class AppContext;
class Web3Server;
class RpcConnector;

/// Backend transaction config class
struct TransactionSettings
{
	TransactionSettings() {}
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

public:
	/// @returns true if the functionId has not be set
	bool isEmpty() const { return functionId.isNull() || functionId.isEmpty(); }
};


/**
 * @brief Ethereum state control
 */
class ClientModel: public QObject
{
	Q_OBJECT

public:
	ClientModel(AppContext* _context);
	~ClientModel();
	/// @returns true if currently executing contract code
	Q_PROPERTY(bool running MEMBER m_running NOTIFY stateChanged)
	/// @returns address of the last executed contract
	Q_PROPERTY(QString contractAddress READ contractAddress NOTIFY contractAddressChanged)
	/// ethereum.js RPC request entry point
	/// @param _message RPC request in Json format
	/// @returns RPC response in Json format
	Q_INVOKABLE QString apiCall(QString const& _message);

public slots:
	/// Run the contract constructor and show debugger window.
	void debugDeployment();
	/// Setup state, run transaction sequence, show debugger for the last transaction
	/// @param _state JS object with state configuration
	void debugState(QVariantMap _state);

private slots:
	/// Update UI with machine states result. Display a modal dialog.
	void showDebugger(QList<QVariableDefinition*> const& _returnParams = QList<QVariableDefinition*>(), QList<QObject*> const& _wStates = QList<QObject*>(), AssemblyDebuggerData const& _code = AssemblyDebuggerData());
	/// Update UI with transaction run error.
	void showDebugError(QString const& _error);

signals:
	/// Transaction execution started
	void runStarted();
	/// Transaction execution completed successfully
	void runComplete();
	/// Transaction execution completed with error
	/// @param _message Error message
	void runFailed(QString const& _message);
	/// Contract address changed
	void contractAddressChanged();
	/// Execution state changed
	void stateChanged();
	/// Show debugger window request
	void showDebuggerWindow();
	/// ethereum.js RPC response ready
	/// @param _message RPC response in Json format
	void apiResponse(QString const& _message);

	/// Emited when machine states are available.
	void dataAvailable(QList<QVariableDefinition*> const& _returnParams = QList<QVariableDefinition*>(), QList<QObject*> const& _wStates = QList<QObject*>(), AssemblyDebuggerData const& _code = AssemblyDebuggerData());

private:
	QString contractAddress() const;
	void executeSequence(std::vector<TransactionSettings> const& _sequence, u256 _balance, TransactionSettings const& _ctrTransaction = TransactionSettings());
	ExecutionResult deployContract(bytes const& _code, TransactionSettings const& _tr = TransactionSettings());
	ExecutionResult callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr);

	AppContext* m_context;
	std::atomic<bool> m_running;
	std::unique_ptr<MixClient> m_client;
	std::unique_ptr<RpcConnector> m_rpcConnector;
	std::unique_ptr<Web3Server> m_web3Server;
};

}
}
