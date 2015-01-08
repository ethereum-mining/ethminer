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
/** @file AssemblyDebuggerControl.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Extension which display debugging steps in assembly code.
 */

#pragma once

#include <atomic>
#include <QKeySequence>
#include "Extension.h"
#include "AssemblyDebuggerModel.h"

using AssemblyDebuggerData = std::tuple<QList<QObject*>, dev::mix::QQMLMap*>;

Q_DECLARE_METATYPE(AssemblyDebuggerData)
Q_DECLARE_METATYPE(dev::mix::DebuggingContent)

class AppContext;

namespace dev
{
namespace mix
{

/**
 * @brief Extension which display transaction creation or transaction call debugging. handle: F5 to deploy contract, F6 to reset state.
 */
class AssemblyDebuggerControl: public Extension
{
	Q_OBJECT

public:
	AssemblyDebuggerControl(AppContext* _context);
	~AssemblyDebuggerControl() {}
	void start() const override;
	QString title() const override;
	QString contentUrl() const override;

	Q_PROPERTY(bool running MEMBER m_running NOTIFY stateChanged)

private:
	void executeSequence(std::vector<TransactionSettings> const& _sequence, u256 _balance);

	std::unique_ptr<AssemblyDebuggerModel> m_modelDebugger;
	std::atomic<bool> m_running;

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
	/// Execution state changed
	void stateChanged();

	/// Emited when machine states are available.
	void dataAvailable(QList<QVariableDefinition*> const& _returnParams = QList<QVariableDefinition*>(), QList<QObject*> const& _wStates = QList<QObject*>(), AssemblyDebuggerData const& _code = AssemblyDebuggerData());
};

}
}
