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

#include <QKeySequence>
#include <QTextDocument>
#include "Extension.h"
#include "ConstantCompilationModel.h"
#include "TransactionListModel.h"
#include "AssemblyDebuggerModel.h"
#include "AppContext.h"

using AssemblyDebuggerData = std::tuple<QList<QObject*>, dev::mix::QQMLMap*>;
enum DebuggingStatusResult
{
	Ok,
	Compilationfailed
};

Q_DECLARE_METATYPE(AssemblyDebuggerData)
Q_DECLARE_METATYPE(DebuggingStatusResult)
Q_DECLARE_METATYPE(dev::mix::DebuggingContent)

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
	AssemblyDebuggerControl(QTextDocument* _doc);
	~AssemblyDebuggerControl() {}
	void start() const override;
	QString title() const override;
	QString contentUrl() const override;

private:
	void deployContract(QString _source);
	void callContract(TransactionSettings _tr, Address _contract);
	void finalizeExecution(DebuggingContent _content);

	std::unique_ptr<AssemblyDebuggerModel> m_modelDebugger;
	std::unique_ptr<ConstantCompilationModel> m_compilation;
	DebuggingContent m_previousDebugResult; //TODO: to be replaced in a more consistent struct. Used for now to keep the contract address in case of future transaction call.
	QTextDocument* m_doc;

public slots:
	/// Handle key pressed. F5 deploy contract - F6 reset state.
	void keyPressed(int);
	/// Update UI with machine states result. Display a modal dialog.
	void updateGUI(bool _success, DebuggingStatusResult const& _reason, QList<QVariableDefinition*> const& _returnParams = QList<QVariableDefinition*>(), QList<QObject*> const& _wStates = QList<QObject*>(), AssemblyDebuggerData const& _code = AssemblyDebuggerData());
	/// Run the given transaction.
	void runTransaction(TransactionSettings const& _tr);

signals:
	/// Emited when machine states are available.
	void dataAvailable(bool _success, DebuggingStatusResult const& _reason, QList<QVariableDefinition*> const& _returnParams = QList<QVariableDefinition*>(), QList<QObject*> const& _wStates = QList<QObject*>(), AssemblyDebuggerData const& _code = AssemblyDebuggerData());
};

}
}
