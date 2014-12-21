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
/** @file AssemblyDebuggerCtrl.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <QKeySequence>
#include "Extension.h"
#include "TransactionListModel.h"
#include "AssemblyDebuggerModel.h"

using AssemblyDebuggerData = std::tuple<QList<QObject*>, dev::mix::QQMLMap*>;
enum DebuggingStatusResult
{
	Ok,
	Compilationfailed
};

Q_DECLARE_METATYPE(AssemblyDebuggerData)
Q_DECLARE_METATYPE(DebuggingStatusResult)
Q_DECLARE_METATYPE(dev::mix::DebuggingContent)

class AppContext;

namespace dev
{
namespace mix
{

class AssemblyDebuggerCtrl: public Extension
{
	Q_OBJECT

public:
	AssemblyDebuggerCtrl(AppContext* _context);
	~AssemblyDebuggerCtrl() {}
	void start() const override;
	QString title() const override;
	QString contentUrl() const override;

private:
	std::unique_ptr<AssemblyDebuggerModel> m_modelDebugger;
	void deployContract();
	void callContract(dev::mix::TransactionSettings _contractAddress);
	void finalizeExecution(DebuggingContent _content);
	DebuggingContent m_previousDebugResult; //used for now to keep the contract address in case of transaction call.

public slots:
	void keyPressed(int);
	void updateGUI(bool _success, DebuggingStatusResult _reason, QList<QVariableDefinition*> _returnParams = QList<QVariableDefinition*>(), QList<QObject*> _wStates = QList<QObject*>(), AssemblyDebuggerData _code = AssemblyDebuggerData());
	void runTransaction(dev::mix::TransactionSettings _tr);

signals:
	void dataAvailable(bool _success, DebuggingStatusResult _reason, QList<QVariableDefinition*> _returnParams = QList<QVariableDefinition*>(), QList<QObject*> _wStates = QList<QObject*>(), AssemblyDebuggerData _code = AssemblyDebuggerData());
};

}

}
