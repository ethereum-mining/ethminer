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
 * display opcode debugging.
 */

#include <QDebug>
#include <QQmlContext>
#include <QModelIndex>
#include "libethereum/Transaction.h"
#include "AssemblyDebuggerModel.h"
#include "AssemblyDebuggerCtrl.h"
#include "TransactionBuilder.h"
#include "KeyEventManager.h"
#include "ApplicationCtx.h"
#include "DebuggingStateWrapper.h"
using namespace dev::mix;

AssemblyDebuggerCtrl::AssemblyDebuggerCtrl(QTextDocument* _doc): Extension(ExtensionDisplayBehavior::ModalDialog)
{
	m_modelDebugger = std::unique_ptr<AssemblyDebuggerModel>(new AssemblyDebuggerModel);
	m_doc = _doc;
}

QString AssemblyDebuggerCtrl::contentUrl() const
{
	return QStringLiteral("qrc:/qml/Debugger.qml");
}

QString AssemblyDebuggerCtrl::title() const
{
	return "debugger";
}

void AssemblyDebuggerCtrl::start() const
{
	//start to listen on F5
	ApplicationCtx::getInstance()->getKeyEventManager()->registerEvent(this, SLOT(keyPressed(int)));
}

void AssemblyDebuggerCtrl::keyPressed(int _key)
{
	if (_key == Qt::Key_F5)
	{
			if (!m_modelDebugger->compile(m_doc->toPlainText()))
			{
				ApplicationCtx::getInstance()->displayMessageDialog("debugger","compilation failed");
				return;
			}

			KeyPair ad = KeyPair::create();
			u256 gasPrice = 10000000000000;
			u256 gas = 1000000;
			u256 amount = 100;
			DebuggingContent debuggingContent = m_modelDebugger->getContractInitiationDebugStates(amount, gasPrice, gas, m_doc->toPlainText(), ad);

			//we need to wrap states in a QObject before sending to QML.
			QList<QObject*> wStates;
			for(int i = 0; i < debuggingContent.states.size(); i++)
			{
				DebuggingStateWrapper* s = new DebuggingStateWrapper(debuggingContent.executionCode, debuggingContent.executionData.toBytes());
				s->setState(debuggingContent.states.at(i));
				wStates.append(s);
			}
			std::tuple<QList<QObject*>, QQMLMap*> code = DebuggingStateWrapper::getHumanReadableCode(debuggingContent.executionCode, this);
			ApplicationCtx::getInstance()->appEngine()->rootContext()->setContextProperty("debugStates", QVariant::fromValue(wStates));
			ApplicationCtx::getInstance()->appEngine()->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(code)));
			ApplicationCtx::getInstance()->appEngine()->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(code)));
			this->addContentOn(this);
	};
}
