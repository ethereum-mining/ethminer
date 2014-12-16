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

#include <QtConcurrent/QtConcurrent>
#include <QDebug>
#include <QQmlContext>
#include <QModelIndex>
#include "libethereum/Transaction.h"
#include "AssemblyDebuggerModel.h"
#include "AssemblyDebuggerCtrl.h"
#include "TransactionBuilder.h"
#include "KeyEventManager.h"
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
using namespace dev::mix;

AssemblyDebuggerCtrl::AssemblyDebuggerCtrl(QTextDocument* _doc): Extension(ExtensionDisplayBehavior::ModalDialog)
{
	qRegisterMetaType<AssemblyDebuggerData>();
	qRegisterMetaType<DebuggingStatusResult>();
	connect(this, SIGNAL(dataAvailable(bool, DebuggingStatusResult, QList<QObject*>, AssemblyDebuggerData)),
			this, SLOT(updateGUI(bool, DebuggingStatusResult, QList<QObject*>, AssemblyDebuggerData)), Qt::QueuedConnection);
	m_modelDebugger = std::unique_ptr<AssemblyDebuggerModel>(new AssemblyDebuggerModel);
	m_doc = _doc;
}

QString AssemblyDebuggerCtrl::contentUrl() const
{
	return QStringLiteral("qrc:/qml/Debugger.qml");
}

QString AssemblyDebuggerCtrl::title() const
{
	return QApplication::tr("debugger");
}

void AssemblyDebuggerCtrl::start() const
{
	//start to listen on F5
	m_ctx->getKeyEventManager()->registerEvent(this, SLOT(keyPressed(int)));
}

void AssemblyDebuggerCtrl::keyPressed(int _key)
{
	if (_key == Qt::Key_F5)
	{
		QString code = m_doc->toPlainText();
		QtConcurrent::run([this, code]()
		{
			if (!m_modelDebugger->compile(m_doc->toPlainText()))
			{
				emit dataAvailable(false, DebuggingStatusResult::Compilationfailed);
				return;
			}

			u256 gasPrice = 10000000000000;
			u256 gas = 1000000;
			u256 amount = 100;
			DebuggingContent debuggingContent = m_modelDebugger->getContractInitiationDebugStates(amount, gasPrice, gas, m_doc->toPlainText());

			//we need to wrap states in a QObject before sending to QML.
			QList<QObject*> wStates;
			for(int i = 0; i < debuggingContent.states.size(); i++)
			{
				DebuggingStateWrapper* s = new DebuggingStateWrapper(debuggingContent.executionCode, debuggingContent.executionData.toBytes(), this);
				s->setState(debuggingContent.states.at(i));
				wStates.append(s);
			}
			AssemblyDebuggerData code = DebuggingStateWrapper::getHumanReadableCode(debuggingContent.executionCode, this);
			emit dataAvailable(true, DebuggingStatusResult::Ok, wStates, code);
		});
	}
}

void AssemblyDebuggerCtrl::updateGUI(bool success, DebuggingStatusResult reason, QList<QObject*> _wStates, AssemblyDebuggerData _code)
{
	Q_UNUSED(reason);
	if (success)
	{
		m_appEngine->rootContext()->setContextProperty("debugStates", QVariant::fromValue(_wStates));
		m_appEngine->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(_code)));
		m_appEngine->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(_code)));
		this->addContentOn(this);
	}
	else
		m_ctx->displayMessageDialog(QApplication::tr("debugger"), QApplication::tr("compilation failed"));
}
