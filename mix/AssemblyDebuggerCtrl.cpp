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
#include <QVariableDefinition.h>
#include <QQmlContext>
#include <QModelIndex>
#include <libdevcore/CommonJS.h>
#include "libethereum/Transaction.h"
#include "AssemblyDebuggerModel.h"
#include "AssemblyDebuggerCtrl.h"
#include "TransactionBuilder.h"
#include "KeyEventManager.h"
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
#include "TransactionListModel.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "ContractCallDataEncoder.h"
using namespace dev::eth;
using namespace dev::mix;

AssemblyDebuggerCtrl::AssemblyDebuggerCtrl(QTextDocument* _doc): Extension(ExtensionDisplayBehavior::ModalDialog)
{
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QVariableDefinitionList*>("QVariableDefinitionList*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");

	qRegisterMetaType<AssemblyDebuggerData>();
	qRegisterMetaType<DebuggingStatusResult>();
	connect(this, SIGNAL(dataAvailable(bool, DebuggingStatusResult, QList<QObject*>, AssemblyDebuggerData)),
			this, SLOT(updateGUI(bool, DebuggingStatusResult, QList<QObject*>, AssemblyDebuggerData)), Qt::QueuedConnection);

	m_modelDebugger = std::unique_ptr<AssemblyDebuggerModel>(new AssemblyDebuggerModel);
	m_compilation  = std::unique_ptr<ConstantCompilationModel>(new ConstantCompilationModel);
	m_senderAddress = KeyPair::create(); //this address will be used as the sender address.
	m_modelDebugger.get()->addBalance(m_senderAddress, 10000000000000*1000000 + 10000000);
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
	m_ctx->getKeyEventManager()->registerEvent(this, SLOT(keyPressed(int)));
}

void AssemblyDebuggerCtrl::keyPressed(int _key)
{
	if (_key == Qt::Key_F5)
	{
		m_previousDebugResult = deployContract();
	}
	/*else if(_key == Qt::Key_F4)
		callContract(m_previousDebugResult.contractAddress);
	else if(_key == Qt::Key_F3)
	{
		//Reset state;
		m_modelDebugger.get()->resetState();
	}*/
}

void AssemblyDebuggerCtrl::callContract(dev::mix::TransactionSettings _tr)
{
	CompilerResult compilerRes = m_compilation.get()->compile(m_doc->toPlainText());
	if (!compilerRes.success)
	{
		AppContext::getInstance()->displayMessageDialog("debugger","compilation failed");
		return;
	}

	ContractCallDataEncoder c;
	std::shared_ptr<QContractDefinition> contractDef = QContractDefinition::Contract(m_doc->toPlainText());

	QFunctionDefinition* f = nullptr;
	for (int k = 0; k < contractDef.get()->functions().size(); k++)
	{
		if (contractDef.get()->functions().at(k)->name() == _tr.functionId)
		{
			f = (QFunctionDefinition*)contractDef->functions().at(k);
		}
	}
	if (!f)
	{
		AppContext::getInstance()->displayMessageDialog("debugger","contract code changed. redeploy contract");
		return;
	}

	c.encode(f->index());
	for (int k = 0; k < f->parameters().size(); k++)
	{
		QVariableDeclaration* var = (QVariableDeclaration*)f->parameters().at(k);
		c.encode(var, _tr.parameterValues[var->name()]);
	}

	DebuggingContent debuggingContent = m_modelDebugger->getContractCallDebugStates(m_previousDebugResult.contractAddress,
																					c.encodedData(),
																					m_senderAddress,
																					_tr);
	debuggingContent.returnParameters = c.decode(f->returnParameters(), debuggingContent.returnValue);
	finalizeExecution(debuggingContent);
}

DebuggingContent AssemblyDebuggerCtrl::deployContract()
{
	CompilerResult compilerRes = m_compilation.get()->compile(m_doc->toPlainText());
	if (!compilerRes.success)
	{
		AppContext::getInstance()->displayMessageDialog("debugger","compilation failed");
		DebuggingContent res;
		res.contentAvailable = false;
		return res;
	}

	DebuggingContent debuggingContent = m_modelDebugger->getContractInitiationDebugStates(compilerRes.bytes, m_senderAddress);
	finalizeExecution(debuggingContent);
	return debuggingContent;
}

void AssemblyDebuggerCtrl::finalizeExecution(DebuggingContent debuggingContent)
{
	//we need to wrap states in a QObject before sending to QML.
	QList<QObject*> wStates;
	for(int i = 0; i < debuggingContent.machineStates.size(); i++)
	{
		DebuggingStateWrapper* s = new DebuggingStateWrapper(debuggingContent.executionCode, debuggingContent.executionData.toBytes(), this);
		s->setState(debuggingContent.machineStates.at(i));
		wStates.append(s);
	}
	std::tuple<QList<QObject*>, QQMLMap*> code = DebuggingStateWrapper::getHumanReadableCode(debuggingContent.executionCode, this);
	AppContext::getInstance()->appEngine()->rootContext()->setContextProperty("debugStates", QVariant::fromValue(wStates));
	AppContext::getInstance()->appEngine()->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(code)));
	AppContext::getInstance()->appEngine()->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(code)));
	AppContext::getInstance()->appEngine()->rootContext()->setContextProperty("contractCallReturnParameters",
																				  QVariant::fromValue(new QVariableDefinitionList(debuggingContent.returnParameters)));
	this->addContentOn(this);
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
		m_ctx->displayMessageDialog("debugger","compilation failed");
}

void AssemblyDebuggerCtrl::runTransaction(TransactionSettings _tr)
{
	callContract(_tr);
}
