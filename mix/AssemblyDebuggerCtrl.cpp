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
#include <QVariableDefinition.h>
#include <QQmlContext>
#include <QModelIndex>
#include <libdevcore/CommonJS.h>
#include "libethereum/Transaction.h"
#include "AssemblyDebuggerModel.h"
#include "AssemblyDebuggerCtrl.h"
#include "TransactionBuilder.h"
#include "KeyEventManager.h"
#include "ApplicationCtx.h"
#include "DebuggingStateWrapper.h"
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
	ApplicationCtx::getInstance()->getKeyEventManager()->registerEvent(this, SLOT(keyPressed(int)));
}

void AssemblyDebuggerCtrl::keyPressed(int _key)
{
	if (_key == Qt::Key_F5)
	{
		m_previousDebugResult = deployContract();
	}
	else if(_key == Qt::Key_F4)
		callContract(m_previousDebugResult.contractAddress);
	else if(_key == Qt::Key_F3)
	{
		//Reset state;
		m_modelDebugger.get()->resetState();
	}
}

void AssemblyDebuggerCtrl::callContract(Address contractAddress)
{
	CompilerResult compilerRes = m_compilation.get()->compile(m_doc->toPlainText());
	if (!compilerRes.success)
	{
		ApplicationCtx::getInstance()->displayMessageDialog("debugger","compilation failed");
		return;
	}
	ContractCallDataEncoder c;
	std::shared_ptr<QContractDefinition> contractDef = QContractDefinition::Contract(m_doc->toPlainText());
	QFunctionDefinition* fo = nullptr;
	for (int i = 0; i < contractDef->functions().size(); i++)
	{
		QFunctionDefinition* f = (QFunctionDefinition*)contractDef->functions().at(i);
		if (f->name() == "test2")
		{
			fo = f;
			c.encode(i);
			for (int k = 0; k < f->parameters().size(); k++)
			{
				c.encode((QVariableDeclaration*)f->parameters().at(k), QString("3"));
			}
		}
	}

	Transaction tr = m_trBuilder.getDefaultBasicTransaction(contractAddress, c.encodedData(), m_senderAddress);
	DebuggingContent debuggingContent = m_modelDebugger->getContractCallDebugStates(tr);
	debuggingContent.returnParameters = c.decode(fo->returnParameters(), debuggingContent.returnValue);
	finalizeExecution(debuggingContent);
}

DebuggingContent AssemblyDebuggerCtrl::deployContract()
{
	CompilerResult compilerRes = m_compilation.get()->compile(m_doc->toPlainText());
	if (!compilerRes.success)
	{
		ApplicationCtx::getInstance()->displayMessageDialog("debugger","compilation failed");
		DebuggingContent res;
		res.contentAvailable = false;
		return res;
	}

	Transaction tr = m_trBuilder.getDefaultCreationTransaction(compilerRes.bytes, m_senderAddress);
	DebuggingContent debuggingContent = m_modelDebugger->getContractInitiationDebugStates(tr);
	finalizeExecution(debuggingContent);
	return debuggingContent;
}

void AssemblyDebuggerCtrl::finalizeExecution(DebuggingContent debuggingContent)
{
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
	ApplicationCtx::getInstance()->appEngine()->rootContext()->setContextProperty("contractCallReturnParameters",
																				  QVariant::fromValue(new QVariableDefinitionList(debuggingContent.returnParameters)));
	this->addContentOn(this);
}
