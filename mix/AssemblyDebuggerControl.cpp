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
/** @file AssemblyDebuggerControl.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * display opcode debugging.
 */

//These 2 includes should be at the top to avoid conflicts with macros defined in windows.h
//@todo fix this is solidity headers
#include <libsolidity/Token.h>
#include <libsolidity/Types.h>
#include <utility>
#include <QtConcurrent/QtConcurrent>
#include <QDebug>
#include <QQmlContext>
#include <QQmlApplicationEngine>
#include <QModelIndex>
#include <libdevcore/CommonJS.h>
#include <libethereum/Transaction.h>
#include "AssemblyDebuggerModel.h"
#include "AssemblyDebuggerControl.h"
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "ContractCallDataEncoder.h"
#include "CodeModel.h"

using namespace dev::eth;
using namespace dev::mix;

/// @todo Move this to QML
dev::u256 fromQString(QString const& _s)
{
	return dev::jsToU256(_s.toStdString());
}

/// @todo Move this to QML
QString toQString(dev::u256 _value)
{
	std::ostringstream s;
	s << _value;
	return QString::fromStdString(s.str());
}

AssemblyDebuggerControl::AssemblyDebuggerControl(AppContext* _context): Extension(_context, ExtensionDisplayBehavior::ModalDialog)
{
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QVariableDefinitionList*>("QVariableDefinitionList*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QList<QVariableDeclaration*>>("QList<QVariableDeclaration*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qRegisterMetaType<AssemblyDebuggerData>("AssemblyDebuggerData");
	qRegisterMetaType<DebuggingStatusResult>("DebuggingStatusResult");

	connect(this, SIGNAL(dataAvailable(bool, DebuggingStatusResult, QList<QVariableDefinition*>, QList<QObject*>, AssemblyDebuggerData)),
			this, SLOT(updateGUI(bool, DebuggingStatusResult, QList<QVariableDefinition*>, QList<QObject*>, AssemblyDebuggerData)), Qt::QueuedConnection);

	_context->appEngine()->rootContext()->setContextProperty("debugModel", this);

	m_modelDebugger = std::unique_ptr<AssemblyDebuggerModel>(new AssemblyDebuggerModel);
}

QString AssemblyDebuggerControl::contentUrl() const
{
	return QStringLiteral("qrc:/qml/Debugger.qml");
}

QString AssemblyDebuggerControl::title() const
{
	return QApplication::tr("debugger");
}

void AssemblyDebuggerControl::start() const
{
}

void AssemblyDebuggerControl::debugDeployment()
{
	deployContract();
}

void AssemblyDebuggerControl::debugState(QObject* _state)
{
	/*
	QString functionId = _transaction->property("functionId").toString();
	u256 value = fromQString(_transaction->property("value").toString());
	u256 gas = fromQString(_transaction->property("gas").toString());
	u256 gasPrice = fromQString(_transaction->property("gasPrice").toString());
	QVariantMap params = _transaction->property("parameters").toMap();
	TransactionSettings transaction(functionId, value, gas, gasPrice);

	for (auto p = params.cbegin(); p != params.cend(); ++p)
		transaction.parameterValues.insert(std::make_pair(p.key(), fromQString(p.value().toString())));
	runTransaction(transaction);
	*/
}

void AssemblyDebuggerControl::resetState()
{
	m_modelDebugger->resetState();
	m_ctx->displayMessageDialog(QApplication::tr("State status"), QApplication::tr("State reseted ... need to redeploy contract"));
}

void AssemblyDebuggerControl::callContract(TransactionSettings _tr, dev::Address _contract)
{
	auto compilerRes = m_ctx->codeModel()->code();
	if (!compilerRes->successfull())
		m_ctx->displayMessageDialog("debugger","compilation failed");
	else
	{
		ContractCallDataEncoder c;
		QContractDefinition const* contractDef = compilerRes->contract();
		QFunctionDefinition* f = nullptr;
		for (int k = 0; k < contractDef->functionsList().size(); k++)
		{
			if (contractDef->functionsList().at(k)->name() == _tr.functionId)
			{
				f = contractDef->functionsList().at(k);
				break;
			}
		}
		if (!f)
			m_ctx->displayMessageDialog(QApplication::tr("debugger"), QApplication::tr("function not found. Please redeploy this contract."));
		else
		{
			c.encode(f->index());
			for (int k = 0; k < f->parametersList().size(); k++)
			{
				QVariableDeclaration* var = (QVariableDeclaration*)f->parametersList().at(k);
				c.encode(var, _tr.parameterValues[var->name()]);
			}
			DebuggingContent debuggingContent = m_modelDebugger->callContract(_contract, c.encodedData(), _tr);
			debuggingContent.returnParameters = c.decode(f->returnParameters(), debuggingContent.returnValue);
			finalizeExecution(debuggingContent);
		}
	}
}

void AssemblyDebuggerControl::deployContract()
{
	auto compilerRes = m_ctx->codeModel()->code();
	if (!compilerRes->successfull())
		emit dataAvailable(false, DebuggingStatusResult::Compilationfailed);
	else
	{
		m_previousDebugResult = m_modelDebugger->deployContract(compilerRes->bytes());
		finalizeExecution(m_previousDebugResult);
	}
}

void AssemblyDebuggerControl::finalizeExecution(DebuggingContent _debuggingContent)
{
	//we need to wrap states in a QObject before sending to QML.
	QList<QObject*> wStates;
	for(int i = 0; i < _debuggingContent.machineStates.size(); i++)
	{
		QPointer<DebuggingStateWrapper> s(new DebuggingStateWrapper(_debuggingContent.executionCode, _debuggingContent.executionData.toBytes()));
		s->setState(_debuggingContent.machineStates.at(i));
		wStates.append(s);
	}
	AssemblyDebuggerData code = DebuggingStateWrapper::getHumanReadableCode(_debuggingContent.executionCode);
	emit dataAvailable(true, DebuggingStatusResult::Ok, _debuggingContent.returnParameters, wStates, code);
}

void AssemblyDebuggerControl::updateGUI(bool _success, DebuggingStatusResult const& _reason, QList<QVariableDefinition*> const& _returnParam, QList<QObject*> const& _wStates, AssemblyDebuggerData const& _code)
{
	Q_UNUSED(_reason);
	if (_success)
	{
		m_appEngine->rootContext()->setContextProperty("debugStates", QVariant::fromValue(_wStates));
		m_appEngine->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(_code)));
		m_appEngine->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(_code)));
		m_appEngine->rootContext()->setContextProperty("contractCallReturnParameters", QVariant::fromValue(new QVariableDefinitionList(_returnParam)));
		this->addContentOn(this);
	}
	else
		m_ctx->displayMessageDialog(QApplication::tr("debugger"), QApplication::tr("compilation failed"));
}

void AssemblyDebuggerControl::runTransaction(TransactionSettings const& _tr)
{
	callContract(_tr, m_previousDebugResult.contractAddress);
}
