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
/** @file ClientModel.cpp
 * @author Yann yann@ethdev.com
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 * Ethereum IDE client.
 */

#include <QtConcurrent/QtConcurrent>
#include <QDebug>
#include <QQmlContext>
#include <QQmlApplicationEngine>
#include <libdevcore/CommonJS.h>
#include <libethereum/Transaction.h>
#include "ClientModel.h"
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "ContractCallDataEncoder.h"
#include "CodeModel.h"
#include "ClientModel.h"

using namespace dev;
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

ClientModel::ClientModel(AppContext* _context):
	m_context(_context), m_running(false)
{
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QVariableDefinitionList*>("QVariableDefinitionList*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QList<QVariableDeclaration*>>("QList<QVariableDeclaration*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qRegisterMetaType<AssemblyDebuggerData>("AssemblyDebuggerData");

	connect(this, &ClientModel::dataAvailable, this, &ClientModel::showDebugger, Qt::QueuedConnection);
	m_client.reset(new MixClient());

	_context->appEngine()->rootContext()->setContextProperty("clientModel", this);
}

void ClientModel::debugDeployment()
{
	executeSequence(std::vector<TransactionSettings>(), 10000000 * ether);
}

void ClientModel::debugState(QVariantMap _state)
{
	u256 balance = fromQString(_state.value("balance").toString());
	QVariantList transactions = _state.value("transactions").toList();

	std::vector<TransactionSettings> transactionSequence;

	for (auto const& t: transactions)
	{
		QVariantMap transaction = t.toMap();

		QString functionId = transaction.value("functionId").toString();
		u256 value = fromQString(transaction.value("value").toString());
		u256 gas = fromQString(transaction.value("gas").toString());
		u256 gasPrice = fromQString(transaction.value("gasPrice").toString());
		QVariantMap params = transaction.value("parameters").toMap();
		TransactionSettings transactionSettings(functionId, value, gas, gasPrice);

		for (auto p = params.cbegin(); p != params.cend(); ++p)
			transactionSettings.parameterValues.insert(std::make_pair(p.key(), fromQString(p.value().toString())));

		transactionSequence.push_back(transactionSettings);
	}
	executeSequence(transactionSequence, balance);
}

void ClientModel::executeSequence(std::vector<TransactionSettings> const& _sequence, u256 _balance)
{
	if (m_running)
		throw (std::logic_error("debugging already running"));
	auto compilerRes = m_context->codeModel()->code();
	std::shared_ptr<QContractDefinition> contractDef = compilerRes->sharedContract();
	m_running = true;

	emit runStarted();
	emit stateChanged();

	//run sequence
	QtConcurrent::run([=]()
	{
		try
		{
			bytes contractCode = compilerRes->bytes();
			std::vector<dev::bytes> transactonData;
			QFunctionDefinition* f;
			ContractCallDataEncoder c;
			//encode data for all transactions
			for (auto const& t: _sequence)
			{
				f = nullptr;
				for (int tf = 0; tf < contractDef->functionsList().size(); tf++)
				{
					if (contractDef->functionsList().at(tf)->name() == t.functionId)
					{
						f = contractDef->functionsList().at(tf);
						break;
					}
				}
				if (!f)
					throw std::runtime_error("function " + t.functionId.toStdString() + " not found");

				c.encode(f);
				for (int p = 0; p < f->parametersList().size(); p++)
				{
					QVariableDeclaration* var = (QVariableDeclaration*)f->parametersList().at(p);
					u256 value = 0;
					auto v = t.parameterValues.find(var->name());
					if (v != t.parameterValues.cend())
						value = v->second;
					c.encode(var, value);
				}
				transactonData.emplace_back(c.encodedData());
			}

			//run contract creation first
			m_client->resetState(_balance);
			ExecutionResult debuggingContent = deployContract(contractCode);
			Address address = debuggingContent.contractAddress;
			for (unsigned i = 0; i < _sequence.size(); ++i)
				debuggingContent = callContract(address, transactonData.at(i), _sequence.at(i));

			QList<QVariableDefinition*> returnParameters;

			if (f)
				returnParameters = c.decode(f->returnParameters(), debuggingContent.returnValue);

			//we need to wrap states in a QObject before sending to QML.
			QList<QObject*> wStates;
			for (unsigned i = 0; i < debuggingContent.machineStates.size(); i++)
			{
				QPointer<DebuggingStateWrapper> s(new DebuggingStateWrapper(debuggingContent.executionCode, debuggingContent.executionData.toBytes()));
				s->setState(debuggingContent.machineStates[i]);
				wStates.append(s);
			}
			//collect states for last transaction
			AssemblyDebuggerData code = DebuggingStateWrapper::getHumanReadableCode(debuggingContent.executionCode);
			emit dataAvailable(returnParameters, wStates, code);
			emit runComplete();
		}
		catch(boost::exception const&)
		{
			emit runFailed(QString::fromStdString(boost::current_exception_diagnostic_information()));
		}

		catch(std::exception const& e)
		{
			emit runFailed(e.what());
		}
		m_running = false;
		emit stateChanged();
	});
}

void ClientModel::showDebugger(QList<QVariableDefinition*> const& _returnParam, QList<QObject*> const& _wStates, AssemblyDebuggerData const& _code)
{
	m_context->appEngine()->rootContext()->setContextProperty("debugStates", QVariant::fromValue(_wStates));
	m_context->appEngine()->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(_code)));
	m_context->appEngine()->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(_code)));
	m_context->appEngine()->rootContext()->setContextProperty("contractCallReturnParameters", QVariant::fromValue(new QVariableDefinitionList(_returnParam)));
	showDebuggerWindow();
}

void ClientModel::showDebugError(QString const& _error)
{
	//TODO: change that to a signal
	m_context->displayMessageDialog(tr("Debugger"), _error);
}

ExecutionResult ClientModel::deployContract(bytes const& _code)
{
	u256 gasPrice = 10000000000000;
	u256 gas = 125000;
	u256 amount = 100;

	Address contractAddress = m_client->transact(m_client->userAccount().secret(), amount, _code, gas, gasPrice);
	ExecutionResult r = m_client->lastExecutionResult();
	r.contractAddress = contractAddress;
	return r;
}

ExecutionResult ClientModel::callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr)
{
	m_client->transact(m_client->userAccount().secret(), _tr.value, _contract, _data, _tr.gas, _tr.gasPrice);
	ExecutionResult r = m_client->lastExecutionResult();
	r.contractAddress = _contract;
	return r;
}

