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
#include <jsonrpccpp/server.h>
#include <libdevcore/CommonJS.h>
#include <libethereum/Transaction.h>
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "ContractCallDataEncoder.h"
#include "CodeModel.h"
#include "ClientModel.h"
#include "QEther.h"
#include "Web3Server.h"
#include "ClientModel.h"

using namespace dev;
using namespace dev::eth;

namespace dev
{
namespace mix
{

class RpcConnector: public jsonrpc::AbstractServerConnector
{
public:
	virtual bool StartListening() override { return true; }
	virtual bool StopListening() override { return true; }
	virtual bool SendResponse(std::string const& _response, void*) override
	{
		m_response = QString::fromStdString(_response);
		return true;
	}
	QString response() const { return m_response; }

private:
	QString m_response;
};

ClientModel::ClientModel(AppContext* _context):
	m_context(_context), m_running(false), m_rpcConnector(new RpcConnector()), m_contractAddress(Address())
{
	qRegisterMetaType<QBigInt*>("QBigInt*");
	qRegisterMetaType<QEther*>("QEther*");
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QVariableDefinitionList*>("QVariableDefinitionList*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QList<QVariableDeclaration*>>("QList<QVariableDeclaration*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qRegisterMetaType<AssemblyDebuggerData>("AssemblyDebuggerData");
	qRegisterMetaType<TransactionLogEntry*>("TransactionLogEntry");

	connect(this, &ClientModel::runComplete, this, &ClientModel::showDebugger, Qt::QueuedConnection);
	m_client.reset(new MixClient());
	connect(m_client.get(), &MixClient::stateReset, this, &ClientModel::onStateReset, Qt::QueuedConnection);
	connect(m_client.get(), &MixClient::newTransaction, this, &ClientModel::onNewTransaction, Qt::QueuedConnection);

	m_web3Server.reset(new Web3Server(*m_rpcConnector.get(), std::vector<dev::KeyPair> { m_client->userAccount() }, m_client.get()));

	_context->appEngine()->rootContext()->setContextProperty("clientModel", this);
}

ClientModel::~ClientModel()
{
}

QString ClientModel::apiCall(QString const& _message)
{
	m_rpcConnector->OnRequest(_message.toStdString(), nullptr);
	return m_rpcConnector->response();
}

void ClientModel::mine()
{
	m_client->mine();
	newBlock();
}

QString ClientModel::contractAddress() const
{
	return QString::fromStdString(dev::toJS(m_contractAddress));
}

void ClientModel::debugDeployment()
{
	executeSequence(std::vector<TransactionSettings>(), 10000000 * ether);
}

void ClientModel::debugState(QVariantMap _state)
{
	u256 balance = (qvariant_cast<QEther*>(_state.value("balance")))->toU256Wei();
	QVariantList transactions = _state.value("transactions").toList();

	std::vector<TransactionSettings> transactionSequence;
	for (auto const& t: transactions)
	{
		QVariantMap transaction = t.toMap();
		QString functionId = transaction.value("functionId").toString();
		u256 gas = (qvariant_cast<QEther*>(transaction.value("gas")))->toU256Wei();
		u256 value = (qvariant_cast<QEther*>(transaction.value("value")))->toU256Wei();
		u256 gasPrice = (qvariant_cast<QEther*>(transaction.value("gasPrice")))->toU256Wei();

		bool isStdContract = (transaction.value("stdContract").toBool());
		if (isStdContract)
		{
			TransactionSettings transactionSettings(transaction.value("url").toString());
			transactionSequence.push_back(transactionSettings);
		}
		else
		{
			QVariantMap params = transaction.value("parameters").toMap();
			TransactionSettings transactionSettings(functionId, value, gas, gasPrice);

			for (auto p = params.cbegin(); p != params.cend(); ++p)
			{
				QBigInt* param = qvariant_cast<QBigInt*>(p.value());
				transactionSettings.parameterValues.insert(std::make_pair(p.key(), boost::get<dev::u256>(param->internalValue())));
			}

			if (transaction.value("executeConstructor").toBool())
				transactionSettings.functionId.clear();

			transactionSequence.push_back(transactionSettings);
		}
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
	emit runStateChanged();

	//run sequence
	QtConcurrent::run([=]()
	{
		try
		{
			bytes contractCode = compilerRes->bytes();
			QFunctionDefinition* f = nullptr;
			ContractCallDataEncoder c;
			m_client->resetState(_balance);
			for (auto const& t: _sequence)
			{
				Address address = m_contractAddress;
				if (!t.stdContractUrl.isEmpty())
				{
					//std contract
					dev::bytes const& stdContractCode = m_context->codeModel()->getStdContractCode(t.stdContractUrl);
					Address address = deployContract(stdContractCode, t);
					m_stdContractAddresses[t.functionId] = address;
				}
				else
				{
					//encode data
					f = nullptr;
					if (t.functionId.isEmpty())
						f = contractDef->constructor();
					else
					{
						for (int tf = 0; tf < contractDef->functionsList().size(); tf++)
						{
							if (contractDef->functionsList().at(tf)->name() == t.functionId)
							{
								f = contractDef->functionsList().at(tf);
								break;
							}
						}
					}
					if (!f)
						throw std::runtime_error("function " + t.functionId.toStdString() + " not found");

					c.encode(f);
					for (int p = 0; p < f->parametersList().size(); p++)
					{
						QVariableDeclaration* var = f->parametersList().at(p);
						u256 value = 0;
						auto v = t.parameterValues.find(var->name());
						if (v != t.parameterValues.cend())
							value = v->second;
						c.encode(var, value);
					}

					if (t.functionId.isEmpty())
					{
						Address newAddress = deployContract(contractCode, t);
						if (newAddress != m_contractAddress)
						{
							m_contractAddress = newAddress;
							contractAddressChanged();
						}
					}
					else
						callContract(address, c.encodedData(), t);
				}
			}
			m_running = false;
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
		emit runStateChanged();
	});
}

void ClientModel::showDebugger()
{
	//we need to wrap states in a QObject before sending to QML.
	QList<QObject*> wStates;
	auto const& lastResult = m_client->record().back().transactions.back();
	for (unsigned i = 0; i < lastResult.machineStates.size(); i++)
	{
		QPointer<DebuggingStateWrapper> s(new DebuggingStateWrapper(lastResult.executionCode, lastResult.executionData.toBytes()));
		s->setState(lastResult.machineStates[i]);
		wStates.append(s);
	}

	QList<QVariableDefinition*> returnParameters;
	//returnParameters = c.decode(f->returnParameters(), debuggingContent.returnValue);

	//collect states for last transaction
	AssemblyDebuggerData code = DebuggingStateWrapper::getHumanReadableCode(lastResult.executionCode);
	m_context->appEngine()->rootContext()->setContextProperty("debugStates", QVariant::fromValue(wStates));
	m_context->appEngine()->rootContext()->setContextProperty("humanReadableExecutionCode", QVariant::fromValue(std::get<0>(code)));
	m_context->appEngine()->rootContext()->setContextProperty("bytesCodeMapping", QVariant::fromValue(std::get<1>(code)));
	m_context->appEngine()->rootContext()->setContextProperty("contractCallReturnParameters", QVariant::fromValue(new QVariableDefinitionList(returnParameters)));
	showDebuggerWindow();
}

void ClientModel::showDebugError(QString const& _error)
{
	//TODO: change that to a signal
	m_context->displayMessageDialog(tr("Debugger"), _error);
}

Address ClientModel::deployContract(bytes const& _code, TransactionSettings const& _ctrTransaction)
{
	Address newAddress;
	if (!_ctrTransaction.isEmpty())
		newAddress = m_client->transact(m_client->userAccount().secret(), _ctrTransaction.value, _code, _ctrTransaction.gas, _ctrTransaction.gasPrice);
	else
	{
		u256 gasPrice = 10000000000000;
		u256 gas = 125000;
		u256 amount = 100;
		newAddress = m_client->transact(m_client->userAccount().secret(), amount, _code, gas, gasPrice);
	}
	return newAddress;
}

void ClientModel::callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr)
{
	m_client->transact(m_client->userAccount().secret(), _tr.value, _contract, _data, _tr.gas, _tr.gasPrice);
}

void ClientModel::onStateReset()
{
	emit stateCleared();
}

void ClientModel::onNewTransaction()
{
	unsigned block = m_client->number();
	unsigned index =  m_client->record().back().transactions.size() - 1;
	ExecutionResult const& tr = m_client->record().back().transactions.back();
	QString address = QString::fromStdString(toJS(tr.address));
	QString value =  QString::fromStdString(dev::toString(tr.value));
	QString contract = address;
	QString function;
	QString returned;
	if (tr.contractAddress)
		returned = QString::fromStdString(toJS(tr.contractAddress));
	else
		returned = QString::fromStdString(toJS(tr.returnValue));

	FixedHash<4> functionHash;
	if (tr.transactionData.size() >= 4)
		functionHash = FixedHash<4>(tr.transactionData);

	if (tr.address == m_contractAddress || tr.contractAddress == m_contractAddress)
	{
		auto compilerRes = m_context->codeModel()->code();
		QContractDefinition* def = compilerRes->contract();
		contract = def->name();
		QFunctionDefinition* funcDef = def->getFunction(functionHash);
		if (funcDef)
			function = funcDef->name();
	}
	else
		function = QString::fromStdString(toJS(functionHash));

	TransactionLogEntry* log = new TransactionLogEntry(block, index, contract, function, value, address, returned);
	QQmlEngine::setObjectOwnership(log, QQmlEngine::JavaScriptOwnership);
	emit newTransaction(log);
}

}
}
