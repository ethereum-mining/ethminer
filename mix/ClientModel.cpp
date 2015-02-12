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
#include <QStandardPaths>
#include <jsonrpccpp/server.h>
#include <libethcore/CommonJS.h>
#include <libethereum/Transaction.h>
#include "AppContext.h"
#include "DebuggingStateWrapper.h"
#include "Exceptions.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "QVariableDefinition.h"
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
	qRegisterMetaType<QIntType*>("QIntType*");
	qRegisterMetaType<QStringType*>("QStringType*");
	qRegisterMetaType<QRealType*>("QRealType*");
	qRegisterMetaType<QHashType*>("QHashType*");
	qRegisterMetaType<QEther*>("QEther*");
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QVariableDefinitionList*>("QVariableDefinitionList*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QList<QVariableDeclaration*>>("QList<QVariableDeclaration*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qRegisterMetaType<QMachineState*>("QMachineState");
	qRegisterMetaType<QInstruction*>("QInstruction");
	qRegisterMetaType<QCode*>("QCode");
	qRegisterMetaType<QCallData*>("QCallData");
	qRegisterMetaType<RecordLogEntry*>("RecordLogEntry");

	connect(this, &ClientModel::runComplete, this, &ClientModel::showDebugger, Qt::QueuedConnection);
	m_client.reset(new MixClient(QStandardPaths::writableLocation(QStandardPaths::TempLocation).toStdString()));

	m_web3Server.reset(new Web3Server(*m_rpcConnector.get(), std::vector<dev::KeyPair> { m_client->userAccount() }, m_client.get()));
	connect(m_web3Server.get(), &Web3Server::newTransaction, this, &ClientModel::onNewTransaction, Qt::DirectConnection);
	_context->appEngine()->rootContext()->setContextProperty("clientModel", this);
}

ClientModel::~ClientModel()
{
}

QString ClientModel::apiCall(QString const& _message)
{
	try
	{
		m_rpcConnector->OnRequest(_message.toStdString(), nullptr);
		return m_rpcConnector->response();
	}
	catch (...)
	{
		std::cerr << boost::current_exception_diagnostic_information();
		return QString();
	}
}

void ClientModel::mine()
{
	if (m_running || m_mining)
		BOOST_THROW_EXCEPTION(ExecutionStateException());
	m_mining = true;
	emit miningStarted();
	emit miningStateChanged();
	QtConcurrent::run([=]()
	{
		try
		{
			m_client->mine();
			newBlock();
			m_mining = false;
			emit miningComplete();
		}
		catch (...)
		{
			m_mining = false;
			std::cerr << boost::current_exception_diagnostic_information();
			emit runFailed(QString::fromStdString(boost::current_exception_diagnostic_information()));
		}
		emit miningStateChanged();
	});
}

QString ClientModel::contractAddress() const
{
	return QString::fromStdString(dev::toJS(m_contractAddress));
}

void ClientModel::debugDeployment()
{
	executeSequence(std::vector<TransactionSettings>(), 10000000 * ether);
}

void ClientModel::setupState(QVariantMap _state)
{
	u256 balance = (qvariant_cast<QEther*>(_state.value("balance")))->toU256Wei();
	QVariantList transactions = _state.value("transactions").toList();

	std::vector<TransactionSettings> transactionSequence;
	for (auto const& t: transactions)
	{
		QVariantMap transaction = t.toMap();
		QString functionId = transaction.value("functionId").toString();

		u256 gas = boost::get<u256>(qvariant_cast<QBigInt*>(transaction.value("gas"))->internalValue());
		u256 value = (qvariant_cast<QEther*>(transaction.value("value")))->toU256Wei();
		u256 gasPrice = (qvariant_cast<QEther*>(transaction.value("gasPrice")))->toU256Wei();

		bool isStdContract = (transaction.value("stdContract").toBool());
		if (isStdContract)
		{
			TransactionSettings transactionSettings(functionId, transaction.value("url").toString());
			transactionSettings.gasPrice = 10000000000000;
			transactionSettings.gas = 125000;
			transactionSettings.value = 0;
			transactionSequence.push_back(transactionSettings);
		}
		else
		{
			QVariantList qParams = transaction.value("qType").toList();
			TransactionSettings transactionSettings(functionId, value, gas, gasPrice);

			for (QVariant const& variant: qParams)
			{
				QVariableDefinition* param = qvariant_cast<QVariableDefinition*>(variant);
				transactionSettings.parameterValues.push_back(param);
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
		BOOST_THROW_EXCEPTION(ExecutionStateException());
	CompilationResult* compilerRes = m_context->codeModel()->code();
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
			m_client->resetState(_balance);
			onStateReset();
			for (TransactionSettings const& transaction: _sequence)
			{
				ContractCallDataEncoder encoder;
				QFunctionDefinition const* f = nullptr;
				if (!transaction.stdContractUrl.isEmpty())
				{
					//std contract
					dev::bytes const& stdContractCode = m_context->codeModel()->getStdContractCode(transaction.functionId, transaction.stdContractUrl);
					Address address = deployContract(stdContractCode, transaction);
					m_stdContractAddresses[transaction.functionId] = address;
					m_stdContractNames[address] = transaction.functionId;
				}
				else
				{
					//encode data
					f = nullptr;
					if (transaction.functionId.isEmpty())
						f = contractDef->constructor();
					else
						for (QFunctionDefinition const* tf: contractDef->functionsList())
							if (tf->name() == transaction.functionId)
							{
								f = tf;
								break;
							}
					if (!f)
						BOOST_THROW_EXCEPTION(FunctionNotFoundException() << FunctionName(transaction.functionId.toStdString()));
					if (!transaction.functionId.isEmpty())
						encoder.encode(f);
					for (int p = 0; p < transaction.parameterValues.size(); p++)
					{
						if (f->parametersList().at(p)->type() != transaction.parameterValues.at(p)->declaration()->type())
							BOOST_THROW_EXCEPTION(ParameterChangedException() << FunctionName(f->parametersList().at(p)->type().toStdString()));
						encoder.push(transaction.parameterValues.at(p)->encodeValue());
					}

					if (transaction.functionId.isEmpty())
					{
						bytes param = encoder.encodedData();
						contractCode.insert(contractCode.end(), param.begin(), param.end());
						Address newAddress = deployContract(contractCode, transaction);
						if (newAddress != m_contractAddress)
						{
							m_contractAddress = newAddress;
							contractAddressChanged();
						}
					}
					else
						callContract(m_contractAddress, encoder.encodedData(), transaction);
				}
				onNewTransaction();
			}
			m_running = false;
			emit runComplete();
		}
		catch(boost::exception const&)
		{
			std::cerr << boost::current_exception_diagnostic_information();
			emit runFailed(QString::fromStdString(boost::current_exception_diagnostic_information()));
		}

		catch(std::exception const& e)
		{
			std::cerr << boost::current_exception_diagnostic_information();
			emit runFailed(e.what());
		}
		m_running = false;
		emit runStateChanged();
	});
}

void ClientModel::showDebugger()
{
	ExecutionResult const& last = m_client->lastExecution();
	showDebuggerForTransaction(last);
}

void ClientModel::showDebuggerForTransaction(ExecutionResult const& _t)
{
	//we need to wrap states in a QObject before sending to QML.
	QDebugData* debugData = new QDebugData();
	QQmlEngine::setObjectOwnership(debugData, QQmlEngine::JavaScriptOwnership);
	QList<QCode*> codes;
	for (bytes const& code: _t.executionCode)
		codes.push_back(QMachineState::getHumanReadableCode(debugData, code));

	QList<QCallData*> data;
	for (bytes const& d: _t.transactionData)
		data.push_back(QMachineState::getDebugCallData(debugData, d));

	QVariantList states;
	for (MachineState const& s: _t.machineStates)
		states.append(QVariant::fromValue(new QMachineState(debugData, s, codes[s.codeIndex], data[s.dataIndex])));

	debugData->setStates(std::move(states));

	//QList<QVariableDefinition*> returnParameters;
	//returnParameters = encoder.decode(f->returnParameters(), debuggingContent.returnValue);

	//collect states for last transaction
	debugDataReady(debugData);
}


void ClientModel::debugRecord(unsigned _index)
{
	ExecutionResult const& e = m_client->executions().at(_index);
	showDebuggerForTransaction(e);
}

void ClientModel::showDebugError(QString const& _error)
{
	//TODO: change that to a signal
	m_context->displayMessageDialog(tr("Debugger"), _error);
}

Address ClientModel::deployContract(bytes const& _code, TransactionSettings const& _ctrTransaction)
{
	Address newAddress = m_client->transact(m_client->userAccount().secret(), _ctrTransaction.value, _code, _ctrTransaction.gas, _ctrTransaction.gasPrice);
	return newAddress;
}

void ClientModel::callContract(Address const& _contract, bytes const& _data, TransactionSettings const& _tr)
{
	m_client->transact(m_client->userAccount().secret(), _tr.value, _contract, _data, _tr.gas, _tr.gasPrice);
}

void ClientModel::onStateReset()
{
	m_contractAddress = dev::Address();
	m_stdContractAddresses.clear();
	m_stdContractNames.clear();
	emit stateCleared();
}

void ClientModel::onNewTransaction()
{
	ExecutionResult const& tr = m_client->lastExecution();
	unsigned block = m_client->number() + 1;
	unsigned recordIndex = m_client->executions().size() - 1;
	QString transactionIndex = tr.isCall() ? QObject::tr("Call") : QString("%1:%2").arg(block).arg(tr.transactionIndex);
	QString address = QString::fromStdString(toJS(tr.address));
	QString value =  QString::fromStdString(dev::toString(tr.value));
	QString contract = address;
	QString function;
	QString returned;

	bool creation = tr.contractAddress != 0;

	//TODO: handle value transfer
	FixedHash<4> functionHash;
	bool abi = false;
	if (creation)
	{
		//contract creation
		auto const stdContractName = m_stdContractNames.find(tr.contractAddress);
		if (stdContractName != m_stdContractNames.end())
		{
			function = stdContractName->second;
			contract = function;
		}
		else
			function = QObject::tr("Constructor");
	}
	else
	{
		//transaction/call
		if (tr.transactionData.size() > 0 && tr.transactionData.front().size() >= 4)
		{
			functionHash = FixedHash<4>(tr.transactionData.front().data(), FixedHash<4>::ConstructFromPointer);
			function = QString::fromStdString(toJS(functionHash));
			abi = true;
		}
		else
			function = QObject::tr("<none>");
	}

	if (creation)
		returned = QString::fromStdString(toJS(tr.contractAddress));

	if (m_contractAddress != 0 && (tr.address == m_contractAddress || tr.contractAddress == m_contractAddress))
	{
		auto compilerRes = m_context->codeModel()->code();
		QContractDefinition* def = compilerRes->contract();
		contract = def->name();
		if (abi)
		{
			QFunctionDefinition* funcDef = def->getFunction(functionHash);
			if (funcDef)
			{
				function = funcDef->name();
				ContractCallDataEncoder encoder;
				QList<QVariableDefinition*> returnValues = encoder.decode(funcDef->returnParameters(), tr.returnValue);
				for (auto const& var: returnValues)
					returned += var->value() + " | ";
			}
		}
	}

	RecordLogEntry* log = new RecordLogEntry(recordIndex, transactionIndex, contract, function, value, address, returned, tr.isCall());
	QQmlEngine::setObjectOwnership(log, QQmlEngine::JavaScriptOwnership);
	emit newRecord(log);
}

}
}
