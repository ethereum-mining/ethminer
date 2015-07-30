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

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#include "ClientModel.h"
#include <QtConcurrent/QtConcurrent>
#include <QDebug>
#include <QQmlContext>
#include <QQmlApplicationEngine>
#include <QStandardPaths>
#include <jsonrpccpp/server.h>
#include <libethcore/CommonJS.h>
#include <libethereum/Transaction.h>
#include <libdevcore/FixedHash.h>
#include "DebuggingStateWrapper.h"
#include "Exceptions.h"
#include "QContractDefinition.h"
#include "QVariableDeclaration.h"
#include "QVariableDefinition.h"
#include "ContractCallDataEncoder.h"
#include "CodeModel.h"
#include "QEther.h"
#include "Web3Server.h"
#include "MixClient.h"

using namespace dev;
using namespace dev::eth;
using namespace std;

namespace dev
{
namespace mix
{

class RpcConnector: public jsonrpc::AbstractServerConnector
{
public:
	virtual bool StartListening() override { return true; }
	virtual bool StopListening() override { return true; }
	virtual bool SendResponse(string const& _response, void*) override
	{
		m_response = QString::fromStdString(_response);
		return true;
	}
	QString response() const { return m_response; }

private:
	QString m_response;
};


ClientModel::ClientModel():
	m_running(false), m_rpcConnector(new RpcConnector())
{
	qRegisterMetaType<QBigInt*>("QBigInt*");
	qRegisterMetaType<QVariableDefinition*>("QVariableDefinition*");
	qRegisterMetaType<QList<QVariableDefinition*>>("QList<QVariableDefinition*>");
	qRegisterMetaType<QList<QVariableDeclaration*>>("QList<QVariableDeclaration*>");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qRegisterMetaType<QSolidityType*>("QSolidityType*");
	qRegisterMetaType<QMachineState*>("QMachineState");
	qRegisterMetaType<QInstruction*>("QInstruction");
	qRegisterMetaType<QCode*>("QCode");
	qRegisterMetaType<QCallData*>("QCallData");
	qRegisterMetaType<RecordLogEntry*>("RecordLogEntry*");
}

ClientModel::~ClientModel()
{
	m_runFuture.waitForFinished();
}

void ClientModel::init(QString _dbpath)
{
	m_dbpath = _dbpath;
	if (m_dbpath.isEmpty())
		m_client.reset(new MixClient(QStandardPaths::writableLocation(QStandardPaths::TempLocation).toStdString()));
	else
		m_client.reset(new MixClient(m_dbpath.toStdString()));

	m_ethAccounts = make_shared<FixedAccountHolder>([=](){return m_client.get();}, std::vector<KeyPair>());
	m_web3Server.reset(new Web3Server(*m_rpcConnector.get(), m_ethAccounts, std::vector<KeyPair>(), m_client.get()));
	connect(m_web3Server.get(), &Web3Server::newTransaction, this, &ClientModel::onNewTransaction, Qt::DirectConnection);
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
		cerr << boost::current_exception_diagnostic_information();
		return QString();
	}
}

void ClientModel::mine()
{
	if (m_mining)
		BOOST_THROW_EXCEPTION(ExecutionStateException());
	m_mining = true;
	emit miningStarted();
	emit miningStateChanged();
	m_runFuture = QtConcurrent::run([=]()
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
			cerr << boost::current_exception_diagnostic_information();
			emit runFailed(QString::fromStdString(boost::current_exception_diagnostic_information()));
		}
		emit miningStateChanged();
	});
}

QString ClientModel::newSecret()
{
	KeyPair a = KeyPair::create();
	return QString::fromStdString(dev::toHex(a.secret().ref()));
}

QString ClientModel::address(QString const& _secret)
{
	return QString::fromStdString(dev::toHex(KeyPair(Secret(_secret.toStdString())).address().ref()));
}

QString ClientModel::toHex(QString const& _int)
{
	return QString::fromStdString(dev::toHex(dev::u256(_int.toStdString())));
}

QString ClientModel::encodeAbiString(QString _string)
{
	ContractCallDataEncoder encoder;
	return QString::fromStdString(dev::toHex(encoder.encodeBytes(_string)));
}

QString ClientModel::encodeStringParam(QString const& _param)
{
	ContractCallDataEncoder encoder;
	return QString::fromStdString(dev::toHex(encoder.encodeStringParam(_param, 32)));
}

QStringList ClientModel::encodeParams(QVariant const& _param, QString const& _contract, QString const& _function)
{
	QStringList ret;
	CompiledContract const& compilerRes = m_codeModel->contract(_contract);
	QList<QVariableDeclaration*> paramsList;
	shared_ptr<QContractDefinition> contractDef = compilerRes.sharedContract();
	if (_contract == _function)
		paramsList = contractDef->constructor()->parametersList();
	else
		for (QFunctionDefinition* tf: contractDef->functionsList())
			if (tf->name() == _function)
			{
				paramsList = tf->parametersList();
				break;
			}
	if (paramsList.length() > 0)
		for (QVariableDeclaration* var: paramsList)
		{
			ContractCallDataEncoder encoder;
			QSolidityType const* type = var->type();
			QVariant value = _param.toMap().value(var->name());
			encoder.encode(value, type->type());
			ret.push_back(QString::fromStdString(dev::toHex(encoder.encodedData())));
		}
	return ret;
}

QVariantMap ClientModel::contractAddresses() const
{
	QVariantMap res;
	for (auto const& c: m_contractAddresses)
		res.insert(c.first.first, QString::fromStdString(toJS(c.second)));
	return res;
}

QVariantList ClientModel::gasCosts() const
{
	QVariantList res;
	for (auto const& c: m_gasCosts)
		res.append(QVariant::fromValue(static_cast<int>(c)));
	return res;
}

void ClientModel::addAccount(QString const& _secret)
{
	KeyPair key(Secret(_secret.toStdString()));
	m_accountsSecret.push_back(key);
	Address address = key.address();
	m_accounts[address] = Account(u256(0), Account::NormalCreation);
	m_ethAccounts->setAccounts(m_accountsSecret);
}

QString ClientModel::resolveAddress(QString const& _secret)
{
	KeyPair key(Secret(_secret.toStdString()));
	return "0x" + QString::fromStdString(key.address().hex());
}

void ClientModel::setupScenario(QVariantMap _scenario)
{
	onStateReset();
	WriteGuard(x_queueTransactions);
	m_running = true;

	QVariantList blocks = _scenario.value("blocks").toList();
	QVariantList stateAccounts = _scenario.value("accounts").toList();
	QVariantList stateContracts = _scenario.value("contracts").toList();

	m_accounts.clear();
	m_accountsSecret.clear();
	for (auto const& b: stateAccounts)
	{
		QVariantMap account = b.toMap();
		Address address = {};
		if (account.contains("secret"))
		{
			KeyPair key(Secret(account.value("secret").toString().toStdString()));
			m_accountsSecret.push_back(key);
			address = key.address();
		}
		else if (account.contains("address"))
			address = Address(fromHex(account.value("address").toString().toStdString()));
		if (!address)
			continue;

		m_accounts[address] = Account(qvariant_cast<QEther*>(account.value("balance"))->toU256Wei(), Account::NormalCreation);
	}
	m_ethAccounts->setAccounts(m_accountsSecret);

	for (auto const& c: stateContracts)
	{
		QVariantMap contract = c.toMap();
		Address address = Address(fromHex(contract.value("address").toString().toStdString()));
		Account account(qvariant_cast<QEther*>(contract.value("balance"))->toU256Wei(), Account::ContractConception);
		bytes code = fromHex(contract.value("code").toString().toStdString());
		account.setCode(std::move(code));
		QVariantMap storageMap = contract.value("storage").toMap();
		for(auto s = storageMap.cbegin(); s != storageMap.cend(); ++s)
			account.setStorage(fromBigEndian<u256>(fromHex(s.key().toStdString())), fromBigEndian<u256>(fromHex(s.value().toString().toStdString())));
		m_accounts[address] = account;
	}

	bool trToExecute = false;
	for (auto const& b: blocks)
	{
		QVariantList transactions = b.toMap().value("transactions").toList();
		m_queueTransactions.push_back(transactions);
		trToExecute = transactions.size() > 0;
	}
	m_client->resetState(m_accounts, Secret(_scenario.value("miner").toMap().value("secret").toString().toStdString()));
	if (m_queueTransactions.count() > 0 && trToExecute)
	{
		setupExecutionChain();
		processNextTransactions();
	}
	else
		m_running = false;
}

void ClientModel::setupExecutionChain()
{
	connect(this, &ClientModel::newBlock, this, &ClientModel::processNextTransactions, Qt::QueuedConnection);
	connect(this, &ClientModel::runFailed, this, &ClientModel::stopExecution, Qt::QueuedConnection);
	connect(this, &ClientModel::runStateChanged, this, &ClientModel::finalizeBlock, Qt::QueuedConnection);
}

void ClientModel::stopExecution()
{
	disconnect(this, &ClientModel::newBlock, this, &ClientModel::processNextTransactions);
	disconnect(this, &ClientModel::runStateChanged, this, &ClientModel::finalizeBlock);
	disconnect(this, &ClientModel::runFailed, this, &ClientModel::stopExecution);
	m_running = false;
}

void ClientModel::finalizeBlock()
{
	m_queueTransactions.pop_front();// pop last execution group. The last block is never mined (pending block)
	if (m_queueTransactions.size() > 0)
		mine();
	else
	{
		stopExecution();
		emit runComplete();
	}
}

TransactionSettings ClientModel::transaction(QVariant const& _tr) const
{
	QVariantMap transaction = _tr.toMap();
	QString contractId = transaction.value("contractId").toString();
	QString functionId = transaction.value("functionId").toString();
	bool gasAuto = transaction.value("gasAuto").toBool();
	u256 gas = 0;
	if (transaction.value("gas").data())
		gas = boost::get<u256>(qvariant_cast<QBigInt*>(transaction.value("gas"))->internalValue());
	else
		gasAuto = true;

	u256 value = (qvariant_cast<QEther*>(transaction.value("value")))->toU256Wei();
	u256 gasPrice = (qvariant_cast<QEther*>(transaction.value("gasPrice")))->toU256Wei();
	QString sender = transaction.value("sender").toString();
	bool isContractCreation = transaction.value("isContractCreation").toBool();
	bool isFunctionCall = transaction.value("isFunctionCall").toBool();
	if (contractId.isEmpty() && m_codeModel->hasContract()) //TODO: This is to support old project files, remove later
		contractId = m_codeModel->contracts().keys()[0];
	Secret f = Secret(sender.toStdString());
	TransactionSettings transactionSettings(contractId, functionId, value, gas, gasAuto, gasPrice, f, isContractCreation, isFunctionCall);
	transactionSettings.parameterValues = transaction.value("parameters").toMap();
	if (contractId == functionId || functionId == "Constructor")
		transactionSettings.functionId.clear();
	return transactionSettings;
}

void ClientModel::processNextTransactions()
{
	WriteGuard(x_queueTransactions);
	vector<TransactionSettings> transactionSequence;
	for (auto const& t: m_queueTransactions.front())
	{
		TransactionSettings transactionSettings = transaction(t);
		transactionSequence.push_back(transactionSettings);
	}
	executeSequence(transactionSequence);
}

void ClientModel::executeSequence(vector<TransactionSettings> const& _sequence)
{
	if (m_running)
	{
		qWarning() << "Waiting for current execution to complete";
		m_runFuture.waitForFinished();
	}
	emit runStarted();
	//run sequence
	m_runFuture = QtConcurrent::run([=]()
	{
		try
		{
			m_gasCosts.clear();
			for (TransactionSettings const& transaction: _sequence)
			{
				std::pair<QString, int> ctrInstance = resolvePair(transaction.contractId);
				QString address = resolveToken(ctrInstance);
				if (!transaction.isFunctionCall)
				{
					callAddress(Address(address.toStdString()), bytes(), transaction);
					onNewTransaction();
					continue;
				}
				ContractCallDataEncoder encoder;
				//encode data
				CompiledContract const& compilerRes = m_codeModel->contract(ctrInstance.first);
				QFunctionDefinition const* f = nullptr;
				bytes contractCode = compilerRes.bytes();
				shared_ptr<QContractDefinition> contractDef = compilerRes.sharedContract();
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
					emit runFailed("Function '" + transaction.functionId + tr("' not found. Please check transactions or the contract code."));
				if (!transaction.functionId.isEmpty())
					encoder.encode(f);
				for (QVariableDeclaration const* p: f->parametersList())
				{
					QSolidityType const* type = p->type();
					QVariant value = transaction.parameterValues.value(p->name());
					if (type->type().type == SolidityType::Type::Address)
					{
						if (type->array())
						{
							QJsonArray jsonDoc = QJsonDocument::fromJson(value.toString().toUtf8()).array();
							int k = 0;
							for (QJsonValue const& item: jsonDoc)
							{
								if (item.toString().startsWith("<"))
								{
									std::pair<QString, int> ctrParamInstance = resolvePair(item.toString());
									jsonDoc.replace(k, resolveToken(ctrParamInstance));
								}
								k++;
							}
							QJsonDocument doc(jsonDoc);
							value = QVariant(doc.toJson(QJsonDocument::Compact));
						}
						else if (value.toString().startsWith("<"))
						{
							std::pair<QString, int> ctrParamInstance = resolvePair(value.toString());
							value = QVariant(resolveToken(ctrParamInstance));
						}
					}
					encoder.encode(value, type->type());
				}

				if (transaction.functionId.isEmpty() || transaction.functionId == ctrInstance.first)
				{
					bytes param = encoder.encodedData();
					contractCode.insert(contractCode.end(), param.begin(), param.end());
					Address newAddress = deployContract(contractCode, transaction);
					std::pair<QString, int> contractToken = retrieveToken(transaction.contractId);
					m_contractAddresses[contractToken] = newAddress;
					m_contractNames[newAddress] = contractToken.first;
					contractAddressesChanged();
					gasCostsChanged();
				}
				else
				{
					auto contractAddressIter = m_contractAddresses.find(ctrInstance);
					if (contractAddressIter == m_contractAddresses.end())
						emit runFailed("Contract '" + transaction.contractId + tr(" not deployed.") + "' " + tr(" Cannot call ") + transaction.functionId);
					callAddress(contractAddressIter->second, encoder.encodedData(), transaction);
				}
				m_gasCosts.append(m_client->lastExecution().gasUsed);
				onNewTransaction();
			}
			emit runComplete();
		}
		catch(boost::exception const&)
		{
			cerr << boost::current_exception_diagnostic_information();
			emit runFailed(QString::fromStdString(boost::current_exception_diagnostic_information()));
		}
		catch(exception const& e)
		{
			cerr << boost::current_exception_diagnostic_information();
			emit runFailed(e.what());
		}
		emit runStateChanged();
	});
}

void ClientModel::executeTr(QVariantMap _tr)
{
	WriteGuard(x_queueTransactions);
	QVariantList trs;
	trs.push_back(_tr);
	m_queueTransactions.push_back(trs);
	if (!m_running)
	{
		m_running = true;
		setupExecutionChain();
		processNextTransactions();
	}
}

std::pair<QString, int> ClientModel::resolvePair(QString const& _contractId)
{
	std::pair<QString, int> ret = std::make_pair(_contractId, 0);
	if (_contractId.startsWith("<") && _contractId.endsWith(">"))
	{
		QStringList values = ret.first.remove("<").remove(">").split(" - ");
		ret = std::make_pair(values[0], values[1].toUInt());
	}
	if (_contractId.startsWith("0x"))
		ret = std::make_pair(_contractId, -2);
	return ret;
}

QString ClientModel::resolveToken(std::pair<QString, int> const& _value)
{
	if (_value.second == -2) //-2: first contains a real address
		return _value.first;
	else if (m_contractAddresses.size() > 0)
		return QString::fromStdString("0x" + dev::toHex(m_contractAddresses[_value].ref()));
	else
		return _value.first;
}

std::pair<QString, int> ClientModel::retrieveToken(QString const& _value)
{
	std::pair<QString, int> ret;
	ret.first = _value;
	ret.second = m_contractAddresses.size();
	return ret;
}

void ClientModel::showDebugger()
{
	ExecutionResult last = m_client->lastExecution();
	showDebuggerForTransaction(last);
}

void ClientModel::showDebuggerForTransaction(ExecutionResult const& _t)
{
	//we need to wrap states in a QObject before sending to QML.
	QDebugData* debugData = new QDebugData();
	QQmlEngine::setObjectOwnership(debugData, QQmlEngine::JavaScriptOwnership);
	QList<QCode*> codes;
	QList<QHash<int, int>> codeMaps;
	QList<AssemblyItems> codeItems;
	QList<CompiledContract const*> contracts;
	for (MachineCode const& code: _t.executionCode)
	{
		QHash<int, int> codeMap;
		codes.push_back(QMachineState::getHumanReadableCode(debugData, code.address, code.code, codeMap));
		codeMaps.push_back(move(codeMap));
		//try to resolve contract for source level debugging
		auto nameIter = m_contractNames.find(code.address);
		CompiledContract const* compilerRes = nullptr;
		if (nameIter != m_contractNames.end() && (compilerRes = m_codeModel->tryGetContract(nameIter->second))) //returned object is guaranteed to live till the end of event handler in main thread
		{
			eth::AssemblyItems assemblyItems = !_t.isConstructor() ? compilerRes->assemblyItems() : compilerRes->constructorAssemblyItems();
			codes.back()->setDocument(compilerRes->documentId());
			codeItems.push_back(move(assemblyItems));
			contracts.push_back(compilerRes);
		}
		else
		{
			codeItems.push_back(AssemblyItems());
			contracts.push_back(nullptr);
		}
	}

	QList<QCallData*> data;
	for (bytes const& d: _t.transactionData)
		data.push_back(QMachineState::getDebugCallData(debugData, d));

	QVariantList states;
	QVariantList solCallStack;
	map<int, QVariableDeclaration*> solLocals; //<stack pos, decl>
	map<QString, QVariableDeclaration*> storageDeclarations; //<name, decl>

	unsigned prevInstructionIndex = 0;
	for (MachineState const& s: _t.machineStates)
	{
		int instructionIndex = codeMaps[s.codeIndex][static_cast<unsigned>(s.curPC)];
		QSolState* solState = nullptr;
		if (!codeItems[s.codeIndex].empty() && contracts[s.codeIndex])
		{
			CompiledContract const* contract = contracts[s.codeIndex];
			AssemblyItem const& instruction = codeItems[s.codeIndex][instructionIndex];

			if (instruction.type() == eth::Push && !instruction.data())
			{
				//register new local variable initialization
				auto localIter = contract->locals().find(LocationPair(instruction.getLocation().start, instruction.getLocation().end));
				if (localIter != contract->locals().end())
					solLocals[s.stack.size()] = new QVariableDeclaration(debugData, localIter.value().name.toStdString(), localIter.value().type);
			}

			if (instruction.type() == eth::Tag)
			{
				//track calls into functions
				AssemblyItem const& prevInstruction = codeItems[s.codeIndex][prevInstructionIndex];
				QString functionName = m_codeModel->resolveFunctionName(instruction.getLocation());
				if (!functionName.isEmpty() && ((prevInstruction.getJumpType() == AssemblyItem::JumpType::IntoFunction) || solCallStack.empty()))
					solCallStack.push_front(QVariant::fromValue(functionName));
				else if (prevInstruction.getJumpType() == AssemblyItem::JumpType::OutOfFunction && !solCallStack.empty())
				{
					solCallStack.pop_front();
					solLocals.clear();
				}
			}

			//format solidity context values
			QVariantMap locals;
			QVariantList localDeclarations;
			QVariantMap localValues;
			for (auto l: solLocals)
				if (l.first < (int)s.stack.size())
				{
					if (l.second->type()->name().startsWith("mapping"))
						break; //mapping type not yet managed
					localDeclarations.push_back(QVariant::fromValue(l.second));
					localValues[l.second->name()] = formatValue(l.second->type()->type(), s.stack[l.first]);
				}
			locals["variables"] = localDeclarations;
			locals["values"] = localValues;

			QVariantMap storage;
			QVariantList storageDeclarationList;
			QVariantMap storageValues;
			for (auto st: s.storage)
				if (st.first < numeric_limits<unsigned>::max())
				{
					auto storageIter = contract->storage().find(static_cast<unsigned>(st.first));
					if (storageIter != contract->storage().end())
					{
						QVariableDeclaration* storageDec = nullptr;
						for (SolidityDeclaration const& codeDec : storageIter.value())
						{
							if (codeDec.type.name.startsWith("mapping"))
								continue; //mapping type not yet managed
							auto decIter = storageDeclarations.find(codeDec.name);
							if (decIter != storageDeclarations.end())
								storageDec = decIter->second;
							else
							{
								storageDec = new QVariableDeclaration(debugData, codeDec.name.toStdString(), codeDec.type);
								storageDeclarations[storageDec->name()] = storageDec;
							}
							storageDeclarationList.push_back(QVariant::fromValue(storageDec));
							storageValues[storageDec->name()] = formatStorageValue(storageDec->type()->type(), s.storage, codeDec.offset, codeDec.slot);
						}
					}
				}
			storage["variables"] = storageDeclarationList;
			storage["values"] = storageValues;

			prevInstructionIndex = instructionIndex;

			// filter out locations that match whole function or contract
			SourceLocation location = instruction.getLocation();
			QString source;
			if (location.sourceName)
				source = QString::fromUtf8(location.sourceName->c_str());
			if (m_codeModel->isContractOrFunctionLocation(location))
				location = dev::SourceLocation(-1, -1, location.sourceName);

			solState = new QSolState(debugData, move(storage), move(solCallStack), move(locals), location.start, location.end, source);
		}

		states.append(QVariant::fromValue(new QMachineState(debugData, instructionIndex, s, codes[s.codeIndex], data[s.dataIndex], solState)));
	}

	debugData->setStates(move(states));
	debugDataReady(debugData);
}

QVariant ClientModel::formatValue(SolidityType const& _type, u256 const& _value)
{
	ContractCallDataEncoder decoder;
	bytes val = toBigEndian(_value);
	QVariant res = decoder.decode(_type, val);
	return res;
}

QVariant ClientModel::formatStorageValue(SolidityType const& _type, unordered_map<u256, u256> const& _storage, unsigned _offset, u256 const& _slot)
{
	u256 slot = _slot;
	QVariantList values;
	ContractCallDataEncoder decoder;
	u256 count = 1;
	if (_type.dynamicSize)
	{
		count = _storage.at(slot);
		slot = fromBigEndian<u256>(sha3(toBigEndian(slot)).asBytes());
	}
	else if (_type.array)
		count = _type.count;

	unsigned offset = _offset;
	while (count--)
	{

		auto slotIter = _storage.find(slot);
		u256 slotValue = slotIter != _storage.end() ? slotIter->second : u256();
		bytes slotBytes = toBigEndian(slotValue);
		auto start = slotBytes.end() - _type.size - offset;
		bytes val(32 - _type.size); //prepend with zeroes
		if (_type.type == SolidityType::SignedInteger && (*start & 0x80)) //extend sign
			std::fill(val.begin(), val.end(), 0xff);
		val.insert(val.end(), start, start + _type.size);
		values.append(decoder.decode(_type, val));
		offset += _type.size;
		if ((offset + _type.size) > 32)
		{
			slot++;
			offset = 0;
		}
	}

	if (!_type.array)
		return values[0];

	return QVariant::fromValue(values);
}

void ClientModel::emptyRecord()
{
	debugDataReady(new QDebugData());
}

void ClientModel::debugRecord(unsigned _index)
{
	ExecutionResult e = m_client->execution(_index);
	showDebuggerForTransaction(e);
}

Address ClientModel::deployContract(bytes const& _code, TransactionSettings const& _ctrTransaction)
{
	eth::TransactionSkeleton ts;
	ts.creation = true;
	ts.value = _ctrTransaction.value;
	ts.data = _code;
	ts.gas = _ctrTransaction.gas;
	ts.gasPrice = _ctrTransaction.gasPrice;
	ts.from = toAddress(_ctrTransaction.sender);
	return m_client->submitTransaction(ts, _ctrTransaction.sender, _ctrTransaction.gasAuto).second;
}

void ClientModel::callAddress(Address const& _contract, bytes const& _data, TransactionSettings const& _tr)
{
	eth::TransactionSkeleton ts;
	ts.creation = false;
	ts.value = _tr.value;
	ts.to = _contract;
	ts.data = _data;
	ts.gas = _tr.gas;
	ts.gasPrice = _tr.gasPrice;
	ts.from = toAddress(_tr.sender);
	m_client->submitTransaction(ts, _tr.sender, _tr.gasAuto);
}

RecordLogEntry* ClientModel::lastBlock() const
{
	eth::BlockInfo blockInfo = m_client->blockInfo();
	stringstream strGas;
	strGas << blockInfo.gasUsed();
	stringstream strNumber;
	strNumber << blockInfo.number();
	RecordLogEntry* record =  new RecordLogEntry(0, QString::fromStdString(strNumber.str()), tr(" - Block - "), tr("Hash: ") + QString(QString::fromStdString(dev::toHex(blockInfo.hash().ref()))), QString(), QString(), QString(), false, RecordLogEntry::RecordType::Block, QString::fromStdString(strGas.str()), QString(), tr("Block"), QVariantMap(), QVariantMap(), QVariantList());
	QQmlEngine::setObjectOwnership(record, QQmlEngine::JavaScriptOwnership);
	return record;
}

void ClientModel::onStateReset()
{
	m_contractAddresses.clear();
	m_contractNames.clear();
	m_stdContractAddresses.clear();
	m_stdContractNames.clear();
	m_queueTransactions.clear();
	emit stateCleared();
}

void ClientModel::onNewTransaction()
{
	ExecutionResult const& tr = m_client->lastExecution();
	unsigned block = m_client->number() + 1;
	unsigned recordIndex = tr.executonIndex;
	QString transactionIndex = tr.isCall() ? QObject::tr("Call") : QString("%1:%2").arg(block).arg(tr.transactionIndex);
	QString address = QString::fromStdString(toJS(tr.address));
	QString value = QString::fromStdString(toString(tr.value));
	QString contract = address;
	QString function;
	QString returned;
	QString gasUsed;

	bool creation = (bool)tr.contractAddress;

	if (!tr.isCall())
		gasUsed = QString::fromStdString(toString(tr.gasUsed));

	//TODO: handle value transfer
	FixedHash<4> functionHash;
	bool abi = false;
	if (creation)
	{
		//contract creation
		function = QObject::tr("Constructor");
		address = QObject::tr("(Create contract)");
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

	Address contractAddress = (bool)tr.address ? tr.address : tr.contractAddress;
	auto contractAddressIter = m_contractNames.find(contractAddress);
	QVariantMap inputParameters;
	QVariantMap returnParameters;
	QVariantList logs;
	if (contractAddressIter != m_contractNames.end())
	{
		ContractCallDataEncoder encoder;
		CompiledContract const& compilerRes = m_codeModel->contract(contractAddressIter->second);
		const QContractDefinition* def = compilerRes.contract();
		contract = def->name();
		if (creation)
			function = contract;
		if (abi)
		{
			QFunctionDefinition const* funcDef = def->getFunction(functionHash);
			if (funcDef)
			{
				function = funcDef->name();
				QStringList returnValues = encoder.decode(funcDef->returnParameters(), tr.result.output);
				returned += "(";
				returned += returnValues.join(", ");
				returned += ")";

				QStringList returnParams = encoder.decode(funcDef->returnParameters(), tr.result.output);
				for (int k = 0; k < returnParams.length(); ++k)
					returnParameters.insert(funcDef->returnParameters().at(k)->name(), returnParams.at(k));

				bytes data = tr.inputParameters;
				data.erase(data.begin(), data.begin() + 4);
				QStringList parameters = encoder.decode(funcDef->parametersList(), data);
				for (int k = 0; k < parameters.length(); ++k)
					inputParameters.insert(funcDef->parametersList().at(k)->name(), parameters.at(k));
			}
		}

		// Fill generated logs and decode parameters
		for (auto const& log: tr.logs)
		{
			QVariantMap l;
			l.insert("address",  QString::fromStdString(log.address.hex()));
			std::ostringstream s;
			s << log.data;
			l.insert("data", QString::fromStdString(s.str()));
			std::ostringstream streamTopic;
			streamTopic << log.topics;
			l.insert("topic", QString::fromStdString(streamTopic.str()));
			auto const& sign = log.topics.front(); // first hash supposed to be the event signature. To check
			auto dataIterator = log.data.begin();
			int topicDataIndex = 1;
			for (auto const& event: def->eventsList())
			{
				if (sign == event->fullHash())
				{
					QVariantList paramsList;
					l.insert("name", event->name());
					for (auto const& e: event->parametersList())
					{
						bytes data;
						QString param;
						if (!e->isIndexed())
						{
							data = bytes(dataIterator, dataIterator + 32);
							dataIterator = dataIterator + 32;
						}
						else
						{
							data = log.topics.at(topicDataIndex).asBytes();
							topicDataIndex++;
						}
						param = encoder.decode(e, data);
						QVariantMap p;
						p.insert("indexed", e->isIndexed());
						p.insert("value", param);
						p.insert("name", e->name());
						paramsList.push_back(p);
					}
					l.insert("param", paramsList);
					break;
				}
			}
			logs.push_back(l);
		}
	}

	QString sender;
	for (auto const& secret: m_accountsSecret)
	{
		if (secret.address() == tr.sender)
		{
			sender = QString::fromStdString(dev::toHex(secret.secret().ref()));
			break;
		}
	}
	QString label;
	if (function != QObject::tr("<none>"))
		label = contract + "." + function + "()";
	else
		label = contract;

	if (!creation)
		for (auto const& ctr: m_contractAddresses)
		{
			if (ctr.second == tr.address)
			{
				contract = "<" + ctr.first.first + " - " + QString::number(ctr.first.second) + ">";
				break;
			}
		}

	RecordLogEntry* log = new RecordLogEntry(recordIndex, transactionIndex, contract, function, value, address, returned, tr.isCall(), RecordLogEntry::RecordType::Transaction,
											 gasUsed, sender, label, inputParameters, returnParameters, logs);
	QQmlEngine::setObjectOwnership(log, QQmlEngine::JavaScriptOwnership);
	emit newRecord(log);

	// retrieving all accounts balance
	QVariantMap state;
	QVariantMap accountBalances;
	for (auto const& ctr : m_contractAddresses)
	{
		u256 wei = m_client->balanceAt(ctr.second, PendingBlock);
		accountBalances.insert("0x" + QString::fromStdString(ctr.second.hex()), QEther(wei, QEther::Wei).format());
	}
	for (auto const& account : m_accounts)
	{
		u256 wei = m_client->balanceAt(account.first, PendingBlock);
		accountBalances.insert("0x" + QString::fromStdString(account.first.hex()),  QEther(wei, QEther::Wei).format());
	}
	state.insert("accounts", accountBalances);
	emit newState(recordIndex, state);
}

}
}
