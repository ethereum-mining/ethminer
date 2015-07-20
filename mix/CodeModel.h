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
/** @file CodeModel.h
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#pragma once

#include <memory>
#include <atomic>
#include <map>
#include <QObject>
#include <QThread>
#include <QHash>
#include <QMetaEnum>
#include <libdevcore/Common.h>
#include <libdevcore/Guards.h>
#include <libevmcore/Params.h>
#include <libevmasm/Assembly.h>
#include <libdevcore/SHA3.h>
#include "SolidityType.h"
#include "QBigInt.h"

class QTextDocument;

namespace dev
{

namespace solidity
{
class CompilerStack;
class Type;
}

namespace mix
{

class CodeModel;
class CodeHighlighter;
class CodeHighlighterSettings;
class QContractDefinition;

//utility class to perform tasks in background thread
class BackgroundWorker: public QObject
{
	Q_OBJECT

public:
	BackgroundWorker(CodeModel* _model): QObject(), m_model(_model) {}

public slots:
	void queueCodeChange(int _jobId);
private:
	CodeModel* m_model;
};

using LocationPair = QPair<int, int>;

///Compilation result model. Contains all the compiled contract data required by UI
class CompiledContract: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QContractDefinition* contract READ contract)
	Q_PROPERTY(QString contractInterface READ contractInterface CONSTANT)
	Q_PROPERTY(QString codeHex READ codeHex CONSTANT)
	Q_PROPERTY(QString documentId READ documentId CONSTANT)

public:
	/// Successful compilation result constructor
	CompiledContract(solidity::CompilerStack const& _compiler, QString const& _contractName, QString const& _source);

	/// @returns contract definition for QML property
	QContractDefinition* contract() const { return m_contract.get(); }
	/// @returns contract definition
	std::shared_ptr<QContractDefinition> sharedContract() const { return m_contract; }
	/// @returns contract bytecode
	dev::bytes const& bytes() const { return m_bytes; }
	/// @returns contract bytecode as hex string
	QString codeHex() const;
	/// @returns contract definition in JSON format
	QString contractInterface() const { return m_contractInterface; }
	/// @return assebly item locations
	eth::AssemblyItems const& assemblyItems() const { return m_assemblyItems; }
	eth::AssemblyItems const& constructorAssemblyItems() const { return m_constructorAssemblyItems; }
	/// @returns contract source Id
	QString documentId() const { return m_documentId; }

	QHash<LocationPair, SolidityDeclaration> const& locals() const { return m_locals; }
	QHash<unsigned, SolidityDeclarations> const& storage() const { return m_storage; }

private:
	uint m_sourceHash;
	std::shared_ptr<QContractDefinition> m_contract;
	QString m_compilerMessage; ///< @todo: use some structure here
	dev::bytes m_bytes;
	QString m_contractInterface;
	QString m_documentId;
	eth::AssemblyItems m_assemblyItems;
	eth::AssemblyItems m_constructorAssemblyItems;
	QHash<LocationPair, SolidityDeclaration> m_locals;
	QHash<unsigned, SolidityDeclarations> m_storage;

	friend class CodeModel;
};

using ContractMap = QMap<QString, CompiledContract*>; //needs to be sorted

/// Source map
using LocationMap = QHash<LocationPair, QString>;

struct SourceMap
{
	LocationMap contracts;
	LocationMap functions;
};

using SourceMaps = QMap<QString, SourceMap>; //by source id
using GasCostsMaps = QMap<QString, QVariantList>; //gas cost by contract name

class GasMap: public QObject
{
	Q_OBJECT
	Q_ENUMS(type)
	Q_PROPERTY(int start MEMBER m_start CONSTANT)
	Q_PROPERTY(int end MEMBER m_end CONSTANT)
	Q_PROPERTY(QString gas MEMBER m_gas CONSTANT)
	Q_PROPERTY(bool isInfinite MEMBER m_isInfinite CONSTANT)
	Q_PROPERTY(QString codeBlockType READ codeBlockType CONSTANT)
	Q_PROPERTY(QString contractName MEMBER m_contractName CONSTANT)
	Q_PROPERTY(QString functionName MEMBER m_functionName CONSTANT)

public:

	enum type
	{
		Statement,
		Function,
		Constructor
	};

	GasMap(int _start, int _end, QString _gas, bool _isInfinite, type _type, QString _contractName, QString _functionName, QObject* _parent): QObject(_parent),
		m_start(_start), m_end(_end), m_gas(_gas), m_isInfinite(_isInfinite), m_type(_type), m_contractName(_contractName), m_functionName(_functionName) {}
	QString contractName() { return m_contractName; }
	QString functionName() { return m_functionName; }

private:
	int m_start;
	int m_end;
	QString m_gas;
	bool m_isInfinite;
	type m_type;
	QString m_contractName;
	QString m_functionName;

	QString codeBlockType() const
	{
		QMetaEnum units = staticMetaObject.enumerator(staticMetaObject.indexOfEnumerator("type"));
		if (m_type)
		{
			const char* key = units.valueToKey(m_type);
			return QString(key).toLower();
		}
		return QString("");
	}
};

class GasMapWrapper: public QObject
{
	Q_OBJECT

	Q_PROPERTY(GasCostsMaps gasMaps MEMBER m_gasMaps CONSTANT)

public:
	GasMapWrapper(QObject* _parent = nullptr): QObject(_parent){}
	void push(QString _source, int _start, int _end, QString _value, bool _isInfinite, GasMap::type _type, QString _contractName = "", QString _functionName = "");
	bool contains(QString _key);
	void insert(QString _source, QVariantList _variantList);
	QVariantList gasCostsByDocId(QString _source);
	QVariantList gasCostsBy(QString _contractName, QString _functionName = "");

private:
	GasCostsMaps m_gasMaps;
};

/// Code compilation model. Compiles contracts in background an provides compiled contract data
class CodeModel: public QObject
{
	Q_OBJECT

public:
	CodeModel();
	~CodeModel();

	Q_PROPERTY(QVariantMap contracts READ contracts NOTIFY codeChanged)
	Q_PROPERTY(bool compiling READ isCompiling NOTIFY stateChanged)
	Q_PROPERTY(bool hasContract READ hasContract NOTIFY codeChanged)
	Q_PROPERTY(bool optimizeCode MEMBER m_optimizeCode WRITE setOptimizeCode)
	Q_PROPERTY(int callStipend READ callStipend)
	Q_PROPERTY(int txGas READ txGas)

	/// @returns latest compilation results for contracts
	QVariantMap contracts() const;
	/// @returns compilation status
	bool isCompiling() const { return m_compiling; }
	/// @returns true there is a contract which has at least one function
	bool hasContract() const;
	/// Get contract code by url. Contract is compiled on first access and cached
	dev::bytes const& getStdContractCode(QString const& _contractName, QString const& _url);
	/// Get contract by name
	/// Throws if not found
	CompiledContract const& contract(QString const& _name) const;
	/// Get contract by name
	/// @returns nullptr if not found
	Q_INVOKABLE CompiledContract const* tryGetContract(QString const& _name) const;
	/// Find a contract by document id
	/// @returns CompiledContract object or null if not found
	Q_INVOKABLE CompiledContract* contractByDocumentId(QString const& _documentId) const;
	/// Reset code model
	Q_INVOKABLE void reset() { reset(QVariantMap()); }
	/// Delete a contract source
	Q_INVOKABLE void unregisterContractSrc(QString const& _documentId);
	/// Convert solidity type info to mix type
	static SolidityType nodeType(dev::solidity::Type const* _type);
	/// Retrieve subtype
	static void retrieveSubType(SolidityType& _wrapperType, dev::solidity::Type const* _type);
	/// Check if given location belongs to contract or function
	bool isContractOrFunctionLocation(dev::SourceLocation const& _location);
	/// Get funciton name by location
	QString resolveFunctionName(dev::SourceLocation const& _location);
	/// Gas estimation for compiled sources
	void gasEstimation(solidity::CompilerStack const& _cs);
	/// Gas cost by doc id
	Q_INVOKABLE QVariantList gasCostByDocumentId(QString const& _documentId) const;
	/// Gas cost by @arg contractName @arg functionName
	Q_INVOKABLE QVariantList gasCostBy(QString const& _contractName, QString const& _functionName) const;
	/// Set optimize code
	Q_INVOKABLE void setOptimizeCode(bool _value);
	/// sha3
	Q_INVOKABLE QString sha3(QString _source) { return QString::fromStdString(dev::sha3(_source.toStdString()).hex()); }
	int txGas() { return static_cast<int>(dev::eth::c_txGas); }
	int callStipend() { return static_cast<int>(dev::eth::c_callStipend); }

signals:
	/// Emited on compilation state change
	void stateChanged();
	/// Emitted on compilation complete
	void compilationComplete();
	/// Emitted on compilation error
	void compilationError(QString _error, QVariantMap _firstErrorLoc, QVariantList _secondErrorLoc);
	/// Internal signal used to transfer compilation job to background thread
	void scheduleCompilationJob(int _jobId);
	/// Emitted if there are any changes in the code model
	void codeChanged();
	/// Emitted if there are any changes in the contract interface
	void contractInterfaceChanged(QString _documentId);
	/// Emitted if there is a new contract compiled for the first time
	void newContractCompiled(QString _documentId);
	/// Emitted if a contract name has been changed
	void contractRenamed(QString _documentId, QString _oldName, QString _newName);

public slots:
	/// Update code model on source code change
	void registerCodeChange(QString const& _documentId, QString const& _code);
	/// Reset code model for a new project
	void reset(QVariantMap const& _documents);

private:
	void runCompilationJob(int _jobId);
	void stop();
	void releaseContracts();
	void collectContracts(dev::solidity::CompilerStack const& _cs, std::vector<std::string> const& _sourceNames);
	QVariantMap resolveCompilationErrorLocation(dev::solidity::CompilerStack const& _cs, dev::SourceLocation const& _location);

	std::atomic<bool> m_compiling;
	mutable dev::Mutex x_contractMap;
	ContractMap m_contractMap;
	SourceMaps m_sourceMaps;
	GasMapWrapper* m_gasCostsMaps = 0;
	std::unique_ptr<CodeHighlighterSettings> m_codeHighlighterSettings;
	QThread m_backgroundThread;
	BackgroundWorker m_backgroundWorker;
	int m_backgroundJobId = 0; //protects from starting obsolete compilation job
	std::map<QString, dev::bytes> m_compiledContracts; //by name
	dev::Mutex x_pendingContracts;
	std::map<QString, QString> m_pendingContracts; //name to source
	bool m_optimizeCode = false;
	friend class BackgroundWorker;
};

}

}
