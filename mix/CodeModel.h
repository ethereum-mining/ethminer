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
#include <libdevcore/Common.h>

class QTextDocument;

namespace dev
{

namespace solidity
{
	class CompilerStack;
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
	void queueCodeChange(int _jobId, QString const& _content);
private:
	CodeModel* m_model;
};

///Compilation result model. Contains all the compiled contract data required by UI
class CompilationResult: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QContractDefinition* contract READ contract)
	Q_PROPERTY(QString compilerMessage READ compilerMessage CONSTANT)
	Q_PROPERTY(bool successful READ successful CONSTANT)
	Q_PROPERTY(QString contractInterface READ contractInterface CONSTANT)
	Q_PROPERTY(QString codeHex READ codeHex CONSTANT)

public:
	/// Empty compilation result constructor
	CompilationResult();
	/// Successful compilation result constructor
	CompilationResult(solidity::CompilerStack const& _compiler);
	/// Failed compilation result constructor
	CompilationResult(CompilationResult const& _prev, QString const& _compilerMessage);

	/// @returns contract definition for QML property
	QContractDefinition* contract() { return m_contract.get(); }
	/// @returns contract definition
	std::shared_ptr<QContractDefinition> sharedContract() { return m_contract; }
	/// Indicates if the compilation was successful
	bool successful() const { return m_successful; }
	/// @returns compiler error message in case of unsuccessful compilation
	QString compilerMessage() const { return m_compilerMessage; }
	/// @returns contract bytecode
	dev::bytes const& bytes() const { return m_bytes; }
	/// @returns contract bytecode as hex string
	QString codeHex() const;
	/// @returns contract definition in JSON format
	QString contractInterface() const { return m_contractInterface; }
	/// Get code highlighter
	std::shared_ptr<CodeHighlighter> codeHighlighter() { return m_codeHighlighter; }

private:
	bool m_successful;
	uint m_codeHash;
	std::shared_ptr<QContractDefinition> m_contract;
	QString m_compilerMessage; ///< @todo: use some structure here
	dev::bytes m_bytes;
	QString m_contractInterface;
	std::shared_ptr<CodeHighlighter> m_codeHighlighter;

	friend class CodeModel;
};

/// Background code compiler
class CodeModel: public QObject
{
	Q_OBJECT

public:
	CodeModel(QObject* _parent);
	~CodeModel();

	/// @returns latest compilation result
	CompilationResult* code() { return m_result.get(); }
	/// @returns latest compilation resul
	CompilationResult const* code() const { return m_result.get(); }

	Q_PROPERTY(CompilationResult* code READ code NOTIFY codeChanged)
	Q_PROPERTY(bool compiling READ isCompiling NOTIFY stateChanged)
	Q_PROPERTY(bool hasContract READ hasContract NOTIFY codeChanged)

	/// @returns compilation status
	bool isCompiling() const { return m_compiling; }
	/// @returns true if contract has at least one function
	bool hasContract() const;
	/// Apply text document formatting. @todo Move this to editor module
	void updateFormatting(QTextDocument* _document);
	/// Get contract code by url. Contract is compiled on first access and cached
	dev::bytes const& getStdContractCode(QString const& _contractName, QString const& _url);

signals:
	/// Emited on compilation state change
	void stateChanged();
	/// Emitted on compilation complete
	void compilationComplete();
	/// Internal signal used to transfer compilation job to background thread
	void scheduleCompilationJob(int _jobId, QString const& _content);
	/// Emitted if there are any changes in the code model
	void codeChanged();
	/// Emitted if there are any changes in the contract interface
	void contractInterfaceChanged();
	/// Emitted on compilation complete. Internal
	void compilationCompleteInternal(CompilationResult* _newResult);

private slots:
	void onCompilationComplete(CompilationResult* _newResult);

public slots:
	/// Update code model on source code change
	void registerCodeChange(QString const& _code);

private:
	void runCompilationJob(int _jobId, QString const& _content);
	void stop();

	std::atomic<bool> m_compiling;
	std::unique_ptr<CompilationResult> m_result;
	std::unique_ptr<CodeHighlighterSettings> m_codeHighlighterSettings;
	QThread m_backgroundThread;
	BackgroundWorker m_backgroundWorker;
	int m_backgroundJobId = 0; //protects from starting obsolete compilation job
	std::map<QString, dev::bytes> m_compiledContracts; //by name
	friend class BackgroundWorker;
};

}

}
