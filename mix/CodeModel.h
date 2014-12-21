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
#include <QObject>
#include <QThread>
#include <libdevcore/Common.h>

namespace dev
{

namespace solidity
{
	class CompilerStack;
}

namespace mix
{

class CodeModel;
class QContractDefinition;

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


class CompilationResult : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QContractDefinition const* contract READ contract)

public:
	/// Successfull compilation result constructor
	CompilationResult(solidity::CompilerStack const& _compiler, QObject* parent);
	/// Failed compilation result constructor
	CompilationResult(CompilationResult const& _prev, QString const& _compilerMessage, QObject* parent);
	QContractDefinition const* contract() { return m_contract.get(); }

	bool successfull() const { return m_successfull; }
	QContractDefinition const* contract() const { return m_contract.get(); }
	QString compilerMessage() const { return m_compilerMessage; }
	dev::bytes const& bytes() const { return m_bytes; }
	QString assemblyCode() const { return m_assemblyCode; }

private:
	bool m_successfull;
	std::shared_ptr<QContractDefinition const> m_contract;
	QString m_compilerMessage; ///< @todo: use some structure here
	dev::bytes m_bytes;
	QString m_assemblyCode;
	///@todo syntax highlighting, etc
};

class CodeModel : public QObject
{
	enum Status
	{
		Idle,
		Compiling,
	};

	Q_OBJECT

public:
	CodeModel(QObject* _parent);
	~CodeModel();

	std::shared_ptr<CompilationResult> lastCompilationResult() { return m_result; }

signals:
	void statusChanged(Status _from, Status _to);
	void compilationComplete();
	void scheduleCompilationJob(int _jobId, QString const& _content);

public slots:
	void registerCodeChange(QString const& _code);

private:
	void runCompilationJob(int _jobId, QString const& _content);
	void stop();

	std::shared_ptr<CompilationResult> m_result;
	QThread m_backgroundThread;
	BackgroundWorker m_backgroundWorker;
	int m_backgroundJobId = 0; //protects from starting obsolete compilation job
	friend class BackgroundWorker;
};

}

}
