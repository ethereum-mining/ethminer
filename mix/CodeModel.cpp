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
/** @file CodeModel.cpp
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <sstream>
#include <QDebug>
#include <QApplication>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libevmcore/Instruction.h>
#include "QContractDefinition.h"
#include "CodeModel.h"

namespace dev
{
namespace mix
{

void BackgroundWorker::queueCodeChange(int _jobId, QString const& _content)
{
	m_model->runCompilationJob(_jobId, _content);
}

CompilationResult::CompilationResult(const solidity::CompilerStack& _compiler, QObject *_parent):
	QObject(_parent), m_successfull(true),
	m_contract(new QContractDefinition(&_compiler.getContractDefinition(std::string()))),
	m_bytes(_compiler.getBytecode()),
	m_assemblyCode(QString::fromStdString((dev::eth::disassemble(m_bytes))))
{}

CompilationResult::CompilationResult(CompilationResult const& _prev, QString const& _compilerMessage, QObject* _parent):
	QObject(_parent), m_successfull(false),
	m_contract(_prev.m_contract),
	m_compilerMessage(_compilerMessage),
	m_bytes(_prev.m_bytes),
	m_assemblyCode(_prev.m_assemblyCode)
{}

CodeModel::CodeModel(QObject* _parent) : QObject(_parent),
	m_backgroundWorker(this), m_backgroundJobId(0)
{
	m_backgroundWorker.moveToThread(&m_backgroundThread);

	//connect(this, &CodeModel::compilationComplete, this, &CodeModel::onCompilationComplete, Qt::QueuedConnection);
	connect(this, &CodeModel::scheduleCompilationJob, &m_backgroundWorker, &BackgroundWorker::queueCodeChange, Qt::QueuedConnection);
	m_backgroundThread.start();
}

CodeModel::~CodeModel()
{
	stop();
	disconnect(this);
}

void CodeModel::stop()
{
	///@todo: cancel bg job
	m_backgroundThread.exit();
	m_backgroundThread.wait();
}

void CodeModel::registerCodeChange(const QString &_code)
{
	// launch the background thread
	m_backgroundJobId++;
	emit scheduleCompilationJob(m_backgroundJobId, _code);
}


void CodeModel::runCompilationJob(int _jobId, QString const& _code)
{
	if (_jobId != m_backgroundJobId)
		return; //obsolete job

	solidity::CompilerStack cs;
	try
	{
		cs.setSource(_code.toStdString());
		cs.compile(false);
		std::shared_ptr<CompilationResult> result(new CompilationResult(cs, nullptr));
		m_result.swap(result);
		qDebug() << QString(QApplication::tr("compilation succeeded"));
	}
	catch (dev::Exception const& _exception)
	{
		std::ostringstream error;
		solidity::SourceReferenceFormatter::printExceptionInformation(error, _exception, "Error", cs);
		std::shared_ptr<CompilationResult> result(new CompilationResult(*m_result, QString::fromStdString(error.str()), nullptr));
		m_result.swap(result);
		qDebug() << QString(QApplication::tr("compilation failed") + " " + m_result->compilerMessage());
	}
	emit compilationComplete();
}

}
}
