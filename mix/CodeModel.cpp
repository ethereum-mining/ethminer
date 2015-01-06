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
#include <QtQml>
#include <libsolidity/CompilerStack.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libevmcore/Instruction.h>
#include "QContractDefinition.h"
#include "QFunctionDefinition.h"
#include "QVariableDeclaration.h"
#include "CodeHighlighter.h"
#include "CodeModel.h"

using namespace dev::mix;

void BackgroundWorker::queueCodeChange(int _jobId, QString const& _content)
{
	m_model->runCompilationJob(_jobId, _content);
}

CompilationResult::CompilationResult():
	QObject(nullptr),
	m_successful(false),
	m_codeHash(qHash(QString())),
	m_contract(new QContractDefinition()),
	m_codeHighlighter(new CodeHighlighter())
{}

CompilationResult::CompilationResult(const solidity::CompilerStack& _compiler):
	QObject(nullptr),
	m_successful(true),
	m_codeHash(qHash(QString()))
{
	if (!_compiler.getContractNames().empty())
	{
		m_contract.reset(new QContractDefinition(&_compiler.getContractDefinition(std::string())));
		m_bytes = _compiler.getBytecode();
		m_assemblyCode = QString::fromStdString(dev::eth::disassemble(m_bytes));
	}
	else
		m_contract.reset(new QContractDefinition());
}

CompilationResult::CompilationResult(CompilationResult const& _prev, QString const& _compilerMessage):
	QObject(nullptr),
	m_successful(false),
	m_codeHash(qHash(QString())),
	m_contract(_prev.m_contract),
	m_compilerMessage(_compilerMessage),
	m_bytes(_prev.m_bytes),
	m_assemblyCode(_prev.m_assemblyCode),
	m_codeHighlighter(_prev.m_codeHighlighter)
{}

CodeModel::CodeModel(QObject* _parent):
	QObject(_parent),
	m_compiling(false),
	m_result(new CompilationResult()),
	m_codeHighlighterSettings(new CodeHighlighterSettings()),
	m_backgroundWorker(this),
	m_backgroundJobId(0)
{
	m_backgroundWorker.moveToThread(&m_backgroundThread);
	connect(this, &CodeModel::scheduleCompilationJob, &m_backgroundWorker, &BackgroundWorker::queueCodeChange, Qt::QueuedConnection);
	connect(this, &CodeModel::compilationCompleteInternal, this, &CodeModel::onCompilationComplete, Qt::QueuedConnection);
	qRegisterMetaType<CompilationResult*>("CompilationResult*");
	qRegisterMetaType<QContractDefinition*>("QContractDefinition*");
	qRegisterMetaType<QFunctionDefinition*>("QFunctionDefinition*");
	qRegisterMetaType<QVariableDeclaration*>("QVariableDeclaration*");
	qmlRegisterType<QFunctionDefinition>("org.ethereum.qml", 1, 0, "QFunctionDefinition");
	qmlRegisterType<QVariableDeclaration>("org.ethereum.qml", 1, 0, "QVariableDeclaration");
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

void CodeModel::registerCodeChange(QString const& _code)
{
	// launch the background thread
	uint hash = qHash(_code);
	if (m_result->m_codeHash == hash)
		return;
	m_backgroundJobId++;
	m_compiling = true;
	emit stateChanged();
	emit scheduleCompilationJob(m_backgroundJobId, _code);
}

void CodeModel::runCompilationJob(int _jobId, QString const& _code)
{
	if (_jobId != m_backgroundJobId)
		return; //obsolete job

	solidity::CompilerStack cs;
	std::unique_ptr<CompilationResult> result;

	std::string source = _code.toStdString();
	// run syntax highlighting first
	// @todo combine this with compilation step
	auto codeHighlighter = std::make_shared<CodeHighlighter>();
	codeHighlighter->processSource(source);

	// run compilation
	try
	{
		cs.setSource(source);
		cs.compile(false);
		codeHighlighter->processAST(cs.getAST());
		result.reset(new CompilationResult(cs));
		qDebug() << QString(QApplication::tr("compilation succeeded"));
	}
	catch (dev::Exception const& _exception)
	{
		std::ostringstream error;
		solidity::SourceReferenceFormatter::printExceptionInformation(error, _exception, "Error", cs);
		result.reset(new CompilationResult(*m_result, QString::fromStdString(error.str())));
		codeHighlighter->processError(_exception);
		qDebug() << QString(QApplication::tr("compilation failed:") + " " + result->compilerMessage());
	}
	result->m_codeHighlighter = codeHighlighter;
	result->m_codeHash = qHash(_code);

	emit compilationCompleteInternal(result.release());
}

void CodeModel::onCompilationComplete(CompilationResult*_newResult)
{
	m_compiling = false;
	m_result.reset(_newResult);
	emit compilationComplete();
	emit stateChanged();
	if (m_result->successfull())
		emit codeChanged();
}

bool CodeModel::hasContract() const
{
	return m_result->contract()->functionsList().size() > 0;
}

void CodeModel::updateFormatting(QTextDocument* _document)
{
	m_result->codeHighlighter()->updateFormatting(_document, *m_codeHighlighterSettings);
}
