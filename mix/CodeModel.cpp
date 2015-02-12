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
#include <libsolidity/InterfaceHandler.h>
#include <libevmcore/Instruction.h>
#include <libethcore/CommonJS.h>
#include "QContractDefinition.h"
#include "QFunctionDefinition.h"
#include "QVariableDeclaration.h"
#include "CodeHighlighter.h"
#include "FileIo.h"
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
	m_contractInterface("[]"),
	m_codeHighlighter(new CodeHighlighter())
{}

CompilationResult::CompilationResult(const dev::solidity::CompilerStack& _compiler):
	QObject(nullptr),
	m_successful(true),
	m_codeHash(qHash(QString()))
{
	if (!_compiler.getContractNames().empty())
	{
		auto const& contractDefinition = _compiler.getContractDefinition(std::string());
		m_contract.reset(new QContractDefinition(&contractDefinition));
		m_bytes = _compiler.getBytecode();
		dev::solidity::InterfaceHandler interfaceHandler;
		m_contractInterface = QString::fromStdString(*interfaceHandler.getABIInterface(contractDefinition));
		if (m_contractInterface.isEmpty())
			m_contractInterface = "[]";
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
	m_contractInterface(_prev.m_contractInterface),
	m_codeHighlighter(_prev.m_codeHighlighter)
{}

QString CompilationResult::codeHex() const
{
	return QString::fromStdString(toJS(m_bytes));
}

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

	solidity::CompilerStack cs(true);
	std::unique_ptr<CompilationResult> result;

	std::string source = _code.toStdString();
	// run syntax highlighting first
	// @todo combine this with compilation step
	auto codeHighlighter = std::make_shared<CodeHighlighter>();
	codeHighlighter->processSource(source);

	cs.addSource("configUser", R"(contract configUser{function configAddr()constant returns(address a){ return 0xf025d81196b72fba60a1d4dddad12eeb8360d828;}})");

	// run compilation
	try
	{
		cs.addSource("", source);
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

void CodeModel::onCompilationComplete(CompilationResult* _newResult)
{
	m_compiling = false;
	bool contractChanged = m_result->contractInterface() != _newResult->contractInterface();
	m_result.reset(_newResult);
	emit compilationComplete();
	emit stateChanged();
	if (m_result->successful())
	{
		emit codeChanged();
		if (contractChanged)
			emit contractInterfaceChanged();
	}
}

bool CodeModel::hasContract() const
{
	return m_result->successful();
}

void CodeModel::updateFormatting(QTextDocument* _document)
{
	m_result->codeHighlighter()->updateFormatting(_document, *m_codeHighlighterSettings);
}

dev::bytes const& CodeModel::getStdContractCode(const QString& _contractName, const QString& _url)
{
	auto cached = m_compiledContracts.find(_contractName);
	if (cached != m_compiledContracts.end())
		return cached->second;

	FileIo fileIo;
	std::string source = fileIo.readFile(_url).toStdString();
	solidity::CompilerStack cs(false);
	cs.setSource(source);
	cs.compile(false);
	for (std::string const& name: cs.getContractNames())
	{
		dev::bytes code = cs.getBytecode(name);
		m_compiledContracts.insert(std::make_pair(QString::fromStdString(name), std::move(code)));
	}
	return m_compiledContracts.at(_contractName);
}

