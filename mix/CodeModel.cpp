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

#include "CodeModel.h"

namespace dev
{
namespace mix
{

void BackgroundWorker::queueCodeChange(int _jobId, QString const& _content)
{
	m_model->runCompilationJob(_jobId, _content);
}


CodeModel::CodeModel(QObject* _parent) : QObject(_parent), m_backgroundWorker(this)
{
	m_backgroundWorker.moveToThread(&m_backgroundThread);

	connect(this, &CodeModel::compilationComplete, this, &CodeModel::onCompilationComplete, Qt::QueuedConnection);
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

void CodeModel::onCompilationComplete(std::shared_ptr<CompilationResult> _compilationResult)
{
	m_result.swap(_compilationResult);
}


void CodeModel::runCompilationJob(int _jobId, QString const& _content)
{
	if (_jobId != m_backgroundJobId)
		return; //obsolete job


}


}
}
