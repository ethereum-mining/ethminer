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
/** @file LogPanel.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "LogPanel.h"
#include <sstream>
#include <QClipboard>
#include <QSettings>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libevmcore/Instruction.h>
#include <libethereum/Client.h>
#include "ui_LogPanel.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

static QString filterOutTerminal(QString _s)
{
	return _s.replace(QRegExp("\x1b\\[(\\d;)?\\d+m"), "");
}

LogPanel::LogPanel(MainFace* _m):
	Plugin(_m, "LogPanel"),
	m_ui(new Ui::LogPanel)
{
	dock(Qt::RightDockWidgetArea, "Log")->setWidget(new QWidget);
	m_ui->setupUi(dock()->widget());
	connect(m_ui->verbosity, SIGNAL(valueChanged(int)), SLOT(on_verbosity_valueChanged()));

	g_logPost = [=](string const& s, char const* c)
	{
		simpleDebugOut(s, c);
		m_logLock.lock();
		m_logHistory.append(filterOutTerminal(QString::fromStdString(s)) + "\n");
		m_logChanged = true;
		m_logLock.unlock();
	};
	startTimer(100);

	on_verbosity_valueChanged();
}

LogPanel::~LogPanel()
{
	// Must do this here since otherwise m_ethereum'll be deleted (and therefore clearWatches() called by the destructor)
	// *after* the client is dead.
	g_logPost = simpleDebugOut;
}

void LogPanel::readSettings(QSettings const& _s)
{
	m_ui->verbosity->setValue(_s.value("verbosity", 1).toInt());
}

void LogPanel::writeSettings(QSettings& _s)
{
	_s.setValue("verbosity", m_ui->verbosity->value());
}

void LogPanel::timerEvent(QTimerEvent*)
{
	if (m_logChanged)
	{
		m_logLock.lock();
		m_logChanged = false;
		m_ui->log->appendPlainText(m_logHistory.mid(0, m_logHistory.length() - 1));
		m_logHistory.clear();
		m_logLock.unlock();
	}
}

void LogPanel::on_verbosity_valueChanged()
{
	g_logVerbosity = m_ui->verbosity->value();
	m_ui->verbosityLabel->setText(QString::number(g_logVerbosity));
}

