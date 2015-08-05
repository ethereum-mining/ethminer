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
/** @file AllAccounts.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#pragma once

#include <QMutex>
#include <QString>
#include <QPair>
#include <QList>
#include "MainFace.h"

namespace Ui
{
class LogPanel;
}

namespace dev
{
namespace az
{

class LogPanel: public QObject, public Plugin
{
	Q_OBJECT

public:
	LogPanel(MainFace* _m);
	~LogPanel();

private slots:
	void on_verbosity_valueChanged();

private:
	void timerEvent(QTimerEvent*) override;
	void readSettings(QSettings const&) override;
	void writeSettings(QSettings&) override;

	Ui::LogPanel* m_ui;

	QMutex m_logLock;
	QString m_logHistory;
	bool m_logChanged = true;
};

}
}
