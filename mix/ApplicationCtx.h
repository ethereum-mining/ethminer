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
/** @file ApplicationCtx.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * Provide an access to the current QQmlApplicationEngine which is used to add QML file on the fly.
 * In the future this class can be extended to add more variable related to the context of the application.
 */

#pragma once

#include <QQmlApplicationEngine>

namespace dev
{

namespace mix
{

class ApplicationCtx: public QObject
{
	Q_OBJECT

public:
	ApplicationCtx(QQmlApplicationEngine* _engine) { m_applicationEngine = _engine; }
	~ApplicationCtx() { delete m_applicationEngine; }
	static ApplicationCtx* getInstance() { return Instance; }
	static void setApplicationContext(QQmlApplicationEngine* _engine);
	QQmlApplicationEngine* appEngine();

private:
	static ApplicationCtx* Instance;
	QQmlApplicationEngine* m_applicationEngine;

public slots:
	void quitApplication() { delete Instance; }
};

}

}
