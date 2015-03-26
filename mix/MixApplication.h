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
/** @file MixApplication.h
 * @author Yann yann@ethdev.com
 * @date 2014
 * This class will be use instead of QApplication to launch the application. the method 'notify' allows to catch all exceptions.
 * Not use for now: TODO.
 */

#pragma once

#include <memory>
#include <QApplication>

class QQmlApplicationEngine;

namespace dev
{
namespace mix
{

class ApplicationService: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int systemPointSize READ systemPointSize CONSTANT)

public:
	ApplicationService();
	int systemPointSize() const { return m_systemPointSize; }
public slots:
	bool waitForSignal(QObject* _item, QString _signalName, int _timeout);

private:
	int m_systemPointSize = 0;
};


class MixApplication: public QApplication
{
	Q_OBJECT

public:
	MixApplication(int& _argc, char* _argv[]);
	static void initialize();
	virtual ~MixApplication();
	QQmlApplicationEngine* engine() { return m_engine.get(); }

private:
	std::unique_ptr<QQmlApplicationEngine> m_engine;
};

}
}
