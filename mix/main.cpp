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
/** @file main.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Ethereum IDE client.
 */

#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQuickItem>
#include "CodeEditorExtensionManager.h"
#include "AppContext.h"
#include "MixApplication.h"
using namespace dev::mix;

int main(int _argc, char *_argv[])
{
	QApplication app(_argc, _argv);
	qmlRegisterType<CodeEditorExtensionManager>("CodeEditorExtensionManager", 1, 0, "CodeEditorExtensionManager");
	QQmlApplicationEngine* engine = new QQmlApplicationEngine();
	AppContext::setApplicationContext(engine);
	QObject::connect(&app, SIGNAL(lastWindowClosed()), AppContext::getInstance(), SLOT(quitApplication())); //use to kill ApplicationContext and other stuff
	QObject::connect(engine, SIGNAL(objectCreated(QObject*, QUrl)), AppContext::getInstance(), SLOT(resourceLoaded(QObject*, QUrl)));
	engine->load(QUrl("qrc:/qml/main.qml"));
	return app.exec();
}
