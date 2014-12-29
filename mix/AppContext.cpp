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
/** @file AppContext.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Provides access to the current QQmlApplicationEngine which is used to add QML file on the fly.
 * In the future this class can be extended to add more variable related to the context of the application.
 * For now AppContext provides reference to:
 * - QQmlApplicationEngine
 * - dev::WebThreeDirect (and dev::eth::Client)
 * - KeyEventManager
 */

#include <QDebug>
#include <QMessageBox>
#include <QQmlComponent>
#include <QQmlApplicationEngine>
#include <libwebthree/WebThree.h>
#include <libdevcrypto/FileSystem.h>
#include <libsolidity/CompilerStack.h>
#include "KeyEventManager.h"
#include "AppContext.h"
using namespace dev;
using namespace dev::eth;
using namespace dev::solidity;
using namespace dev::mix;

AppContext* AppContext::Instance = nullptr;

AppContext::AppContext(QQmlApplicationEngine* _engine)
{
	m_applicationEngine = std::unique_ptr<QQmlApplicationEngine>(_engine);
	m_keyEventManager = std::unique_ptr<KeyEventManager>(new KeyEventManager());
	m_webThree = std::unique_ptr<dev::WebThreeDirect>(new WebThreeDirect(std::string("Mix/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM), getDataDir() + "/Mix", false, {"eth", "shh"}));
	m_compiler = std::unique_ptr<CompilerStack>(new CompilerStack()); //TODO : to move in a codel model structure.
}

QQmlApplicationEngine* AppContext::appEngine()
{
	return m_applicationEngine.get();
}

void AppContext::initKeyEventManager(QObject* _res)
{
	QObject* mainContent = _res->findChild<QObject*>("mainContent", Qt::FindChildrenRecursively);
	if (mainContent)
		QObject::connect(mainContent, SIGNAL(keyPressed(QVariant)), m_keyEventManager.get(), SLOT(keyPressed(QVariant)));
	else
		qDebug() << "Unable to find QObject of mainContent.qml. KeyEvent will not be handled!";
}

KeyEventManager* AppContext::getKeyEventManager()
{
	return m_keyEventManager.get();
}

CompilerStack* AppContext::compiler()
{
	return m_compiler.get();
}

void AppContext::setApplicationContext(QQmlApplicationEngine* _engine)
{
	if (Instance == nullptr)
		Instance = new AppContext(_engine);
}

void AppContext::displayMessageDialog(QString _title, QString _message)
{
	// TODO : move to a UI dedicated layer.
	QObject* dialogWin = m_applicationEngine->rootObjects().at(0)->findChild<QObject*>("alertMessageDialog", Qt::FindChildrenRecursively);
	QObject* dialogWinComponent = m_applicationEngine->rootObjects().at(0)->findChild<QObject*>("alertMessageDialogContent", Qt::FindChildrenRecursively);
	dialogWinComponent->setProperty("source", QString("qrc:/qml/BasicMessage.qml"));
	dialogWin->setProperty("title", _title);
	dialogWin->setProperty("width", "250");
	dialogWin->setProperty("height", "100");
	dialogWin->findChild<QObject*>("messageContent", Qt::FindChildrenRecursively)->setProperty("text", _message);
	QMetaObject::invokeMethod(dialogWin, "open");
}
