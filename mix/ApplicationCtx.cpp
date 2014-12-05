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
/** @file ApplicationCtx.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 * Provides an access to the current QQmlApplicationEngine which is used to add QML file on the fly.
 * In the future this class can be extended to add more variable related to the context of the application.
 * For now ApplicationCtx provides reference to:
 * - QQmlApplicationEngine
 * - dev::WebThreeDirect (and dev::eth::Client)
 * - KeyEventManager
 */

#include <QDebug>
#include <QMessageBox>
#include <QQmlComponent>
#include <QQmlApplicationEngine>
#include "libdevcrypto/FileSystem.h"
#include "KeyEventManager.h"
#include "ApplicationCtx.h"
using namespace dev;
using namespace dev::mix;
using namespace dev::eth;

ApplicationCtx* ApplicationCtx::Instance = nullptr;

ApplicationCtx::ApplicationCtx(QQmlApplicationEngine* _engine)
{
	m_applicationEngine = _engine;
	m_keyEventManager = std::unique_ptr<KeyEventManager>(new KeyEventManager());
	m_webThree = std::unique_ptr<dev::WebThreeDirect>(new WebThreeDirect(std::string("Mix/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM), getDataDir() + "/Mix", false, {"eth", "shh"}));
}

ApplicationCtx::~ApplicationCtx()
{
	delete m_applicationEngine;
}

QQmlApplicationEngine* ApplicationCtx::appEngine()
{
	return m_applicationEngine;
}

dev::eth::Client* ApplicationCtx::getEthereumClient()
{
	return m_webThree.get()->ethereum();
}

void ApplicationCtx::initKeyEventManager()
{
	QObject* mainContent = m_applicationEngine->rootObjects().at(0)->findChild<QObject*>("mainContent", Qt::FindChildrenRecursively);
	if (mainContent)
	{
		QObject::connect(mainContent, SIGNAL(keyPressed(QVariant)), m_keyEventManager.get(), SLOT(keyPressed(QVariant)));
	}
	else
		qDebug() << "Unable to find QObject of mainContent.qml. KeyEvent will not be handled!";
}

KeyEventManager* ApplicationCtx::getKeyEventManager()
{
	return m_keyEventManager.get();
}

void ApplicationCtx::setApplicationContext(QQmlApplicationEngine* _engine)
{
	if (Instance == nullptr)
		Instance = new ApplicationCtx(_engine);
}

void ApplicationCtx::displayMessageDialog(QString _title, QString _message)
{
	QQmlComponent component(m_applicationEngine, QUrl("qrc:/qml/BasicMessage.qml"));
	QObject* dialog = component.create();
	dialog->findChild<QObject*>("messageContent", Qt::FindChildrenRecursively)->setProperty("text", _message);
	QObject* dialogWin = ApplicationCtx::getInstance()->appEngine()->rootObjects().at(0)->findChild<QObject*>("messageDialog", Qt::FindChildrenRecursively);
	QMetaObject::invokeMethod(dialogWin, "close");
	dialogWin->setProperty("contentItem", QVariant::fromValue(dialog));
	dialogWin->setProperty("title", _title);
	dialogWin->setProperty("width", "250");
	dialogWin->setProperty("height", "100");
	QMetaObject::invokeMethod(dialogWin, "open");
}
