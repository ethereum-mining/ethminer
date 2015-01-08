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
#include <QQmlContext>
#include <QQmlApplicationEngine>
#include <QStandardPaths>
#include <QFile>
#include <QDir>
#include <libdevcrypto/FileSystem.h>
#include <libwebthree/WebThree.h>
#include "AppContext.h"
#include "CodeModel.h"

using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

const QString c_projectFileName = "project.mix";

AppContext::AppContext(QQmlApplicationEngine* _engine)
{
	m_applicationEngine = _engine;
	//m_webThree = std::unique_ptr<dev::WebThreeDirect>(new WebThreeDirect(std::string("Mix/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM), getDataDir() + "/Mix", false, {"eth", "shh"}));
	m_codeModel = std::unique_ptr<CodeModel>(new CodeModel(this));
	m_applicationEngine->rootContext()->setContextProperty("codeModel", m_codeModel.get());
	m_applicationEngine->rootContext()->setContextProperty("appContext", this);
}

AppContext::~AppContext()
{
}

void AppContext::loadProject()
{
	QString path = QStandardPaths::locate(QStandardPaths::DataLocation, c_projectFileName);
	if (!path.isEmpty())
	{
		QFile file(path);
		if (file.open(QIODevice::ReadOnly | QIODevice::Text))
		{
			QTextStream stream(&file);
			QString json = stream.readAll();
			emit projectLoaded(json);
		}
	}
}

QQmlApplicationEngine* AppContext::appEngine()
{
	return m_applicationEngine;
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

void AppContext::saveProject(QString const& _json)
{
	QDir dirPath(QStandardPaths::writableLocation(QStandardPaths::DataLocation));
	QString path = QDir(dirPath).filePath(c_projectFileName);
	if (!path.isEmpty())
	{
		dirPath.mkpath(dirPath.path());
		QFile file(path);
		if (file.open(QIODevice::WriteOnly | QIODevice::Text))
		{
			QTextStream stream(&file);
			stream << _json;
		}
	}
}
