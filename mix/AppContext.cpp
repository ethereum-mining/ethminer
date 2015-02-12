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

#include <QMessageBox>
#include <QClipboard>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlApplicationEngine>
#include <QWindow>
#include "CodeModel.h"
#include "FileIo.h"
#include "ClientModel.h"
#include "CodeEditorExtensionManager.h"
#include "Exceptions.h"
#include "QEther.h"
#include "QVariableDefinition.h"
#include "HttpServer.h"
#include "AppContext.h"

using namespace dev;
using namespace dev::eth;
using namespace dev::mix;

const QString c_projectFileName = "project.mix";

AppContext::AppContext(QQmlApplicationEngine* _engine)
{
	m_applicationEngine = _engine;
	m_codeModel.reset(new CodeModel(this));
	m_clientModel.reset(new ClientModel(this));
	m_fileIo.reset(new FileIo());
	connect(QApplication::clipboard(), &QClipboard::dataChanged, [this] { emit clipboardChanged();});
}

AppContext::~AppContext()
{
}

void AppContext::load()
{
	m_applicationEngine->rootContext()->setContextProperty("appContext", this);
	QFont f;
	m_applicationEngine->rootContext()->setContextProperty("systemPointSize", f.pointSize());
	qmlRegisterType<FileIo>("org.ethereum.qml", 1, 0, "FileIo");
	m_applicationEngine->rootContext()->setContextProperty("codeModel", m_codeModel.get());
	m_applicationEngine->rootContext()->setContextProperty("fileIo", m_fileIo.get());
	qmlRegisterType<QEther>("org.ethereum.qml.QEther", 1, 0, "QEther");
	qmlRegisterType<QBigInt>("org.ethereum.qml.QBigInt", 1, 0, "QBigInt");
	qmlRegisterType<QIntType>("org.ethereum.qml.QIntType", 1, 0, "QIntType");
	qmlRegisterType<QRealType>("org.ethereum.qml.QRealType", 1, 0, "QRealType");
	qmlRegisterType<QStringType>("org.ethereum.qml.QStringType", 1, 0, "QStringType");
	qmlRegisterType<QHashType>("org.ethereum.qml.QHashType", 1, 0, "QHashType");
	qmlRegisterType<QBoolType>("org.ethereum.qml.QBoolType", 1, 0, "QBoolType");
	qmlRegisterType<QVariableDeclaration>("org.ethereum.qml.QVariableDeclaration", 1, 0, "QVariableDeclaration");
	QQmlComponent projectModelComponent(m_applicationEngine, QUrl("qrc:/qml/ProjectModel.qml"));
	QObject* projectModel = projectModelComponent.create();
	if (projectModelComponent.isError())
	{
		QmlLoadException exception;
		for (auto const& e : projectModelComponent.errors())
			exception << QmlErrorInfo(e);
		BOOST_THROW_EXCEPTION(exception);
	}
	m_applicationEngine->rootContext()->setContextProperty("projectModel", projectModel);
	qmlRegisterType<CodeEditorExtensionManager>("CodeEditorExtensionManager", 1, 0, "CodeEditorExtensionManager");
	qmlRegisterType<HttpServer>("HttpServer", 1, 0, "HttpServer");
	m_applicationEngine->load(QUrl("qrc:/qml/main.qml"));
	QWindow *window = qobject_cast<QWindow*>(m_applicationEngine->rootObjects().at(0));
	window->setIcon(QIcon(":/res/mix_256x256x32.png"));
	appLoaded();
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

QString AppContext::clipboard() const
{
	QClipboard *clipboard = QApplication::clipboard();
	return clipboard->text();
}

void AppContext::toClipboard(QString _text)
{
	QClipboard *clipboard = QApplication::clipboard();
	clipboard->setText(_text);
}
