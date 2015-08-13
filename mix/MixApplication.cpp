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
/** @file MixApplication.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include "MixApplication.h"
#include <boost/exception/diagnostic_information.hpp>
#include <QQmlApplicationEngine>
#include <QUrl>
#include <QIcon>
#include <QFont>
#ifdef ETH_HAVE_WEBENGINE
#include <QtWebEngine/QtWebEngine>
#endif
#include "CodeModel.h"
#include "ClientModel.h"
#include "FileIo.h"
#include "QEther.h"
#include "QVariableDeclaration.h"
#include "SortFilterProxyModel.h"
#include "Clipboard.h"
#include "HttpServer.h"
#include "InverseMouseArea.h"

extern int qInitResources_js();
using namespace dev::mix;




ApplicationService::ApplicationService()
{
#ifdef ETH_HAVE_WEBENGINE
	QtWebEngine::initialize();
#endif
	QFont f;
	m_systemPointSize = f.pointSize();
}

MixApplication::MixApplication(int& _argc, char* _argv[]):
	QApplication(_argc, _argv), m_engine(new QQmlApplicationEngine())
{
	setWindowIcon(QIcon(":/res/mix_256x256x32.png"));
	m_engine->load(QUrl("qrc:/qml/Application.qml"));
}

void MixApplication::initialize()
{
#if __linux
	//work around ubuntu appmenu-qt5 bug
	//https://bugs.launchpad.net/ubuntu/+source/appmenu-qt5/+bug/1323853
	putenv((char*)"QT_QPA_PLATFORMTHEME=");
	putenv((char*)"QSG_RENDER_LOOP=threaded");
#endif
#if (defined(_WIN32) || defined(_WIN64))
	if (!getenv("OPENSSL_CONF"))
		putenv((char*)"OPENSSL_CONF=c:\\");
#endif
#ifdef ETH_HAVE_WEBENGINE
	qInitResources_js();
#endif

	setOrganizationName(tr("Ethereum"));
	setOrganizationDomain(tr("ethereum.org"));
	setApplicationName(tr("Mix"));
	setApplicationVersion("0.1");

	qmlRegisterType<CodeModel>("org.ethereum.qml.CodeModel", 1, 0, "CodeModel");
	qmlRegisterType<ClientModel>("org.ethereum.qml.ClientModel", 1, 0, "ClientModel");
	qmlRegisterType<ApplicationService>("org.ethereum.qml.ApplicationService", 1, 0, "ApplicationService");
	qmlRegisterType<FileIo>("org.ethereum.qml.FileIo", 1, 0, "FileIo");
	qmlRegisterType<QEther>("org.ethereum.qml.QEther", 1, 0, "QEther");
	qmlRegisterType<QBigInt>("org.ethereum.qml.QBigInt", 1, 0, "QBigInt");
	qmlRegisterType<QVariableDeclaration>("org.ethereum.qml.QVariableDeclaration", 1, 0, "QVariableDeclaration");
	qmlRegisterType<RecordLogEntry>("org.ethereum.qml.RecordLogEntry", 1, 0, "RecordLogEntry");
	qmlRegisterType<SortFilterProxyModel>("org.ethereum.qml.SortFilterProxyModel", 1, 0, "SortFilterProxyModel");
	qmlRegisterType<QSolidityType>("org.ethereum.qml.QSolidityType", 1, 0, "QSolidityType");
	qmlRegisterType<Clipboard>("org.ethereum.qml.Clipboard", 1, 0, "Clipboard");
	qmlRegisterType<HttpServer>("HttpServer", 1, 0, "HttpServer");
	qmlRegisterType<InverseMouseArea>("org.ethereum.qml.InverseMouseArea", 1, 0, "InverseMouseArea");
	qRegisterMetaType<CodeModel*>("CodeModel*");
	qRegisterMetaType<ClientModel*>("ClientModel*");
}

MixApplication::~MixApplication()
{
}

bool MixApplication::notify(QObject * receiver, QEvent * event)
{
	try
	{
		return QApplication::notify(receiver, event);
	}
	catch (...)
	{
		std::cerr << boost::current_exception_diagnostic_information();
	}
	return false;
}
