#include <QtNetwork/QNetworkReply>
#include <QtQuick/QQuickView>
#include <QtQml/QQmlContext>
#include <QtQml/QQmlEngine>
#include <QtQml/QtQml>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <libethcore/FileSystem.h>
#include <libethcore/Dagger.h>
#include <libevmface/Instruction.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include "BuildInfo.h"
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;
using namespace eth;

Main::Main(QWidget *parent) :
	QObject(parent)
{
/*	qRegisterMetaType<eth::u256>("eth::u256");
	qRegisterMetaType<eth::KeyPair>("eth::KeyPair");
	qRegisterMetaType<eth::Secret>("eth::Secret");
	qRegisterMetaType<eth::Address>("eth::Address");
	qRegisterMetaType<QmlAccount*>("QmlAccount*");
	qRegisterMetaType<QmlEthereum*>("QmlEthereum*");

	qmlRegisterType<QmlEthereum>("org.ethereum", 1, 0, "Ethereum");
	qmlRegisterType<QmlAccount>("org.ethereum", 1, 0, "Account");
	qmlRegisterSingletonType<QmlU256Helper>("org.ethereum", 1, 0, "Balance", QmlEthereum::constructU256Helper);
	qmlRegisterSingletonType<QmlKeyHelper>("org.ethereum", 1, 0, "Key", QmlEthereum::constructKeyHelper);
*/
	/*
	ui->librariesView->setModel(m_libraryMan);
	ui->graphsView->setModel(m_graphMan);
	*/




//	QQmlContext* context = m_view->rootContext();
//	context->setContextProperty("u256", new U256Helper(this));
}

Main::~Main()
{
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_MainWin.cpp"

#endif
