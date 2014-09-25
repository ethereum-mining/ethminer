#include <QtNetwork/QNetworkReply>
#include <QtQuick/QQuickView>
//#include <QtQml/QQmlContext>
//#include <QtQml/QQmlEngine>
#include <QtQml/QtQml>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <libdevcrypto/FileSystem.h>
#include <libethcore/Dagger.h>
#include <libevmface/Instruction.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include "BuildInfo.h"
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;

// types
using dev::bytes;
using dev::bytesConstRef;
using dev::h160;
using dev::h256;
using dev::u160;
using dev::u256;
using dev::u256s;
using dev::Address;
using dev::eth::BlockInfo;
using dev::eth::Client;
using dev::eth::Instruction;
using dev::KeyPair;
using dev::eth::NodeMode;
using dev::p2p::PeerInfo;
using dev::RLP;
using dev::Secret;
using dev::eth::Transaction;

// functions
using dev::toHex;
using dev::fromHex;
using dev::right160;
using dev::simpleDebugOut;
using dev::toLog2;
using dev::toString;
using dev::eth::units;
using dev::eth::disassemble;
using dev::eth::formatBalance;

// vars
using dev::g_logPost;
using dev::g_logVerbosity;

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Ethereum.png"));

	g_qmlMain = this;

	m_client.reset(new Client("Walleth", Address(), dev::getDataDir() + "/Walleth"));
	
	g_qmlClient = m_client.get();

	qRegisterMetaType<dev::u256>("dev::u256");
	qRegisterMetaType<dev::KeyPair>("dev::KeyPair");
	qRegisterMetaType<dev::Secret>("dev::Secret");
	qRegisterMetaType<dev::Address>("dev::Address");
	qRegisterMetaType<QmlAccount*>("QmlAccount*");
	qRegisterMetaType<QmlEthereum*>("QmlEthereum*");

	qmlRegisterType<QmlEthereum>("org.ethereum", 1, 0, "Ethereum");
	qmlRegisterType<QmlAccount>("org.ethereum", 1, 0, "Account");
	qmlRegisterSingletonType<QmlU256Helper>("org.ethereum", 1, 0, "Balance", QmlEthereum::constructU256Helper);
	qmlRegisterSingletonType<QmlKeyHelper>("org.ethereum", 1, 0, "Key", QmlEthereum::constructKeyHelper);

	/*
	ui->librariesView->setModel(m_libraryMan);
	ui->graphsView->setModel(m_graphMan);
	*/

	m_view = new QQuickView();

//	QQmlContext* context = m_view->rootContext();
//	context->setContextProperty("u256", new U256Helper(this));

	m_view->setSource(QUrl("qrc:/Simple.qml"));

	QWidget* w = QWidget::createWindowContainer(m_view);
	m_view->setResizeMode(QQuickView::SizeRootObjectToView);
	ui->fullDisplay->insertWidget(0, w);
	m_view->create();

//	m_timelinesItem = m_view->rootObject()->findChild<TimelinesItem*>("timelines");

	readSettings();
	refresh();

	m_refreshNetwork = new QTimer(this);
	connect(m_refreshNetwork, SIGNAL(timeout()), SLOT(refreshNetwork()));
	m_refreshNetwork->start(1000);

	connect(&m_webCtrl, &QNetworkAccessManager::finished, [&](QNetworkReply* _r)
	{
		m_servers = QString::fromUtf8(_r->readAll()).split("\n", QString::SkipEmptyParts);
		if (m_servers.size())
		{
			ui->net->setChecked(true);
			on_net_triggered(true);
		}
	});
	QNetworkRequest r(QUrl("http://www.ethereum.org/servers.poc" + QString(dev::Version).section('.', 1, 1) + ".txt"));
	r.setHeader(QNetworkRequest::UserAgentHeader, "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1712.0 Safari/537.36");
	m_webCtrl.get(r);
	srand(time(0));

	startTimer(200);

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->blockCount);
}

Main::~Main()
{
	writeSettings();
}

void Main::timerEvent(QTimerEvent *)
{

}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About Walleth PoC-" + QString(dev::Version).section('.', 1, 1), QString("Walleth/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) "\n" DEV_QUOTED(ETH_COMMIT_HASH) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBy Gav Wood, 2014.\nBased on a design by Vitalik Buterin.\n\nThanks to the various contributors including: Alex Leverington, Tim Hughes, caktux, Eric Lombrozo, Marko Simovic.");
}

void Main::writeSettings()
{
	QSettings s("ethereum", "walleth");
	QByteArray b;
	b.resize(sizeof(Secret) * m_myKeys.size());
	auto p = b.data();
	for (auto i: m_myKeys)
	{
		memcpy(p, &(i.secret()), sizeof(Secret));
		p += sizeof(Secret);
	}
	s.setValue("address", b);

	s.setValue("upnp", ui->upnp->isChecked());
	s.setValue("clientName", m_clientName);
	s.setValue("idealPeers", m_idealPeers);
	s.setValue("port", m_port);

	bytes d = client()->savePeers();
	if (d.size())
		m_peers = QByteArray((char*)d.data(), (int)d.size());
	s.setValue("peers", m_peers);

	s.setValue("geometry", saveGeometry());
	s.setValue("windowState", saveState());
}

void Main::readSettings()
{
	QSettings s("ethereum", "walleth");

	restoreGeometry(s.value("geometry").toByteArray());
	restoreState(s.value("windowState").toByteArray());

	QByteArray b = s.value("address").toByteArray();
	if (b.isEmpty())
		m_myKeys.append(KeyPair::create());
	else
	{
		h256 k;
		for (unsigned i = 0; i < b.size() / sizeof(Secret); ++i)
		{
			memcpy(&k, b.data() + i * sizeof(Secret), sizeof(Secret));
			m_myKeys.append(KeyPair(k));
		}
	}
	//m_eth->setAddress(m_myKeys.last().address());
	m_peers = s.value("peers").toByteArray();
	ui->upnp->setChecked(s.value("upnp", true).toBool());
	m_clientName = s.value("clientName", "").toString();
	m_idealPeers = s.value("idealPeers", 5).toInt();
	m_port = s.value("port", 30303).toInt();
}

void Main::refreshNetwork()
{
	auto ps = client()->peers();
	ui->peerCount->setText(QString::fromStdString(toString(ps.size())) + " peer(s)");
}

void Main::refresh()
{
	auto d = client()->blockChain().details();
	auto diff = BlockInfo(client()->blockChain().block()).difficulty;
	ui->blockCount->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

	m_keysChanged = false;
	u256 totalBalance = 0;
	for (auto i: m_myKeys)
	{
		u256 b = m_client->balanceAt(i.address());
		totalBalance += b;
	}
	ui->balance->setText(QString::fromStdString(formatBalance(totalBalance)));
}

void Main::on_net_triggered(bool _auto)
{
    string n = string("Walleth/v") + dev::Version;
	if (m_clientName.size())
		n += "/" + m_clientName.toStdString();
	n +=  "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM);
	client()->setClientVersion(n);
	if (ui->net->isChecked())
	{
		if (_auto)
		{
			QString s = m_servers[rand() % m_servers.size()];
			client()->startNetwork(m_port, s.section(':', 0, 0).toStdString(), s.section(':', 1).toInt(), NodeMode::Full, m_idealPeers, "", ui->upnp->isChecked());
		}
		else
			client()->startNetwork(m_port, string(), 0, NodeMode::Full, m_idealPeers, "", ui->upnp->isChecked());
		if (m_peers.size())
			client()->restorePeers(bytesConstRef((byte*)m_peers.data(), m_peers.size()));
	}
	else
		client()->stopNetwork();
}

void Main::on_connect_triggered()
{
	if (!ui->net->isChecked())
	{
		ui->net->setChecked(true);
		on_net_triggered();
	}
	bool ok = false;
	QString s = QInputDialog::getItem(this, "Connect to a Network Peer", "Enter a peer to which a connection may be made:", m_servers, m_servers.count() ? rand() % m_servers.count() : 0, true, &ok);
	if (ok && s.contains(":"))
	{
		string host = s.section(":", 0, 0).toStdString();
		unsigned short port = s.section(":", 1).toInt();
		client()->connect(host, port);
	}
}

void Main::on_mine_triggered()
{
	if (ui->mine->isChecked())
	{
		client()->setAddress(m_myKeys.last().address());
		client()->startMining();
	}
	else
		client()->stopMining();
}

void Main::on_create_triggered()
{
	m_myKeys.append(KeyPair::create());
	m_keysChanged = true;
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_MainWin.cpp"

// include qrc content
#include\
"qrc_Resources.cpp"

#endif
