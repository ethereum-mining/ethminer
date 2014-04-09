#include <QtNetwork/QNetworkReply>
#include <QtQuick/QQuickView>
//#include <QtQml/QQmlContext>
//#include <QtQml/QQmlEngine>
#include <QtQml/QtQml>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <libethereum/Dagger.h>
#include <libethereum/Client.h>
#include <libethereum/Instruction.h>
#include <libethereum/FileSystem.h>
#include <libethereum/PeerServer.h>
#include "BuildInfo.h"
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;

// types
using eth::bytes;
using eth::bytesConstRef;
using eth::h160;
using eth::h256;
using eth::u160;
using eth::u256;
using eth::u256s;
using eth::Address;
using eth::BlockInfo;
using eth::Client;
using eth::Instruction;
using eth::KeyPair;
using eth::NodeMode;
using eth::PeerInfo;
using eth::RLP;
using eth::Secret;
using eth::Transaction;

// functions
using eth::toHex;
using eth::assemble;
using eth::compileLisp;
using eth::disassemble;
using eth::formatBalance;
using eth::fromHex;
using eth::right160;
using eth::simpleDebugOut;
using eth::toLog2;
using eth::toString;
using eth::units;

// vars
using eth::g_logPost;
using eth::g_logVerbosity;
using eth::c_instructionInfo;

// Horrible global for the mainwindow. Needed for the QEthereums to find the Main window which acts as multiplexer for now.
// Can get rid of this once we've sorted out ITC for signalling & multiplexed querying.
Main* g_main = nullptr;

QEthereum::QEthereum(QObject* _p): QObject(_p)
{
	connect(g_main, SIGNAL(changed()), SIGNAL(changed()));
}

QEthereum::~QEthereum()
{
}

Client* QEthereum::client() const
{
	return g_main->client();
}

Address QEthereum::coinbase() const
{
	return client()->address();
}

void QEthereum::setCoinbase(Address _a)
{
	if (client()->address() != _a)
	{
		client()->setAddress(_a);
		changed();
	}
}

QAccount::QAccount(QObject*)
{
}

QAccount::~QAccount()
{
}

void QAccount::setEthereum(QEthereum* _eth)
{
	if (m_eth == _eth)
		return;
	if (m_eth)
		disconnect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	m_eth = _eth;
	if (m_eth)
		connect(m_eth, SIGNAL(changed()), this, SIGNAL(changed()));
	ethChanged();
	changed();
}

u256 QAccount::balance() const
{
	if (m_eth)
		return m_eth->balanceAt(m_address);
	return 0;
}

double QAccount::txCount() const
{
	if (m_eth)
		return m_eth->txCountAt(m_address);
	return 0;
}

bool QAccount::isContract() const
{
	if (m_eth)
		return m_eth->isContractAt(m_address);
	return 0;
}

u256 QEthereum::balanceAt(Address _a) const
{
	return client()->postState().balance(_a);
}

bool QEthereum::isContractAt(Address _a) const
{
	return client()->postState().isContractAddress(_a);
}

bool QEthereum::isMining() const
{
	return client()->isMining();
}

bool QEthereum::isListening() const
{
	return client()->haveNetwork();
}

void QEthereum::setMining(bool _l)
{
	if (_l)
		client()->startMining();
	else
		client()->stopMining();
}

void QEthereum::setListening(bool _l)
{
	if (_l)
		client()->startNetwork();
	else
		client()->stopNetwork();
}

double QEthereum::txCountAt(Address _a) const
{
	return (double)client()->postState().transactionsFrom(_a);
}

unsigned QEthereum::peerCount() const
{
	return (unsigned)client()->peerCount();
}

void QEthereum::transact(Secret _secret, u256 _amount, u256 _gasPrice, u256 _gas, QByteArray _code, QByteArray _init)
{
	client()->transact(_secret, _amount, bytes(_code.data(), _code.data() + _code.size()), bytes(_init.data(), _init.data() + _init.size()), _gas, _gasPrice);
}

void QEthereum::transact(Secret _secret, Address _dest, u256 _amount, u256 _gasPrice, u256 _gas, QByteArray _data)
{
	client()->transact(_secret, _amount, _dest, bytes(_data.data(), _data.data() + _data.size()), _gas, _gasPrice);
}

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	setWindowIcon(QIcon(":/Ethereum.png"));

	g_main = this;

	m_client.reset(new Client("Walleth", Address(), eth::getDataDir() + "/Walleth"));

	qRegisterMetaType<eth::u256>("eth::u256");
	qRegisterMetaType<eth::KeyPair>("eth::KeyPair");
	qRegisterMetaType<eth::Secret>("eth::Secret");
	qRegisterMetaType<eth::Address>("eth::Address");
	qRegisterMetaType<QAccount*>("QAccount*");
	qRegisterMetaType<QEthereum*>("QEthereum*");

	qmlRegisterType<QEthereum>("org.ethereum", 1, 0, "Ethereum");
	qmlRegisterType<QAccount>("org.ethereum", 1, 0, "Account");
	qmlRegisterSingletonType<U256Helper>("org.ethereum", 1, 0, "Balance", QEthereum::constructU256Helper);
	qmlRegisterSingletonType<KeyHelper>("org.ethereum", 1, 0, "Key", QEthereum::constructKeyHelper);

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

	connect(this, SIGNAL(changed()), SLOT(refresh()));

	connect(&m_webCtrl, &QNetworkAccessManager::finished, [&](QNetworkReply* _r)
	{
		m_servers = QString::fromUtf8(_r->readAll()).split("\n", QString::SkipEmptyParts);
		if (m_servers.size())
		{
			ui->net->setChecked(true);
			on_net_triggered(true);
		}
	});
	QNetworkRequest r(QUrl("http://www.ethereum.org/servers.poc" + QString(ETH_QUOTED(ETH_VERSION)).section('.', 1, 1) + ".txt"));
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
	if (m_client->changed())
		changed();
}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About Walleth PoC-" + QString(ETH_QUOTED(ETH_VERSION)).section('.', 1, 1), "Walleth/v" ETH_QUOTED(ETH_VERSION) "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM) " - " ETH_QUOTED(ETH_COMMIT_HASH) "\nBy Gav Wood, 2014.\nBased on a design by Vitalik Buterin.\n\nTeam Ethereum++ includes: Tim Hughes, Eric Lombrozo, Marko Simovic, Alex Leverington and several others.");
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

	if (client()->peerServer())
	{
		bytes d = client()->peerServer()->savePeers();
		m_peers = QByteArray((char*)d.data(), (int)d.size());

	}
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

eth::State const& Main::state() const
{
	return ui->preview->isChecked() ? client()->postState() : client()->state();
}

void Main::refresh()
{
	eth::ClientGuard l(client());
	auto const& st = state();

	auto d = client()->blockChain().details();
	auto diff = BlockInfo(client()->blockChain().block()).difficulty;
	ui->blockCount->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

	m_keysChanged = false;
	u256 totalBalance = 0;
	for (auto i: m_myKeys)
	{
		u256 b = st.balance(i.address());
		totalBalance += b;
	}
	ui->balance->setText(QString::fromStdString(formatBalance(totalBalance)));
}

void Main::on_net_triggered(bool _auto)
{
	string n = "Walleth/v" ETH_QUOTED(ETH_VERSION);
	if (m_clientName.size())
		n += "/" + m_clientName.toStdString();
	n +=  "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM);
	client()->setClientVersion(n);
	if (ui->net->isChecked())
	{
		if (_auto)
		{
			QString s = m_servers[rand() % m_servers.size()];
			client()->startNetwork(m_port, s.section(':', 0, 0).toStdString(), s.section(':', 1).toInt(), NodeMode::Full, m_idealPeers, std::string(), ui->upnp->isChecked());
		}
		else
			client()->startNetwork(m_port, string(), 0, NodeMode::Full, m_idealPeers, std::string(), ui->upnp->isChecked());
		if (m_peers.size())
			client()->peerServer()->restorePeers(bytesConstRef((byte*)m_peers.data(), m_peers.size()));
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

// specify library dependencies, it's easier to do here than in the project since we can control the "d" debug suffix
#ifdef _DEBUG
#define QTLIB(x) x"d.lib"
#else 
#define QTLIB(x) x".lib"
#endif

#pragma comment(lib, QTLIB("Qt5PlatformSupport"))
#pragma comment(lib, QTLIB("Qt5Core"))
#pragma comment(lib, QTLIB("Qt5GUI"))
#pragma comment(lib, QTLIB("Qt5Widgets"))
#pragma comment(lib, QTLIB("Qt5Network"))
#pragma comment(lib, QTLIB("Qt5Quick"))
#pragma comment(lib, QTLIB("Qt5Declarative"))
#pragma comment(lib, QTLIB("Qt5Qml"))
#pragma comment(lib, QTLIB("qwindows"))
#pragma comment(lib, "Imm32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "winmm.lib")


#endif
