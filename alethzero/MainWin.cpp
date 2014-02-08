#include <QtNetwork/QNetworkReply>
#include <QtWidgets>
#include <QtCore>
#include <libethereum/Dagger.h>
#include <libethereum/Client.h>
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;
using namespace eth;

static void initUnits(QComboBox* _b)
{
	for (int n = units().size() - 1; n >= 0; --n)
		_b->addItem(QString::fromStdString(units()[n].second), n);
	_b->setCurrentIndex(6);
}

#define ETH_QUOTED(A) #A

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	initUnits(ui->valueUnits);
	initUnits(ui->feeUnits);
	g_logPost = [=](std::string const& s, char const*) { ui->log->addItem(QString::fromStdString(s)); };
	m_client = new Client("AlethZero/v" ETH_QUOTED(ETH_VERSION));

	readSettings();
	refresh();

	m_refresh = new QTimer(this);
	connect(m_refresh, SIGNAL(timeout()), SLOT(refresh()));
	m_refresh->start(1000);

#if ETH_DEBUG
	m_servers.append("192.168.0.10:30301");
#else
	connect(&m_webCtrl, &QNetworkAccessManager::finished, [&](QNetworkReply* _r)
	{
		m_servers = QString::fromUtf8(_r->readAll()).split("\n", QString::SkipEmptyParts);
	});
	QNetworkRequest r(QUrl("http://www.ethereum.org/servers.txt"));
	r.setHeader(QNetworkRequest::UserAgentHeader, "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1712.0 Safari/537.36");
	m_webCtrl.get(r);
	srand(time(0));
#endif

	on_verbosity_sliderMoved();

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->blockChain);
}

Main::~Main()
{
	g_logPost = simpleDebugOut;
	writeSettings();
	delete ui;
}

void Main::writeSettings()
{
	QSettings s("ethereum", "alethzero");
	QByteArray b;
	b.resize(sizeof(Secret) * m_myKeys.size());
	auto p = b.data();
	for (auto i: m_myKeys)
	{
		memcpy(p, &(i.secret()), sizeof(Secret));
		p += sizeof(Secret);
	}
	s.setValue("address", b);

	// TODO: save peers - implement it in PeerNetwork though returning RLP bytes
	/*for (uint i = 0; !s.value(QString("peer%1").arg(i)).isNull(); ++i)
	{
		s.value(QString("peer%1").arg(i)).toString();
	}*/
}

void Main::readSettings()
{
	QSettings s("ethereum", "alethzero");
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
	m_client->setAddress(m_myKeys.back().address());

	writeSettings();

	// TODO: restore peers - implement it in PeerNetwork though giving RLP bytes
	/*for (uint i = 0; !s.value(QString("peer%1").arg(i)).isNull(); ++i)
	{
		s.value(QString("peer%1").arg(i)).toString();
	}*/
}

void Main::refresh()
{
	m_client->lock();
	//if (m_client->changed())
	{
		ui->peerCount->setText(QString::fromStdString(toString(m_client->peerCount())) + " peer(s)");
		ui->peers->clear();
		for (PeerInfo const& i: m_client->peers())
			ui->peers->addItem(QString("%3 ms - %1:%2 - %4").arg(i.host.c_str()).arg(i.port).arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()).arg(i.clientVersion.c_str()));

		auto d = m_client->blockChain().details();
		auto diff = BlockInfo(m_client->blockChain().block()).difficulty;
		ui->blockChain->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

		auto acs = m_client->state().addresses();
		ui->accounts->clear();
		for (auto i: acs)
			ui->accounts->addItem(QString("%1 @ %2").arg(formatBalance(i.second).c_str()).arg(asHex(i.first.asArray()).c_str()));

		ui->transactionQueue->clear();
		for (pair<h256, bytes> const& i: m_client->transactionQueue().transactions())
		{
			Transaction t(i.second);
			ui->transactionQueue->addItem(QString("%1 (%2 fee) @ %3 <- %4")
								  .arg(formatBalance(t.value).c_str())
								  .arg(formatBalance(t.fee).c_str())
								  .arg(asHex(t.receiveAddress.asArray()).c_str())
								  .arg(asHex(t.sender().asArray()).c_str()) );
		}

		ui->transactions->clear();
		auto const& bc = m_client->blockChain();
		for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
		{
			auto d = bc.details(h);
			ui->transactions->addItem(QString("# %1 ==== %2").arg(d.number).arg(asHex(h.asArray()).c_str()));
			for (auto const& i: RLP(bc.block(h))[1])
			{
				Transaction t(i.data());
				ui->transactions->addItem(QString("%1 (%2) @ %3 <- %4")
								  .arg(formatBalance(t.value).c_str())
								  .arg(formatBalance(t.fee).c_str())
								  .arg(asHex(t.receiveAddress.asArray()).c_str())
								  .arg(asHex(t.sender().asArray()).c_str()) );
			}
		}
	}

	ui->ourAccounts->clear();
	u256 totalBalance = 0;
	for (auto i: m_myKeys)
	{
		u256 b = m_client->state().balance(i.address());
		ui->ourAccounts->addItem(QString("%1 @ %2").arg(formatBalance(b).c_str()).arg(asHex(i.address().asArray()).c_str()));
		totalBalance += b;
	}
	ui->balance->setText(QString::fromStdString(formatBalance(totalBalance)));
	m_client->unlock();
}

void Main::on_ourAccounts_doubleClicked()
{
	qApp->clipboard()->setText(ui->ourAccounts->currentItem()->text().section(" @ ", 1));
}

void Main::on_log_doubleClicked()
{
	qApp->clipboard()->setText(ui->log->currentItem()->text());
}

void Main::on_accounts_doubleClicked()
{
	qApp->clipboard()->setText(ui->accounts->currentItem()->text().section(" @ ", 1));
}

void Main::on_net_triggered()
{
	ui->port->setEnabled(!ui->net->isChecked());
	if (ui->net->isChecked())
		m_client->startNetwork(ui->port->value(), string(), 0, NodeMode::Full, 5, std::string(), ui->upnp->isChecked());
	else
		m_client->stopNetwork();
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
		short port = s.section(":", 1).toInt();
		m_client->connect(host, port);
	}
}

void Main::on_verbosity_sliderMoved()
{
	g_logVerbosity = ui->verbosity->value();
}

void Main::on_mine_triggered()
{
	if (ui->mine->isChecked())
	{
		m_client->setAddress(m_myKeys.last().address());
		m_client->startMining();
	}
	else
		m_client->stopMining();
}

void Main::on_send_clicked()
{
	u256 value = ui->value->value() * units()[units().size() - 1 - ui->valueUnits->currentIndex()].first;
	u256 fee = ui->fee->value() * units()[units().size() - 1 - ui->feeUnits->currentIndex()].first;
	u256 totalReq = value + fee;
	m_client->lock();
	for (auto i: m_myKeys)
		if (m_client->state().balance(i.address()) >= totalReq)
		{
			m_client->unlock();
			Secret s = m_myKeys.front().secret();
			Address r = Address(fromUserHex(ui->destination->text().toStdString()));
			auto ds = ui->data->toPlainText().split(QRegExp("[^0-9a-fA-Fx]+"));
			u256s data;
			data.reserve(ds.size());
			for (QString const& i: ds)
				data.push_back(u256(i.toStdString()));
			m_client->transact(s, r, value, fee, data);
			refresh();
			return;
		}
	m_client->unlock();
	statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
}

void Main::on_create_triggered()
{
	m_myKeys.append(KeyPair::create());
}
