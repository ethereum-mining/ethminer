#include <QtNetwork/QNetworkReply>
#include <QtWidgets>
#include <QtCore>
#include <libethereum/Dagger.h>
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;
using namespace eth;

Main::Main(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::Main),
	m_client("AlephZero/v0.1")
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);

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
}

Main::~Main()
{
	writeSettings();
	delete ui;
}

void Main::writeSettings()
{
	QSettings s("ethereum", "alephzero");
	QByteArray b;
	b.resize(32);
	memcpy(b.data(), &m_myKey, 32);
	s.setValue("address", b);
}

void Main::readSettings()
{
	QSettings s("ethereum", "alephzero");
	QByteArray b = s.value("address").toByteArray();
	if (b.isEmpty())
		m_myKey = KeyPair::create();
	else
	{
		h256 k;
		memcpy(&k, b.data(), 32);
		m_myKey = KeyPair(k);
	}
	m_client.setAddress(m_myKey.address());

	writeSettings();

	/*for (uint i = 0; !s.value(QString("peer%1").arg(i)).isNull(); ++i)
	{
		s.value(QString("peer%1").arg(i)).toString();
	}*/
}

std::string formatBalance(u256 _b)
{
	static const vector<pair<u256, string>> c_units =
	{
		{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000, "Uether"},
		{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000, "Vether"},
		{((((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000, "Dether"},
		{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000000, "Nether"},
		{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000000, "Yether"},
		{(((u256(1000000000) * 1000000000) * 1000000000) * 1000000000) * 1000, "Zether"},
		{((u256(1000000000) * 1000000000) * 1000000000) * 1000000000, "Eether"},
		{((u256(1000000000) * 1000000000) * 1000000000) * 1000000, "Pether"},
		{((u256(1000000000) * 1000000000) * 1000000000) * 1000, "Tether"},
		{(u256(1000000000) * 1000000000) * 1000000000, "Gether"},
		{(u256(1000000000) * 1000000000) * 1000000, "Mether"},
		{(u256(1000000000) * 1000000000) * 1000, "Kether"},
		{u256(1000000000) * 1000000000, "ether"},
		{u256(1000000000) * 1000000, "finney"},
		{u256(1000000000) * 1000, "szabo"},
		{u256(1000000000), "Gwei"},
		{u256(1000000), "Mwei"},
		{u256(1000), "Kwei"}
	};
	ostringstream ret;
	if (_b > c_units[0].first * 10000)
	{
		ret << (_b / c_units[0].first) << " " << c_units[0].second;
		return ret.str();
	}
	ret << setprecision(5);
	for (auto const& i: c_units)
		if (_b >= i.first * 100)
		{
			ret << (double(_b / (i.first / 1000)) / 1000.0) << " " << i.second;
			return ret.str();
		}
	ret << _b << " wei";
	return ret.str();
}

void Main::refresh()
{
	m_client.lock();
	ui->balance->setText(QString::fromStdString(formatBalance(m_client.state().balance(m_myKey.address()))));
	ui->peerCount->setText(QString::fromStdString(toString(m_client.peerCount())) + " peer(s)");
	ui->address->setText(QString::fromStdString(asHex(m_client.state().address().asArray())));
	ui->peers->clear();
	for (PeerInfo const& i: m_client.peers())
		ui->peers->addItem(QString("%3 ms - %1:%2 - %4").arg(i.host.c_str()).arg(i.port).arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()).arg(i.clientVersion.c_str()));

	auto d = m_client.blockChain().details();
	auto diff = BlockInfo(m_client.blockChain().block()).difficulty;
	ui->blockChain->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

	auto acs = m_client.state().addresses();
	ui->accounts->clear();
	for (auto i: acs)
		ui->accounts->addItem(QString("%1 @ %2").arg(formatBalance(i.second).c_str()).arg(asHex(i.first.asArray()).c_str()));

	ui->transactionQueue->clear();
	for (pair<h256, bytes> const& i: m_client.transactionQueue().transactions())
	{
		Transaction t(i.second);
		ui->transactionQueue->addItem(QString("%1 (%2 fee) @ %3 <- %4")
							  .arg(formatBalance(t.value).c_str())
							  .arg(formatBalance(t.fee).c_str())
							  .arg(asHex(t.receiveAddress.asArray()).c_str())
							  .arg(asHex(t.sender().asArray()).c_str()) );
	}

	ui->transactions->clear();
	auto const& bc = m_client.blockChain();
	for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
	{
		auto d = bc.details(h);
		ui->transactions->addItem(QString("# %1 ==== %2").arg(d.number).arg(asHex(h.asArray()).c_str()));
		for (auto const& i: RLP(bc.block(h))[1])
		{
			Transaction t(i.data());
			ui->transactions->addItem(QString("%1 wei (%2 fee) @ %3 <- %4")
							  .arg(toString(t.value).c_str())
							  .arg(toString(t.fee).c_str())
							  .arg(asHex(t.receiveAddress.asArray()).c_str())
							  .arg(asHex(t.sender().asArray()).c_str()) );
		}
	}

	m_client.unlock();
}

void Main::on_net_toggled()
{
	if (ui->net->isChecked())
		m_client.startNetwork(ui->port->value(), string(), 30303, 6);
	else
		m_client.stopNetwork();
}

void Main::on_connect_clicked()
{
	if (!ui->net->isChecked())
		ui->net->setChecked(true);
	bool ok = false;
	QString s = QInputDialog::getItem(this, "Connect to a Network Peer", "Enter a peer to which a connection may be made:", m_servers, m_servers.count() ? rand() % m_servers.count() : 0, true, &ok);
	if (ok)
	{
		string host = s.section(":", 0, 0).toStdString();
		short port = s.section(":", 1).toInt();
		m_client.connect(host, port);
	}
}

void Main::on_mine_toggled()
{
	if (ui->mine->isChecked())
		m_client.startMining();
	else
		m_client.stopMining();
}

void Main::on_send_clicked()
{
	Secret s = m_myKey.secret();
	Address r = Address(fromUserHex(ui->destination->text().toStdString()));
	m_client.transact(s, r, ui->value->value(), ui->fee->value());
	refresh();
}

void Main::on_create_clicked()
{
	KeyPair p = KeyPair::create();
	QString s = QString::fromStdString("The new secret key is:\n" + asHex(p.secret().asArray()) + "\n\nAddress:\n" + asHex(p.address().asArray()));
	QMessageBox::information(this, "Create Key", s, QMessageBox::Ok);
}
