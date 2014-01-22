#include <QtWidgets>
#include <QtCore>
#include <libethereum/Dagger.h>
#include "Main.h"
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

	/*for (uint i = 0; !s.value(QString("peer%1").arg(i)).isNull(); ++i)
	{
		s.value(QString("peer%1").arg(i)).toString();
	}*/
}

void Main::refresh()
{
	ui->balance->setText(QString::fromStdString(toString(m_client.state().balance(m_myKey.address()))) + " wei");
	ui->peerCount->setText(QString::fromStdString(toString(m_client.peerCount())) + " peer(s)");
	ui->address->setText(QString::fromStdString(asHex(m_client.state().address().asArray())));
	ui->peers->clear();
	for (PeerInfo const& i: m_client.peers())
		ui->peers->addItem(QString("%3 ms - %1:%2 - %4").arg(i.endpoint.address().to_string().c_str()).arg(i.endpoint.port()).arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()).arg(i.clientVersion.c_str()));

	auto d = m_client.blockChain().details();
	auto diff = BlockInfo(m_client.blockChain().block()).difficulty;
	ui->blockChain->setText(QString("%1 blocks @ (%3) - %2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));
}

void Main::on_net_toggled()
{
	if (ui->net->isChecked())
		m_client.startNetwork(ui->port->value());
	else
		m_client.stopNetwork();
}

void Main::on_connect_clicked()
{
	QString s = QInputDialog::getText(this, "Enter first peer", "Enter a peer to which a connection may be made", QLineEdit::Normal, "127.0.0.1:30303");
	string host = s.section(":", 0, 0).toStdString();
	short port = s.section(":", 1).toInt();
	m_client.connect(host, port);
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
	Secret s = Secret(fromUserHex(ui->address->text().toStdString()));
	Address r = Address(fromUserHex(ui->destination->text().toStdString()));
	m_client.transact(s, r, ui->value->value(), ui->fee->value());
	refresh();
}

void Main::on_create_clicked()
{
	KeyPair p = KeyPair::create();
	QString s = QString::fromStdString("New key:\n" + asHex(p.secret().asArray()) + "\nAddress: " + asHex(p.address().asArray()));
	QMessageBox::information(this, "New Key", s, QMessageBox::Ok);
}
