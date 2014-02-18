#include <QtNetwork/QNetworkReply>
#include <QtWidgets>
#include <QtCore>
#include <libethereum/Dagger.h>
#include <libethereum/Client.h>
#include <libethereum/Instruction.h>
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

#define ADD_QUOTES_HELPER(s) #s
#define ADD_QUOTES(s) ADD_QUOTES_HELPER(s)

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	g_logPost = [=](std::string const& s, char const* c) { simpleDebugOut(s, c); ui->log->addItem(QString::fromStdString(s)); };
	m_client = new Client("AlethZero");

	readSettings();
	refresh();

	m_refresh = new QTimer(this);
	connect(m_refresh, SIGNAL(timeout()), SLOT(refresh()));
	m_refresh->start(100);

#if ETH_DEBUG
	m_servers.append("192.168.0.10:30301");
#else
	connect(&m_webCtrl, &QNetworkAccessManager::finished, [&](QNetworkReply* _r)
	{
		m_servers = QString::fromUtf8(_r->readAll()).split("\n", QString::SkipEmptyParts);
	});
	QNetworkRequest r(QUrl("http://www.ethereum.org/servers.poc3.txt"));
	r.setHeader(QNetworkRequest::UserAgentHeader, "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1712.0 Safari/537.36");
	m_webCtrl.get(r);
	srand(time(0));
#endif

	on_verbosity_sliderMoved();

	initUnits(ui->valueUnits);
	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->blockCount);
}

Main::~Main()
{
	g_logPost = simpleDebugOut;
	writeSettings();
	delete ui;
}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About AlethZero PoC-3", "AlethZero/v" ADD_QUOTES(ETH_VERSION) "/" ADD_QUOTES(ETH_BUILD_TYPE) "/" ADD_QUOTES(ETH_BUILD_PLATFORM) "\nBy Gav Wood, 2014.\nBased on a design by Vitalik Buterin.\n\nTeam Ethereum++ includes: Eric Lombrozo, Marko Simovic, Alex Leverington, Tim Hughes and several others.");
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

	s.setValue("upnp", ui->upnp->isChecked());
	s.setValue("clientName", ui->clientName->text());
	s.setValue("idealPeers", ui->idealPeers->value());
	s.setValue("port", ui->port->value());

	if (m_client->peerServer())
	{
		bytes d = m_client->peerServer()->savePeers();
		m_peers = QByteArray((char*)d.data(), d.size());

	}
	s.setValue("peers", m_peers);

	s.setValue("geometry", saveGeometry());
}

void Main::readSettings()
{
	QSettings s("ethereum", "alethzero");

	restoreGeometry(s.value("geometry").toByteArray());

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
	m_peers = s.value("peers").toByteArray();
	ui->upnp->setChecked(s.value("upnp", true).toBool());
	ui->clientName->setText(s.value("clientName", "").toString());
	ui->idealPeers->setValue(s.value("idealPeers", ui->idealPeers->value()).toInt());
	ui->port->setValue(s.value("port", ui->port->value()).toInt());
}

void Main::refresh()
{
	m_client->lock();
	bool c = m_client->changed();
	if (c)
	{
		ui->peerCount->setText(QString::fromStdString(toString(m_client->peerCount())) + " peer(s)");
		ui->peers->clear();
		for (PeerInfo const& i: m_client->peers())
			ui->peers->addItem(QString("%3 ms - %1:%2 - %4").arg(i.host.c_str()).arg(i.port).arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()).arg(i.clientVersion.c_str()));

		auto d = m_client->blockChain().details();
		auto diff = BlockInfo(m_client->blockChain().block()).difficulty;
		ui->blockCount->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

		auto acs = m_client->state().addresses();
		ui->accounts->clear();
		for (auto i: acs)
		{
			(new QListWidgetItem(QString("%1 @ %2").arg(formatBalance(i.second).c_str()).arg(i.first.abridged().c_str()), ui->accounts))
				->setData(Qt::UserRole, QByteArray((char const*)i.first.data(), Address::size));
		}

		ui->transactionQueue->clear();
		for (pair<h256, Transaction> const& i: m_client->pending())
		{
			ui->transactionQueue->addItem(QString("%1 [%4] @ %2 <- %3")
								.arg(formatBalance(i.second.value).c_str())
								.arg(i.second.receiveAddress.abridged().c_str())
								.arg(i.second.safeSender().abridged().c_str())
								.arg((unsigned)i.second.nonce));
		}

		ui->blocks->clear();
		auto const& bc = m_client->blockChain();
		for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
		{
			auto d = bc.details(h);
			QListWidgetItem* blockItem = new QListWidgetItem(QString("# %1 ==== %2").arg(d.number).arg(h.abridged().c_str()), ui->blocks);
			blockItem->setData(Qt::UserRole, QByteArray((char const*)h.data(), h.size));
			int n = 0;
			for (auto const& i: RLP(bc.block(h))[1])
			{
				Transaction t(i.data());
				QListWidgetItem* txItem = new QListWidgetItem(QString("%1 [%4] @ %2 <- %3")
													 .arg(formatBalance(t.value).c_str())
													 .arg(t.receiveAddress.abridged().c_str())
													 .arg(t.safeSender().abridged().c_str())
													 .arg((unsigned)t.nonce), ui->blocks);
				txItem->setData(Qt::UserRole, QByteArray((char const*)h.data(), h.size));
				txItem->setData(Qt::UserRole + 1, n);
				n++;
			}
		}
	}

	if (c || m_keysChanged)
	{
		m_keysChanged = false;
		ui->ourAccounts->clear();
		u256 totalBalance = 0;
		for (auto i: m_myKeys)
		{
			u256 b = m_client->state().balance(i.address());
			(new QListWidgetItem(QString("%1 @ %2").arg(formatBalance(b).c_str()).arg(i.address().abridged().c_str()), ui->ourAccounts))
				->setData(Qt::UserRole, QByteArray((char const*)i.address().data(), Address::size));
			totalBalance += b;
		}
		ui->balance->setText(QString::fromStdString(formatBalance(totalBalance)));
	}
	m_client->unlock();
}

void Main::on_blocks_currentItemChanged()
{
	ui->info->clear();
	m_client->lock();
	if (auto item = ui->blocks->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 32);
		auto h = h256((byte const*)hba.data(), h256::ConstructFromPointer);
		auto details = m_client->blockChain().details(h);
		auto blockData = m_client->blockChain().block(h);
		auto block = RLP(blockData);
		BlockInfo info(blockData);

		stringstream s;

		if (item->data(Qt::UserRole + 1).isNull())
		{
			char timestamp[64];
			time_t rawTime = (time_t)(uint64_t)info.timestamp;
			strftime(timestamp, 64, "%c", localtime(&rawTime));
			s << "<h3>" << h << "</h3>";
			s << "<h4>#" << details.number;
			s << "&nbsp;&emsp;&nbsp;<b>" << timestamp << "</b></h4>";
			s << "<br/>D/TD: <b>2^" << log2((double)info.difficulty) << "</b>/<b>2^" << log2((double)details.totalDifficulty) << "</b>";
			s << "&nbsp;&emsp;&nbsp;Children: <b>" << details.children.size() << "</b></h5>";
			s << "<br/>Coinbase: <b>" << info.coinbaseAddress << "</b>";
			s << "<br/>State: <b>" << info.stateRoot << "</b>";
			s << "<br/>Nonce: <b>" << info.nonce << "</b>";
			s << "<br/>Transactions: <b>" << block[1].itemCount() << "</b> @<b>" << info.sha3Transactions << "</b>";
			s << "<br/>Uncles: <b>" << block[2].itemCount() << "</b> @<b>" << info.sha3Uncles << "</b>";
		}
		else
		{
			unsigned txi = item->data(Qt::UserRole + 1).toInt();
			Transaction tx(block[1][txi].data());
			h256 th = tx.sha3();
			s << "<h3>" << th << "</h3>";
			s << "<h4>" << h << "[<b>" << txi << "</b>]</h4>";
			s << "<br/>From: <b>" << tx.safeSender() << "</b>";
			if (tx.receiveAddress)
				s << "<br/>To: <b>" << tx.receiveAddress << "</b>";
			else
				s << "<br/>Creates: <b>" << right160(th) << "</b>";
			s << "<br/>Value: <b>" << formatBalance(tx.value) << "</b>";
			s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce << "</b>";
			if (tx.data.size())
			{
				s << "<br/>Data:&nbsp;&emsp;&nbsp;";
//				for (auto i: tx.data)
//					s << "0x<b>" << hex << i << "</b>&emsp;";
				s << "</br>" << disassemble(tx.data);
			}
		}


		ui->info->appendHtml(QString::fromStdString(s.str()));
	}
	m_client->unlock();
}

void Main::on_idealPeers_valueChanged()
{
	if (m_client->peerServer())
		m_client->peerServer()->setIdealPeerCount(ui->idealPeers->value());
}

void Main::on_ourAccounts_doubleClicked()
{
	auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(asHex(h.asArray())));
}

void Main::on_log_doubleClicked()
{
	qApp->clipboard()->setText(ui->log->currentItem()->text());
}

void Main::on_accounts_doubleClicked()
{
	auto hba = ui->accounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(asHex(h.asArray())));
}

void Main::on_destination_textChanged()
{
	updateFee();
}

void Main::on_data_textChanged()
{
	m_data = ui->data->toPlainText().split(QRegExp("[^0-9a-zA-Z]+"), QString::SkipEmptyParts);
	updateFee();
}

u256 Main::fee() const
{
	return (ui->destination->text().isEmpty() || !ui->destination->text().toInt()) ? m_client->state().fee(m_data.size()) : m_client->state().fee();
}

u256 Main::value() const
{
	return ui->value->value() * units()[units().size() - 1 - ui->valueUnits->currentIndex()].first;
}

u256 Main::total() const
{
	return value() + fee();
}

void Main::updateFee()
{
	ui->fee->setText(QString("(fee: %1)").arg(formatBalance(fee()).c_str()));
	auto totalReq = total();
	ui->total->setText(QString("Total: %1").arg(formatBalance(totalReq).c_str()));

	bool ok = false;
	for (auto i: m_myKeys)
		if (m_client->state().balance(i.address()) >= totalReq)
		{
			ok = true;
			break;
		}
	ui->send->setEnabled(ok);
	QPalette p = ui->total->palette();
	p.setColor(QPalette::WindowText, QColor(ok ? 0x00 : 0x80, 0x00, 0x00));
	ui->total->setPalette(p);
}

void Main::on_net_triggered()
{
	ui->port->setEnabled(!ui->net->isChecked());
	ui->clientName->setEnabled(!ui->net->isChecked());
	string n = "AlethZero/v" ADD_QUOTES(ETH_VERSION);
	if (ui->clientName->text().size())
		n += "/" + ui->clientName->text().toStdString();
	n +=  "/" ADD_QUOTES(ETH_BUILD_TYPE) "/" ADD_QUOTES(ETH_BUILD_PLATFORM);
	m_client->setClientVersion(n);
	if (ui->net->isChecked())
	{
		m_client->startNetwork(ui->port->value(), string(), 0, NodeMode::Full, ui->idealPeers->value(), std::string(), ui->upnp->isChecked());
		if (m_peers.size())
			m_client->peerServer()->restorePeers(bytesConstRef((byte*)m_peers.data(), m_peers.size()));
	}
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
		unsigned short port = s.section(":", 1).toInt();
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
	u256 totalReq = value() + fee();
	m_client->lock();
	for (auto i: m_myKeys)
		if (m_client->state().balance(i.address()) >= totalReq/* && i.address() != Address(fromUserHex(ui->destination->text().toStdString()))*/)
		{
			m_client->unlock();
			Secret s = i.secret();
			Address r = ui->destination->text().size() ? Address(fromUserHex(ui->destination->text().toStdString())) : Address();
			m_client->transact(s, r, value(), assemble(ui->data->toPlainText().toStdString()));
			refresh();
			return;
		}
	m_client->unlock();
	statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
}

void Main::on_create_triggered()
{
	m_myKeys.append(KeyPair::create());
	m_keysChanged = true;
}
