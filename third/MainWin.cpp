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
/** @file MainWin.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <fstream>
#include <QtNetwork/QNetworkReply>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWebKitWidgets/QWebFrame>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <boost/algorithm/string.hpp>
#include <libserpent/funcs.h>
#include <libserpent/util.h>
#include <libethcore/Dagger.h>
#include <liblll/Compiler.h>
#include <liblll/CodeFragment.h>
#include <libevm/VM.h>
#include <libethereum/BlockChain.h>
#include <libethereum/ExtVM.h>
#include <libethereum/Client.h>
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
using eth::Address;
using eth::BlockInfo;
using eth::Client;
using eth::Instruction;
using eth::KeyPair;
using eth::NodeMode;
using eth::BlockChain;
using eth::PeerInfo;
using eth::RLP;
using eth::Secret;
using eth::Transaction;
using eth::Executive;

// functions
using eth::toHex;
using eth::compileLLL;
using eth::disassemble;
using eth::formatBalance;
using eth::fromHex;
using eth::sha3;
using eth::left160;
using eth::right160;
using eth::simpleDebugOut;
using eth::toLog2;
using eth::toString;
using eth::units;
using eth::operator<<;

// vars
using eth::g_logPost;
using eth::g_logVerbosity;
using eth::c_instructionInfo;

static QString fromRaw(eth::h256 _n, unsigned* _inc = nullptr)
{
	if (_n)
	{
		std::string s((char const*)_n.data(), 32);
		auto l = s.find_first_of('\0');
		if (!l)
			return QString();
		if (l != string::npos)
		{
			auto p = s.find_first_not_of('\0', l);
			if (!(p == string::npos || (_inc && p == 31)))
				return QString();
			if (_inc)
				*_inc = (byte)s[31];
			s.resize(l);
		}
		for (auto i: s)
			if (i < 32)
				return QString();
		return QString::fromStdString(s);
	}
	return QString();
}


Address c_config = Address("661005d2720d855f1d9976f88bb10c1a3398c77f");

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);

    cerr << "State root: " << BlockChain::genesis().stateRoot << endl;
    cerr << "Block Hash: " << sha3(BlockChain::createGenesisBlock()) << endl;
    cerr << "Block RLP: " << RLP(BlockChain::createGenesisBlock()) << endl;
    cerr << "Block Hex: " << toHex(BlockChain::createGenesisBlock()) << endl;
	cerr << "Network protocol version: " << eth::c_protocolVersion << endl;
	cerr << "Client database version: " << eth::c_databaseVersion << endl;

	ui->ownedAccountsDock->hide();

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->mineStatus);
	statusBar()->addPermanentWidget(ui->blockCount);
	
	connect(ui->ourAccounts->model(), SIGNAL(rowsMoved(const QModelIndex &, int, int, const QModelIndex &, int)), SLOT(ourAccountsRowsMoved()));

	m_client.reset(new Client("Third"));

	connect(ui->webView, &QWebView::loadStarted, [this]()
	{
		// NOTE: no need to delete as QETH_INSTALL_JS_NAMESPACE adopts it.
		m_ethereum = new QEthereum(this, m_client.get(), owned());

		QWebFrame* f = ui->webView->page()->mainFrame();
		f->disconnect(SIGNAL(javaScriptWindowObjectCleared()));
		auto qeth = m_ethereum;
		connect(f, &QWebFrame::javaScriptWindowObjectCleared, QETH_INSTALL_JS_NAMESPACE(f, qeth, this));
	});
	
	connect(ui->webView, &QWebView::loadFinished, [=]()
	{
		m_ethereum->poll();
	});
	
	connect(ui->webView, &QWebView::titleChanged, [=]()
	{
		ui->tabWidget->setTabText(0, ui->webView->title());
	});
	
	readSettings();

	installWatches();

	startTimer(100);

	{
		QSettings s("ethereum", "third");
		if (s.value("splashMessage", true).toBool())
		{
			QMessageBox::information(this, "Here Be Dragons!", "This is proof-of-concept software. The project as a whole is not even at the alpha-testing stage. It is here to show you, if you have a technical bent, the sort of thing that might be possible down the line.\nPlease don't blame us if it does something unexpected or if you're underwhelmed with the user-experience. We have great plans for it in terms of UX down the line but right now we just want to get the groundwork sorted. We welcome contributions, be they in code, testing or documentation!\nAfter you close this message it won't appear again.");
			s.setValue("splashMessage", false);
		}
	}
}

Main::~Main()
{
	// Must do this here since otherwise m_ethereum'll be deleted (and therefore clearWatches() called by the destructor)
	// *after* the client is dead.
	m_ethereum->clientDieing();

	writeSettings();
}

void Main::onKeysChanged()
{
	installBalancesWatch();
}

unsigned Main::installWatch(eth::MessageFilter const& _tf, std::function<void()> const& _f)
{
	auto ret = m_client->installWatch(_tf);
	m_handlers[ret] = _f;
	return ret;
}

unsigned Main::installWatch(eth::h256 _tf, std::function<void()> const& _f)
{
	auto ret = m_client->installWatch(_tf);
	m_handlers[ret] = _f;
	return ret;
}

void Main::installWatches()
{
	installWatch(eth::MessageFilter().altered(c_config, 0), [=](){ installNameRegWatch(); });
	installWatch(eth::MessageFilter().altered(c_config, 1), [=](){ installCurrenciesWatch(); });
	installWatch(eth::ChainChangedFilter, [=](){ onNewBlock(); });
}

void Main::installNameRegWatch()
{
	m_client->uninstallWatch(m_nameRegFilter);
	m_nameRegFilter = installWatch(eth::MessageFilter().altered((u160)m_client->stateAt(c_config, 0)), [=](){ onNameRegChange(); });
}

void Main::installCurrenciesWatch()
{
	m_client->uninstallWatch(m_currenciesFilter);
	m_currenciesFilter = installWatch(eth::MessageFilter().altered((u160)m_client->stateAt(c_config, 1)), [=](){ onCurrenciesChange(); });
}

void Main::installBalancesWatch()
{
	eth::MessageFilter tf;

	vector<Address> altCoins;
	Address coinsAddr = right160(m_client->stateAt(c_config, 1));
	for (unsigned i = 0; i < m_client->stateAt(coinsAddr, 0); ++i)
		altCoins.push_back(right160(m_client->stateAt(coinsAddr, i + 1)));
	for (auto i: m_myKeys)
	{
		tf.altered(i.address());
		for (auto c: altCoins)
			tf.altered(c, (u160)i.address());
	}

	m_client->uninstallWatch(m_balancesFilter);
	m_balancesFilter = installWatch(tf, [=](){ onBalancesChange(); });
}

void Main::onNameRegChange()
{
	cwatch << "NameReg changed!";

	// update any namereg-dependent stuff - for now force a full update.
	refreshAll();
}

void Main::onCurrenciesChange()
{
	cwatch << "Currencies changed!";
	installBalancesWatch();

	// TODO: update any currency-dependent stuff?
}

void Main::onBalancesChange()
{
	cwatch << "Our balances changed!";

	refreshBalances();
}

void Main::onNewBlock()
{
	cwatch << "Blockchain changed!";

	// update blockchain dependent views.
	refreshBlockCount();
}

void Main::note(QString _s)
{
	cnote << _s.toStdString();
}

void Main::debug(QString _s)
{
	cdebug << _s.toStdString();
}

void Main::warn(QString _s)
{
	cwarn << _s.toStdString();
}

void Main::eval(QString const& _js)
{
	if (_js.trimmed().isEmpty())
		return;
	ui->webView->page()->currentFrame()->evaluateJavaScript("___RET=(" + _js + ")");
}

QString Main::pretty(eth::Address _a) const
{
	h256 n;

	if (h160 nameReg = (u160)m_client->stateAt(c_config, 0))
		n = m_client->stateAt(nameReg, (u160)(_a));

	return fromRaw(n);
}

QString Main::render(eth::Address _a) const
{
	QString p = pretty(_a);
	if (!p.isNull())
		return p + " (" + QString::fromStdString(_a.abridged()) + ")";
	return QString::fromStdString(_a.abridged());
}

Address Main::fromString(QString const& _a) const
{
	if (_a == "(Create Contract)")
		return Address();

	string sn = _a.toStdString();
	if (sn.size() > 32)
		sn.resize(32);
	h256 n;
	memcpy(n.data(), sn.data(), sn.size());
	memset(n.data() + sn.size(), 0, 32 - sn.size());
	if (_a.size())
	{
		if (h160 nameReg = (u160)m_client->stateAt(c_config, 0))
			if (h256 a = m_client->stateAt(nameReg, n))
				return right160(a);
	}
	if (_a.size() == 40)
		return Address(fromHex(_a.toStdString()));
	else
		return Address();
}

QString Main::lookup(QString const& _a) const
{
	if (!_a.endsWith(".eth"))
		return _a;

	string sn = _a.mid(0, _a.size() - 4).toStdString();
	if (sn.size() > 32)
		sn = sha3(sn, false);
	h256 n;
	memcpy(n.data(), sn.data(), sn.size());

/*	string sn2 = _a.toStdString();
	if (sn2.size() > 32)
		sn2 = sha3(sn2, false);
	h256 n2;
	memcpy(n2.data(), sn2.data(), sn2.size());
*/

	h256 ret;
	if (h160 dnsReg = (u160)m_client->stateAt(c_config, 4, 0))
		ret = m_client->stateAt(dnsReg, n);
/*	if (!ret)
		if (h160 nameReg = (u160)m_client->stateAt(c_config, 0, 0))
			ret = m_client->stateAt(nameReg, n2);
*/
	if (ret && !((u256)ret >> 32))
		return QString("%1.%2.%3.%4").arg((int)ret[28]).arg((int)ret[29]).arg((int)ret[30]).arg((int)ret[31]);
	// TODO: support IPv6.
	else if (ret)
		return fromRaw(ret);
	else
		return _a;
}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About Third PoC-" + QString(eth::EthVersion).section('.', 1, 1), QString("Third/v") + eth::EthVersion + "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM) "\n" ETH_QUOTED(ETH_COMMIT_HASH) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBy Gav Wood, 2014.\nBased on a design by Vitalik Buterin.\n\nThanks to the various contributors including: Alex Leverington, Tim Hughes, caktux, Eric Lombrozo, Marko Simovic.");
}

void Main::writeSettings()
{
	QSettings s("ethereum", "third");
	QByteArray b;
	b.resize(sizeof(Secret) * m_myKeys.size());
	auto p = b.data();
	for (auto i: m_myKeys)
	{
		memcpy(p, &(i.secret()), sizeof(Secret));
		p += sizeof(Secret);
	}
	s.setValue("address", b);
	s.setValue("url", ui->urlEdit->text());

	bytes d = m_client->savePeers();
	if (d.size())
		m_peers = QByteArray((char*)d.data(), (int)d.size());
	s.setValue("peers", m_peers);

	s.setValue("geometry", saveGeometry());
	s.setValue("windowState", saveState());
}

void Main::readSettings(bool _skipGeometry)
{
	QSettings s("ethereum", "third");

	if (!_skipGeometry)
		restoreGeometry(s.value("geometry").toByteArray());
	restoreState(s.value("windowState").toByteArray());

	m_myKeys.clear();
	QByteArray b = s.value("address").toByteArray();
	if (b.isEmpty())
		m_myKeys.append(KeyPair::create());
	else
	{
		h256 k;
		for (unsigned i = 0; i < b.size() / sizeof(Secret); ++i)
		{
			memcpy(&k, b.data() + i * sizeof(Secret), sizeof(Secret));
			if (!count(m_myKeys.begin(), m_myKeys.end(), KeyPair(k)))
				m_myKeys.append(KeyPair(k));
		}
	}
	m_client->setAddress(m_myKeys.back().address());
	m_peers = s.value("peers").toByteArray();
	ui->urlEdit->setText(s.value("url", "about:blank").toString());	//http://gavwood.com/gavcoin.html
	on_urlEdit_returnPressed();
}

void Main::on_importKey_triggered()
{
	QString s = QInputDialog::getText(this, "Import Account Key", "Enter account's secret key");
	bytes b = fromHex(s.toStdString());
	if (b.size() == 32)
	{
		auto k = KeyPair(h256(b));
		if (std::find(m_myKeys.begin(), m_myKeys.end(), k) == m_myKeys.end())
		{
			m_myKeys.append(k);
			refreshBalances();
		}
		else
			QMessageBox::warning(this, "Already Have Key", "Could not import the secret key: we already own this account.");
	}
	else
		QMessageBox::warning(this, "Invalid Entry", "Could not import the secret key; invalid key entered. Make sure it is 64 hex characters (0-9 or A-F).");
}

void Main::on_exportKey_triggered()
{
	if (ui->ourAccounts->currentRow() >= 0 && ui->ourAccounts->currentRow() < m_myKeys.size())
	{
		auto k = m_myKeys[ui->ourAccounts->currentRow()];
		QMessageBox::information(this, "Export Account Key", "Secret key to account " + render(k.address()) + " is:\n" + QString::fromStdString(toHex(k.sec().ref())));
	}
}

void Main::on_urlEdit_returnPressed()
{
	QString s = ui->urlEdit->text();
	QRegExp r("([a-z]+://)?([^/]*)(.*)");
	if (r.exactMatch(s))
		if (r.cap(2).isEmpty())
			s = (r.cap(1).isEmpty() ? "file://" : r.cap(1)) + r.cap(3);
		else
			s = (r.cap(1).isEmpty() ? "http://" : r.cap(1)) + lookup(r.cap(2)) + r.cap(3);
	else{}
	qDebug() << s;
	ui->webView->setUrl(s);
}

void Main::refreshMining()
{
	eth::MineProgress p = m_client->miningProgress();
	ui->mineStatus->setText(m_client->isMining() ? QString("%1s @ %2kH/s").arg(p.ms / 1000).arg(p.ms ? p.hashes / p.ms : 0) : "Not mining");
}

void Main::refreshBalances()
{
	cwatch << "refreshBalances()";
	// update all the balance-dependent stuff.
	ui->ourAccounts->clear();
	u256 totalBalance = 0;
	map<Address, pair<QString, u256>> altCoins;
	Address coinsAddr = right160(m_client->stateAt(c_config, 1));
	for (unsigned i = 0; i < m_client->stateAt(coinsAddr, 0); ++i)
		altCoins[right160(m_client->stateAt(coinsAddr, m_client->stateAt(coinsAddr, i + 1)))] = make_pair(fromRaw(m_client->stateAt(coinsAddr, i + 1)), 0);
	for (auto i: m_myKeys)
	{
		u256 b = m_client->balanceAt(i.address());
		(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(b).c_str()).arg(render(i.address())).arg((unsigned)m_client->countAt(i.address())), ui->ourAccounts))
			->setData(Qt::UserRole, QByteArray((char const*)i.address().data(), Address::size));
		totalBalance += b;

		for (auto& c: altCoins)
			c.second.second += (u256)m_client->stateAt(c.first, (u160)i.address());
	}

	QString b;
	for (auto const& c: altCoins)
		if (c.second.second)
			b += QString::fromStdString(toString(c.second.second)) + " " + c.second.first.toUpper() + " | ";
	ui->balance->setText(b + QString::fromStdString(formatBalance(totalBalance)));
}

void Main::refreshNetwork()
{
	auto ps = m_client->peers();

	ui->peerCount->setText(QString::fromStdString(toString(ps.size())) + " peer(s)");
}

void Main::refreshAll()
{
	refreshBlockCount();
	refreshBalances();
}

void Main::refreshBlockCount()
{
	cwatch << "refreshBlockCount()";
	auto d = m_client->blockChain().details();
	auto diff = BlockInfo(m_client->blockChain().block()).difficulty;
	ui->blockCount->setText(QString("#%1 @%3 T%2 N%4 D%5").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)).arg(eth::c_protocolVersion).arg(eth::c_databaseVersion));
}

void Main::timerEvent(QTimerEvent*)
{
	// 7/18, Alex: aggregating timers, prelude to better threading?
	// Runs much faster on slower dual-core processors
	static int interval = 100;
	
	// refresh mining every 200ms
	if (interval / 100 % 2 == 0)
		refreshMining();

	// refresh peer list every 1000ms, reset counter
	if (interval == 1000)
	{
		interval = 0;
		ensureNetwork();
		refreshNetwork();
	}
	else
		interval += 100;
	
	if (m_ethereum)
		m_ethereum->poll();

	for (auto const& i: m_handlers)
		if (m_client->checkWatch(i.first))
			i.second();
}

void Main::ourAccountsRowsMoved()
{
	QList<KeyPair> myKeys;
	for (int i = 0; i < ui->ourAccounts->count(); ++i)
	{
		auto hba = ui->ourAccounts->item(i)->data(Qt::UserRole).toByteArray();
		auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		for (auto i: m_myKeys)
			if (i.address() == h)
				myKeys.push_back(i);
	}
	m_myKeys = myKeys;
	if (m_ethereum)
		m_ethereum->setAccounts(myKeys);
}

void Main::on_ourAccounts_doubleClicked()
{
	auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::ensureNetwork()
{
	string n = string("Third/v") + eth::EthVersion;
	n +=  "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM);
	m_client->setClientVersion(n);

	int pocnumber = QString(eth::EthVersion).section('.', 1, 1).toInt();
	string defPeer;
	if (pocnumber == 5)
		defPeer = "54.72.69.180";
	else if (pocnumber == 6)
		defPeer = "54.76.56.74";

	if (!m_client->haveNetwork())
		m_client->startNetwork(30303, defPeer);
	else
		if (!m_client->peerCount())
			m_client->connect(defPeer);
	if (m_peers.size())
		m_client->restorePeers(bytesConstRef((byte*)m_peers.data(), m_peers.size()));
}

void Main::on_connect_triggered()
{
	bool ok = false;
	QString s = QInputDialog::getItem(this, "Connect to a Network Peer", "Enter a peer to which a connection may be made:", m_servers, m_servers.count() ? rand() % m_servers.count() : 0, true, &ok);
	if (ok && s.contains(":"))
	{
		string host = s.section(":", 0, 0).toStdString();
		unsigned short port = s.section(":", 1).toInt();
		m_client->connect(host, port);
	}
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

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_MainWin.cpp"

#include\
"moc_MiningView.cpp"

#endif
