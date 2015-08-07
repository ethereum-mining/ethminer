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

#include <boost/algorithm/string.hpp>

#include <QtNetwork/QNetworkReply>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWebKitWidgets/QWebFrame>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <libserpent/funcs.h>
#include <libserpent/util.h>
#include <libdevcore/FileSystem.h>
#include <liblll/Compiler.h>
#include <liblll/CodeFragment.h>
#include <libevm/VM.h>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/ExtVM.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include <libwebthree/WebThree.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include "BuildInfo.h"
#include "MainWin.h"
#include "ui_Main.h"
using namespace std;
using namespace dev;
using namespace dev::eth;
using namespace dev::p2p;

static QString fromRaw(dev::h256 _n, unsigned* _inc = nullptr)
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

static std::vector<dev::KeyPair> keysAsVector(QList<dev::KeyPair> const& keys)
{
	auto list = keys.toStdList();
	return {std::begin(list), std::end(list)};
}

static QString contentsOfQResource(std::string const& res)
{
	QFile file(QString::fromStdString(res));
	if (!file.open(QFile::ReadOnly))
		return "";
	QTextStream in(&file);
	return in.readAll();
}

Address c_config = Address("661005d2720d855f1d9976f88bb10c1a3398c77f");

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);

	cerr << "State root: " << CanonBlockChain::genesis().stateRoot() << endl;
	auto gb = CanonBlockChain::createGenesisBlock();
	cerr << "Block Hash: " << sha3(gb) << endl;
	cerr << "Block RLP: " << RLP(gb) << endl;
	cerr << "Block Hex: " << toHex(gb) << endl;
	cerr << "Network protocol version: " << dev::eth::c_protocolVersion << endl;
	cerr << "Client database version: " << dev::eth::c_databaseVersion << endl;

	ui->ownedAccountsDock->hide();

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->mineStatus);
	statusBar()->addPermanentWidget(ui->blockCount);

	connect(ui->ourAccounts->model(), SIGNAL(rowsMoved(const QModelIndex &, int, int, const QModelIndex &, int)), SLOT(ourAccountsRowsMoved()));

	bytesConstRef networkConfig((byte*)m_networkConfig.data(), m_networkConfig.size());
	m_web3.reset(new WebThreeDirect("Third", getDataDir() + "/Third", false, {"eth", "shh"}, NetworkPreferences(), networkConfig));
	m_web3->connect(Host::pocHost());

	m_httpConnector.reset(new jsonrpc::HttpServer(8080));
	m_server.reset(new WebThreeStubServer(*m_httpConnector, *web3(), keysAsVector(m_myKeys)));
//	m_server = unique_ptr<WebThreeStubServer>(new WebThreeStubServer(m_httpConnector, *web3(), keysAsVector(m_myKeys)));
	m_server->setIdentities(keysAsVector(owned()));
	m_server->StartListening();

	connect(ui->webView, &QWebView::loadStarted, [this]()
	{
		QWebFrame* f = ui->webView->page()->mainFrame();
		f->disconnect(SIGNAL(javaScriptWindowObjectCleared()));
		connect(f, &QWebFrame::javaScriptWindowObjectCleared, [f, this]()
		{
			f->disconnect();
			f->addToJavaScriptWindowObject("env", this, QWebFrame::QtOwnership);
			f->evaluateJavaScript(contentsOfQResource(":/js/bignumber.min.js"));
			f->evaluateJavaScript(contentsOfQResource(":/js/webthree.js"));
			f->evaluateJavaScript(contentsOfQResource(":/js/setup.js"));
		});
	});

	connect(ui->webView, &QWebView::loadFinished, [=]()
	{
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
	writeSettings();
}

eth::Client* Main::ethereum() const
{
	return m_web3->ethereum();
}

std::shared_ptr<dev::shh::WhisperHost> Main::whisper() const
{
	return m_web3->whisper();
}

void Main::onKeysChanged()
{
	installBalancesWatch();
}

unsigned Main::installWatch(dev::eth::LogFilter const& _tf, WatchHandler const& _f)
{
	auto ret = ethereum()->installWatch(_tf);
	m_handlers[ret] = _f;
	return ret;
}

unsigned Main::installWatch(dev::h256 _tf, WatchHandler const& _f)
{
	auto ret = ethereum()->installWatch(_tf);
	m_handlers[ret] = _f;
	return ret;
}

void Main::installWatches()
{
	installWatch(dev::eth::LogFilter().address(c_config).topic(0, (u256)0), [=](LocalisedLogEntries const&){ installNameRegWatch(); });
	installWatch(dev::eth::LogFilter().address(c_config).topic(0, (u256)1), [=](LocalisedLogEntries const&){ installCurrenciesWatch(); });
	installWatch(dev::eth::ChainChangedFilter, [=](LocalisedLogEntries const&){ onNewBlock(); });
}

void Main::installNameRegWatch()
{
	ethereum()->uninstallWatch(m_nameRegFilter);
	m_nameRegFilter = installWatch(dev::eth::LogFilter().address(u160(ethereum()->stateAt(c_config, PendingBlock))), [=](LocalisedLogEntries const&){ onNameRegChange(); });
}

void Main::installCurrenciesWatch()
{
	ethereum()->uninstallWatch(m_currenciesFilter);
	m_currenciesFilter = installWatch(dev::eth::LogFilter().address(u160(ethereum()->stateAt(c_config, LatestBlock))), [=](LocalisedLogEntries const&){ onCurrenciesChange(); });
}

void Main::installBalancesWatch()
{
	dev::eth::LogFilter tf;

	vector<Address> altCoins;
	Address coinsAddr = right160(ethereum()->stateAt(c_config, LatestBlock));
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, PendingBlock); ++i)
		altCoins.push_back(right160(ethereum()->stateAt(coinsAddr, i + 1)));
	for (auto i: m_myKeys)
		for (auto c: altCoins)
			tf.address(c).topic(0, (u256)(u160)i.address());

	ethereum()->uninstallWatch(m_balancesFilter);
	m_balancesFilter = installWatch(tf, [=](LocalisedLogEntries const&){ onBalancesChange(); });
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

QString Main::pretty(dev::Address _a) const
{
	h256 n;

	if (h160 nameReg = (u160)ethereum()->stateAt(c_config, PendingBlock))
		n = ethereum()->stateAt(nameReg, (u160)(_a));

	return fromRaw(n);
}

QString Main::render(dev::Address _a) const
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
		if (h160 nameReg = (u160)ethereum()->stateAt(c_config, PendingBlock))
			if (h256 a = ethereum()->stateAt(nameReg, n))
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
	if (h160 dnsReg = (u160)ethereum()->stateAt(c_config, 4, PendingBlock))
		ret = ethereum()->stateAt(dnsReg, n);
/*	if (!ret)
		if (h160 nameReg = (u160)ethereum()->stateAt(c_config, 0, PendingBlock))
			ret = ethereum()->stateAt(nameReg, n2);
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
	QMessageBox::about(this, "About Third PoC-" + QString(dev::Version).section('.', 1, 1), QString("Third/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) "\n" DEV_QUOTED(ETH_COMMIT_HASH) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBy Gav Wood, 2014.\nThis software wouldn't be where it is today without the many leaders & contributors including:\n\nVitalik Buterin, Tim Hughes, caktux, Nick Savers, Eric Lombrozo, Marko Simovic, the many testers and the Berlin \304\220\316\236V team.");
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

	bytes d = m_web3->saveNetwork();
	if (d.size())
		m_networkConfig = QByteArray((char*)d.data(), (int)d.size());
	s.setValue("peers", m_networkConfig);

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
	ethereum()->setBeneficiary(m_myKeys.back().address());
	m_networkConfig = s.value("peers").toByteArray();
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
	dev::eth::MiningProgress p = ethereum()->miningProgress();
	ui->mineStatus->setText(ethereum()->isMining() ? QString("%1s @ %2kH/s").arg(p.ms / 1000).arg(p.ms ? p.hashes / p.ms : 0) : "Not mining");
}

void Main::refreshBalances()
{
	cwatch << "refreshBalances()";
	// update all the balance-dependent stuff.
	ui->ourAccounts->clear();
	u256 totalBalance = 0;
	map<Address, pair<QString, u256>> altCoins;
	Address coinsAddr = right160(ethereum()->stateAt(c_config, LatestBlock));
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, PendingBlock); ++i)
		altCoins[right160(ethereum()->stateAt(coinsAddr, ethereum()->stateAt(coinsAddr, i + 1)))] = make_pair(fromRaw(ethereum()->stateAt(coinsAddr, i + 1)), 0);
	for (auto i: m_myKeys)
	{
		u256 b = ethereum()->balanceAt(i.address());
		(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(b).c_str()).arg(render(i.address())).arg((unsigned)ethereum()->countAt(i.address())), ui->ourAccounts))
			->setData(Qt::UserRole, QByteArray((char const*)i.address().data(), Address::size));
		totalBalance += b;

		for (auto& c: altCoins)
			c.second.second += (u256)ethereum()->stateAt(c.first, (u160)i.address());
	}

	QString b;
	for (auto const& c: altCoins)
		if (c.second.second)
			b += QString::fromStdString(toString(c.second.second)) + " " + c.second.first.toUpper() + " | ";
	ui->balance->setText(b + QString::fromStdString(formatBalance(totalBalance)));
}

void Main::refreshNetwork()
{
	auto ps = m_web3->peers();

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
	auto d = ethereum()->blockChain().details();
	auto diff = BlockInfo(ethereum()->blockChain().block()).difficulty;
	ui->blockCount->setText(QString("#%1 @%3 T%2 N%4 D%5").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)).arg(dev::eth::c_protocolVersion).arg(dev::eth::c_databaseVersion));
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

	for (auto const& i: m_handlers)
	{
		auto ls = ethereum()->checkWatchSafe(i.first);
		if (ls.size())
			i.second(ls);
	}
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

	if (m_server.get())
		m_server->setAccounts(keysAsVector(myKeys));
}

void Main::on_ourAccounts_doubleClicked()
{
	auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::ensureNetwork()
{
	string n = string("Third/v") + dev::Version;
	n +=  "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM);
	web3()->setClientVersion(n);

	int pocnumber = QString(dev::Version).section('.', 1, 1).toInt();
	string defPeer;
	if (pocnumber == 5)
		defPeer = "54.72.69.180";
	else if (pocnumber == 6)
		defPeer = "54.76.56.74";

	if (!web3()->haveNetwork())
	{
		web3()->startNetwork();
		web3()->connect(defPeer);
	}
//	else
//		if (!m_web3->peerCount())
//			m_web3->connect(defPeer);
}

void Main::on_connect_triggered()
{
	bool ok = false;
	QString s = QInputDialog::getItem(this, "Connect to a Network Peer", "Enter a peer to which a connection may be made:", m_servers, m_servers.count() ? rand() % m_servers.count() : 0, true, &ok);
	if (ok && s.contains(":"))
	{
		string host = s.section(":", 0, 0).toStdString();
		unsigned short port = s.section(":", 1).toInt();
		web3()->connect(host, port);
	}
}

void Main::on_mine_triggered()
{
	if (ui->mine->isChecked())
	{
		ethereum()->setBeneficiary(m_myKeys.last().address());
		ethereum()->startMining();
	}
	else
		ethereum()->stopMining();
}
