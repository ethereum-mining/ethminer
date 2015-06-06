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

// Make sure boost/asio.hpp is included before windows.h.
#include <boost/asio.hpp>

#pragma GCC diagnostic ignored "-Wpedantic"
//pragma GCC diagnostic ignored "-Werror=pedantic"
#include <QtNetwork/QNetworkReply>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QListWidgetItem>
#include <QtWebEngine/QtWebEngine>
#include <QtWebEngineWidgets/QWebEngineView>
#include <QtWebEngineWidgets/QWebEngineCallback>
#include <QtWebEngineWidgets/QWebEngineSettings>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <boost/algorithm/string.hpp>
#include <test/JsonSpiritHeaders.h>
#ifndef _MSC_VER
#include <libserpent/funcs.h>
#include <libserpent/util.h>
#endif
#include <libdevcore/FileSystem.h>
#include <libethcore/CommonJS.h>
#include <libethcore/EthashAux.h>
#include <libethcore/ICAP.h>
#include <liblll/Compiler.h>
#include <liblll/CodeFragment.h>
#include <libsolidity/Scanner.h>
#include <libsolidity/AST.h>
#include <libsolidity/SourceReferenceFormatter.h>
#include <libevm/VM.h>
#include <libevm/VMFactory.h>
#include <libethereum/CanonBlockChain.h>
#include <libethereum/ExtVM.h>
#include <libethereum/Client.h>
#include <libethereum/Utility.h>
#include <libethereum/EthereumHost.h>
#include <libethereum/DownloadMan.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <jsonrpccpp/server/connectors/httpserver.h>
#include "MainWin.h"
#include "DownloadView.h"
#include "MiningView.h"
#include "BuildInfo.h"
#include "OurWebThreeStubServer.h"
#include "Transact.h"
#include "Debugger.h"
#include "DappLoader.h"
#include "DappHost.h"
#include "WebPage.h"
#include "ExportState.h"
#include "ui_Main.h"
#include "ui_GetPassword.h"
#include "ui_GasPricing.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
namespace js = json_spirit;

string Main::fromRaw(h256 _n, unsigned* _inc)
{
	if (_n)
	{
		string s((char const*)_n.data(), 32);
		auto l = s.find_first_of('\0');
		if (!l)
			return string();
		if (l != string::npos)
		{
			auto p = s.find_first_not_of('\0', l);
			if (!(p == string::npos || (_inc && p == 31)))
				return string();
			if (_inc)
				*_inc = (byte)s[31];
			s.resize(l);
		}
		for (auto i: s)
			if (i < 32)
				return string();
		return s;
	}
	return string();
}

QString contentsOfQResource(string const& res)
{
	QFile file(QString::fromStdString(res));
	if (!file.open(QFile::ReadOnly))
		BOOST_THROW_EXCEPTION(FileError());
	QTextStream in(&file);
	return in.readAll();
}

//Address c_config = Address("661005d2720d855f1d9976f88bb10c1a3398c77f");
Address c_newConfig = Address("c6d9d2cd449a754c494264e1809c50e34d64562b");
//Address c_nameReg = Address("ddd1cea741d548f90d86fb87a3ae6492e18c03a1");

static QString filterOutTerminal(QString _s)
{
	return _s.replace(QRegExp("\x1b\\[(\\d;)?\\d+m"), "");
}

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main),
	m_transact(nullptr),
	m_dappLoader(nullptr),
	m_webPage(nullptr)
{
	QtWebEngine::initialize();
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	g_logPost = [=](string const& s, char const* c)
	{
		simpleDebugOut(s, c);
		m_logLock.lock();
		m_logHistory.append(filterOutTerminal(QString::fromStdString(s)) + "\n");
		m_logChanged = true;
		m_logLock.unlock();
//		ui->log->addItem(QString::fromStdString(s));
	};

	// Open Key Store
	bool opened = false;
	if (m_keyManager.exists())
		while (!opened)
		{
			QString s = QInputDialog::getText(nullptr, "Master password", "Enter your MASTER account password.", QLineEdit::Password, QString());
			if (m_keyManager.load(s.toStdString()))
				opened = true;
			else if (QMessageBox::question(
					nullptr,
					"Invalid password entered",
					"The password you entered is incorrect. If you have forgotten your password, and you wish to start afresh, manually remove the file: " + QString::fromStdString(getDataDir("ethereum")) + "/keys.info",
					QMessageBox::Retry,
					QMessageBox::Abort)
				== QMessageBox::Abort)
				exit(0);
		}
	if (!opened)
	{
		QString password;
		while (true)
		{
			password = QInputDialog::getText(nullptr, "Master password", "Enter a MASTER password for your key store. Make it strong. You probably want to write it down somewhere and keep it safe and secure; your identity will rely on this - you never want to lose it.", QLineEdit::Password, QString());
			QString confirm = QInputDialog::getText(nullptr, "Master password", "Confirm this password by typing it again", QLineEdit::Password, QString());
			if (password == confirm)
				break;
			QMessageBox::warning(nullptr, "Try again", "You entered two different passwords - please enter the same password twice.", QMessageBox::Ok);
		}
		m_keyManager.create(password.toStdString());
		m_keyManager.import(Secret::random(), "Default identity");
	}

#if ETH_DEBUG
	m_servers.append("127.0.0.1:30300");
#endif
	m_servers.append(QString::fromStdString(Host::pocHost() + ":30303"));

	cerr << "State root: " << CanonBlockChain::genesis().stateRoot << endl;
	auto block = CanonBlockChain::createGenesisBlock();
	cerr << "Block Hash: " << CanonBlockChain::genesis().hash() << endl;
	cerr << "Block RLP: " << RLP(block) << endl;
	cerr << "Block Hex: " << toHex(block) << endl;
	cerr << "eth Network protocol version: " << eth::c_protocolVersion << endl;
	cerr << "Client database version: " << c_databaseVersion << endl;

	ui->configDock->close();
	on_verbosity_valueChanged();

	statusBar()->addPermanentWidget(ui->cacheUsage);
	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->mineStatus);
	statusBar()->addPermanentWidget(ui->chainStatus);
	statusBar()->addPermanentWidget(ui->blockCount);

	ui->blockCount->setText(QString("PV%1.%2 D%3 %4-%5 v%6").arg(eth::c_protocolVersion).arg(eth::c_minorProtocolVersion).arg(c_databaseVersion).arg(QString::fromStdString(ProofOfWork::name())).arg(ProofOfWork::revision()).arg(dev::Version));

	QSettings s("ethereum", "alethzero");
	m_networkConfig = s.value("peers").toByteArray();
	bytesConstRef network((byte*)m_networkConfig.data(), m_networkConfig.size());
	m_webThree.reset(new WebThreeDirect(string("AlethZero/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM), getDataDir(), WithExisting::Trust, {"eth"/*, "shh"*/}, p2p::NetworkPreferences(), network));

	m_httpConnector.reset(new jsonrpc::HttpServer(SensibleHttpPort, "", "", dev::SensibleHttpThreads));
	m_server.reset(new OurWebThreeStubServer(*m_httpConnector, *web3(), this));
	connect(&*m_server, SIGNAL(onNewId(QString)), SLOT(addNewId(QString)));
	m_server->setIdentities(keysAsVector(owned()));
	m_server->StartListening();

	WebPage* webPage = new WebPage(this);
	m_webPage = webPage;
	connect(webPage, &WebPage::consoleMessage, [this](QString const& _msg) { Main::addConsoleMessage(_msg, QString()); });
	ui->webView->setPage(m_webPage);

	connect(ui->webView, &QWebEngineView::titleChanged, [=]()
	{
		ui->tabWidget->setTabText(0, ui->webView->title());
	});

	m_dappHost.reset(new DappHost(8081));
	m_dappLoader = new DappLoader(this, web3(), getNameReg());
	connect(m_dappLoader, &DappLoader::dappReady, this, &Main::dappLoaded);
	connect(m_dappLoader, &DappLoader::pageReady, this, &Main::pageLoaded);
//	ui->webView->page()->settings()->setAttribute(QWebEngineSettings::DeveloperExtrasEnabled, true);
//	QWebEngineInspector* inspector = new QWebEngineInspector();
//	inspector->setPage(page);
	setBeneficiary(*m_keyManager.accounts().begin());
	readSettings();

	m_transact = new Transact(this, this);
	m_transact->setWindowFlags(Qt::Dialog);
	m_transact->setWindowModality(Qt::WindowModal);

#if !ETH_FATDB
	removeDockWidget(ui->dockWidget_accounts);
#endif
	installWatches();
	startTimer(100);

	{
		QSettings s("ethereum", "alethzero");
		if (s.value("splashMessage", true).toBool())
		{
			QMessageBox::information(this, "Here Be Dragons!", "This is proof-of-concept software. The project as a whole is not even at the alpha-testing stage. It is here to show you, if you have a technical bent, the sort of thing that might be possible down the line.\nPlease don't blame us if it does something unexpected or if you're underwhelmed with the user-experience. We have great plans for it in terms of UX down the line but right now we just want to get the groundwork sorted. We welcome contributions, be they in code, testing or documentation!\nAfter you close this message it won't appear again.");
			s.setValue("splashMessage", false);
		}
	}
}

Main::~Main()
{
	m_httpConnector->StopListening();
	writeSettings();
	// Must do this here since otherwise m_ethereum'll be deleted (and therefore clearWatches() called by the destructor)
	// *after* the client is dead.
	g_logPost = simpleDebugOut;
}

bool Main::confirm() const
{
	return ui->natSpec->isChecked();
}

void Main::on_gasPrices_triggered()
{
	QDialog d;
	Ui_GasPricing gp;
	gp.setupUi(&d);
	d.setWindowTitle("Gas Pricing");
	setValueUnits(gp.bidUnits, gp.bidValue, static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->bid());
	setValueUnits(gp.askUnits, gp.askValue, static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->ask());

	if (d.exec() == QDialog::Accepted)
	{
		static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->setBid(fromValueUnits(gp.bidUnits, gp.bidValue));
		static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->setAsk(fromValueUnits(gp.askUnits, gp.askValue));
		m_transact->resetGasPrice();
	}
}

void Main::on_newIdentity_triggered()
{
	KeyPair kp = KeyPair::create();
	m_myIdentities.append(kp);
	m_server->setIdentities(keysAsVector(owned()));
	refreshWhisper();
}

void Main::refreshWhisper()
{
	ui->shhFrom->clear();
	for (auto i: m_server->ids())
		ui->shhFrom->addItem(QString::fromStdString(toHex(i.first.ref())));
}

void Main::addNewId(QString _ids)
{
	Secret _id = jsToSecret(_ids.toStdString());
	KeyPair kp(_id);
	m_myIdentities.push_back(kp);
	m_server->setIdentities(keysAsVector(owned()));
	refreshWhisper();
}

NetworkPreferences Main::netPrefs() const
{
	auto listenIP = ui->listenIP->text().toStdString();
	try
	{
		listenIP = bi::address::from_string(listenIP).to_string();
	}
	catch (...)
	{
		listenIP.clear();
	}

	auto publicIP = ui->forcePublicIP->text().toStdString();
	try
	{
		publicIP = bi::address::from_string(publicIP).to_string();
	}
	catch (...)
	{
		publicIP.clear();
	}

	if (isPublicAddress(publicIP))
		return NetworkPreferences(publicIP, listenIP, ui->port->value(), ui->upnp->isChecked());
	else
		return NetworkPreferences(listenIP, ui->port->value(), ui->upnp->isChecked());
}

void Main::onKeysChanged()
{
	installBalancesWatch();
}

unsigned Main::installWatch(LogFilter const& _tf, WatchHandler const& _f)
{
	auto ret = ethereum()->installWatch(_tf, Reaping::Manual);
	m_handlers[ret] = _f;
	_f(LocalisedLogEntries());
	return ret;
}

unsigned Main::installWatch(dev::h256 _tf, WatchHandler const& _f)
{
	auto ret = ethereum()->installWatch(_tf, Reaping::Manual);
	m_handlers[ret] = _f;
	_f(LocalisedLogEntries());
	return ret;
}

void Main::uninstallWatch(unsigned _w)
{
	cdebug << "!!! Main: uninstalling watch" << _w;
	ethereum()->uninstallWatch(_w);
	m_handlers.erase(_w);
}

void Main::installWatches()
{
	auto newBlockId = installWatch(ChainChangedFilter, [=](LocalisedLogEntries const&){
		onNewBlock();
	});
	auto newPendingId = installWatch(PendingChangedFilter, [=](LocalisedLogEntries const&){
		onNewPending();
	});

	cdebug << "newBlock watch ID: " << newBlockId;
	cdebug << "newPending watch ID: " << newPendingId;

	installWatch(LogFilter().address(c_newConfig), [=](LocalisedLogEntries const&) { installNameRegWatch(); });
	installWatch(LogFilter().address(c_newConfig), [=](LocalisedLogEntries const&) { installCurrenciesWatch(); });
}

Address Main::getNameReg() const
{
	return Address("c6d9d2cd449a754c494264e1809c50e34d64562b");
//	return abiOut<Address>(ethereum()->call(c_newConfig, abiIn("lookup(uint256)", (u256)1)).output);
}

Address Main::getCurrencies() const
{
	return abiOut<Address>(ethereum()->call(c_newConfig, abiIn("lookup(uint256)", (u256)3)).output);
}

bool Main::doConfirm()
{
	return ui->confirm->isChecked();
}

void Main::installNameRegWatch()
{
	uninstallWatch(m_nameRegFilter);
	m_nameRegFilter = installWatch(LogFilter().address((u160)getNameReg()), [=](LocalisedLogEntries const&){ onNameRegChange(); });
}

void Main::installCurrenciesWatch()
{
	uninstallWatch(m_currenciesFilter);
	m_currenciesFilter = installWatch(LogFilter().address((u160)getCurrencies()), [=](LocalisedLogEntries const&){ onCurrenciesChange(); });
}

void Main::installBalancesWatch()
{
	LogFilter tf;

	vector<Address> altCoins;
	Address coinsAddr = getCurrencies();

	// TODO: Update for new currencies reg.
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, PendingBlock); ++i)
		altCoins.push_back(right160(ethereum()->stateAt(coinsAddr, i + 1)));
	for (auto const& i: m_keyManager.accounts())
		for (auto c: altCoins)
			tf.address(c).topic(0, h256(i, h256::AlignRight));

	uninstallWatch(m_balancesFilter);
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
	refreshBlockChain();
	ui->refreshAccounts->setEnabled(true);

	// We must update balances since we can't filter updates to basic accounts.
	refreshBalances();
}

void Main::onNewPending()
{
	cwatch << "Pending transactions changed!";

	// update any pending-transaction dependent views.
	refreshPending();
	ui->refreshAccounts->setEnabled(true);
}

void Main::on_forceMining_triggered()
{
	ethereum()->setForceMining(ui->forceMining->isChecked());
}

QString Main::contents(QString _s)
{
	return QString::fromStdString(dev::asString(dev::contents(_s.toStdString())));
}

void Main::load(QString _s)
{
	QString contents = QString::fromStdString(dev::asString(dev::contents(_s.toStdString())));
	ui->webView->page()->runJavaScript(contents);
}

void Main::on_newTransaction_triggered()
{
	m_transact->setEnvironment(m_keyManager.accounts(), ethereum(), &m_natSpecDB);
	m_transact->show();
}

void Main::on_loadJS_triggered()
{
	QString f = QFileDialog::getOpenFileName(this, "Load Javascript", QString(), "Javascript (*.js);;All files (*)");
	if (f.size())
		load(f);
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

void Main::on_jsInput_returnPressed()
{
	eval(ui->jsInput->text());
	ui->jsInput->setText("");
}

void Main::eval(QString const& _js)
{
	if (_js.trimmed().isEmpty())
		return;
	auto f = [=](QVariant ev) {
		auto f2 = [=](QVariant jsonEv) {
			QString s;
			if (ev.isNull())
				s = "<span style=\"color: #888\">null</span>";
			else if (ev.type() == QVariant::String)
				s = "<span style=\"color: #444\">\"</span><span style=\"color: #c00\">" + ev.toString().toHtmlEscaped() + "</span><span style=\"color: #444\">\"</span>";
			else if (ev.type() == QVariant::Int || ev.type() == QVariant::Double)
				s = "<span style=\"color: #00c\">" + ev.toString().toHtmlEscaped() + "</span>";
			else if (jsonEv.type() == QVariant::String)
				s = "<span style=\"color: #840\">" + jsonEv.toString().toHtmlEscaped() + "</span>";
			else
				s = "<span style=\"color: #888\">unknown type</span>";
			addConsoleMessage(_js, s);
		};
		ui->webView->page()->runJavaScript("JSON.stringify(___RET)", f2);
	};
	auto c = (_js.startsWith("{") || _js.startsWith("if ") || _js.startsWith("if(")) ? _js : ("___RET=(" + _js + ")");
	ui->webView->page()->runJavaScript(c, f);
}

void Main::addConsoleMessage(QString const& _js, QString const& _s)
{
	m_consoleHistory.push_back(qMakePair(_js, _s));
	QString r = "<html><body style=\"margin: 0;\">" Div(Mono "position: absolute; bottom: 0; border: 0px; margin: 0px; width: 100%");
	for (auto const& i: m_consoleHistory)
		r +=	"<div style=\"border-bottom: 1 solid #eee; width: 100%\"><span style=\"float: left; width: 1em; color: #888; font-weight: bold\">&gt;</span><span style=\"color: #35d\">" + i.first.toHtmlEscaped() + "</span></div>"
				"<div style=\"border-bottom: 1 solid #eee; width: 100%\"><span style=\"float: left; width: 1em\">&nbsp;</span><span>" + i.second + "</span></div>";
	r += "</div></body></html>";
	ui->jsConsole->setHtml(r);
}

static Public stringToPublic(QString const& _a)
{
	string sn = _a.toStdString();
	if (_a.size() == sizeof(Public) * 2)
		return Public(fromHex(_a.toStdString()));
	else if (_a.size() == sizeof(Public) * 2 + 2 && _a.startsWith("0x"))
		return Public(fromHex(_a.mid(2).toStdString()));
	else
		return Public();
}

std::string Main::pretty(dev::Address const& _a) const
{
	auto g_newNameReg = getNameReg();

	if (g_newNameReg)
	{
		string n = toString(abiOut<string32>(ethereum()->call(g_newNameReg, abiIn("name(address)", _a)).output));
		if (!n.empty())
			return n;
	}
	return string();
}

std::string Main::render(dev::Address const& _a) const
{
	string p = pretty(_a);
	string n;
	if (p.size() == 9 && p.find_first_not_of("QWERYUOPASDFGHJKLZXCVBNM1234567890") == string::npos)
		p = ICAP(p, "XREG").encoded();
	else
		DEV_IGNORE_EXCEPTIONS(n = ICAP(_a).encoded());
	if (n.empty())
		n = _a.abridged();
	return p.empty() ? n : (p + " " + n);
}

pair<Address, bytes> Main::fromString(std::string const& _n) const
{
	if (_n == "(Create Contract)")
		return make_pair(Address(), bytes());

	auto g_newNameReg = getNameReg();
	if (g_newNameReg)
	{
		Address a = abiOut<Address>(ethereum()->call(g_newNameReg, abiIn("addr(bytes32)", ::toString32(_n))).output);
		if (a)
			return make_pair(a, bytes());
	}
	if (_n.size() == 40)
	{
		try
		{
			return make_pair(Address(fromHex(_n, WhenError::Throw)), bytes());
		}
		catch (BadHexCharacter& _e)
		{
			cwarn << "invalid hex character, address rejected";
			cwarn << boost::diagnostic_information(_e);
			return make_pair(Address(), bytes());
		}
		catch (...)
		{
			cwarn << "address rejected";
			return make_pair(Address(), bytes());
		}
	}
	else
		try {
			return ICAP::decoded(_n).address([&](Address const& a, bytes const& b) -> bytes
			{
				return ethereum()->call(a, b).output;
			}, g_newNameReg);
		}
		catch (...) {}
	return make_pair(Address(), bytes());
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

	h256 ret;
	// TODO: fix with the new DNSreg contract
//	if (h160 dnsReg = (u160)ethereum()->stateAt(c_config, 4, PendingBlock))
//		ret = ethereum()->stateAt(dnsReg, n);
/*	if (!ret)
		if (h160 nameReg = (u160)ethereum()->stateAt(c_config, 0, PendingBlock))
			ret = ethereum()->stateAt(nameReg, n2);
*/
	if (ret && !((u256)ret >> 32))
		return QString("%1.%2.%3.%4").arg((int)ret[28]).arg((int)ret[29]).arg((int)ret[30]).arg((int)ret[31]);
	// TODO: support IPv6.
	else if (ret)
		return QString::fromStdString(fromRaw(ret));
	else
		return _a;
}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About AlethZero PoC-" + QString(dev::Version).section('.', 1, 1), QString("AlethZero/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) "\n" DEV_QUOTED(ETH_COMMIT_HASH) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBerlin ÐΞV team, 2014.\nOriginally by Gav Wood. Based on a design by Vitalik Buterin.\n\nThanks to the various contributors including: Tim Hughes, caktux, Eric Lombrozo, Marko Simovic.");
}

void Main::on_paranoia_triggered()
{
	ethereum()->setParanoia(ui->paranoia->isChecked());
}

dev::u256 Main::gasPrice() const
{
	return ethereum()->gasPricer()->bid();
}

void Main::writeSettings()
{
	QSettings s("ethereum", "alethzero");
	s.remove("address");
	{
		QByteArray b;
		b.resize(sizeof(Secret) * m_myIdentities.size());
		auto p = b.data();
		for (auto i: m_myIdentities)
		{
			memcpy(p, &(i.secret()), sizeof(Secret));
			p += sizeof(Secret);
		}
		s.setValue("identities", b);
	}

	s.setValue("askPrice", QString::fromStdString(toString(static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->ask())));
	s.setValue("bidPrice", QString::fromStdString(toString(static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->bid())));
	s.setValue("upnp", ui->upnp->isChecked());
	s.setValue("forceAddress", ui->forcePublicIP->text());
	s.setValue("forceMining", ui->forceMining->isChecked());
	s.setValue("turboMining", ui->turboMining->isChecked());
	s.setValue("paranoia", ui->paranoia->isChecked());
	s.setValue("natSpec", ui->natSpec->isChecked());
	s.setValue("showAll", ui->showAll->isChecked());
	s.setValue("showAllAccounts", ui->showAllAccounts->isChecked());
	s.setValue("clientName", ui->clientName->text());
	s.setValue("idealPeers", ui->idealPeers->value());
	s.setValue("listenIP", ui->listenIP->text());
	s.setValue("port", ui->port->value());
	s.setValue("url", ui->urlEdit->text());
	s.setValue("privateChain", m_privateChain);
	s.setValue("verbosity", ui->verbosity->value());
	s.setValue("jitvm", ui->jitvm->isChecked());

	bytes d = m_webThree->saveNetwork();
	if (!d.empty())
		m_networkConfig = QByteArray((char*)d.data(), (int)d.size());
	s.setValue("peers", m_networkConfig);
	s.setValue("nameReg", ui->nameReg->text());

	s.setValue("geometry", saveGeometry());
	s.setValue("windowState", saveState());
}

Secret Main::retrieveSecret(Address const& _a) const
{
	auto info = m_keyManager.accountDetails()[_a];
	while (true)
	{
		Secret s = m_keyManager.secret(_a, [&](){
			QDialog d;
			Ui_GetPassword gp;
			gp.setupUi(&d);
			d.setWindowTitle("Unlock Account");
			gp.label->setText(QString("Enter the password for the account %2 (%1).").arg(QString::fromStdString(_a.abridged())).arg(QString::fromStdString(info.first)));
			gp.entry->setPlaceholderText("Hint: " + QString::fromStdString(info.second));
			return d.exec() == QDialog::Accepted ? gp.entry->text().toStdString() : string();
		});
		if (s || QMessageBox::warning(nullptr, "Unlock Account", "The password you gave is incorrect for this key.", QMessageBox::Retry, QMessageBox::Cancel) == QMessageBox::Cancel)
			return s;
	}
}

void Main::readSettings(bool _skipGeometry)
{
	QSettings s("ethereum", "alethzero");

	if (!_skipGeometry)
		restoreGeometry(s.value("geometry").toByteArray());
	restoreState(s.value("windowState").toByteArray());

	{
		QByteArray b = s.value("address").toByteArray();
		if (!b.isEmpty())
		{
			h256 k;
			for (unsigned i = 0; i < b.size() / sizeof(Secret); ++i)
			{
				memcpy(&k, b.data() + i * sizeof(Secret), sizeof(Secret));
				if (!m_keyManager.accounts().count(KeyPair(k).address()))
					m_keyManager.import(k, "Imported (UNSAFE) key.");
			}
		}
	}

	{
		m_myIdentities.clear();
		QByteArray b = s.value("identities").toByteArray();
		if (!b.isEmpty())
		{
			h256 k;
			for (unsigned i = 0; i < b.size() / sizeof(Secret); ++i)
			{
				memcpy(&k, b.data() + i * sizeof(Secret), sizeof(Secret));
				if (!count(m_myIdentities.begin(), m_myIdentities.end(), KeyPair(k)))
					m_myIdentities.append(KeyPair(k));
			}
		}
	}

	static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->setAsk(u256(s.value("askPrice", "500000000000").toString().toStdString()));
	static_cast<TrivialGasPricer*>(ethereum()->gasPricer().get())->setBid(u256(s.value("bidPrice", "500000000000").toString().toStdString()));

	ui->upnp->setChecked(s.value("upnp", true).toBool());
	ui->forcePublicIP->setText(s.value("forceAddress", "").toString());
	ui->dropPeers->setChecked(false);
	ui->forceMining->setChecked(s.value("forceMining", false).toBool());
	on_forceMining_triggered();
	ui->turboMining->setChecked(s.value("turboMining", false).toBool());
	on_turboMining_triggered();
	ui->paranoia->setChecked(s.value("paranoia", false).toBool());
	ui->natSpec->setChecked(s.value("natSpec", true).toBool());
	ui->showAll->setChecked(s.value("showAll", false).toBool());
	ui->showAllAccounts->setChecked(s.value("showAllAccounts", false).toBool());
	ui->clientName->setText(s.value("clientName", "").toString());
	if (ui->clientName->text().isEmpty())
		ui->clientName->setText(QInputDialog::getText(nullptr, "Enter identity", "Enter a name that will identify you on the peer network"));
	ui->idealPeers->setValue(s.value("idealPeers", ui->idealPeers->value()).toInt());
	ui->listenIP->setText(s.value("listenIP", "").toString());
	ui->port->setValue(s.value("port", ui->port->value()).toInt());
	ui->nameReg->setText(s.value("nameReg", "").toString());
	m_privateChain = s.value("privateChain", "").toString();
	ui->usePrivate->setChecked(m_privateChain.size());
	ui->verbosity->setValue(s.value("verbosity", 1).toInt());
	ui->jitvm->setChecked(s.value("jitvm", true).toBool());

	ui->urlEdit->setText(s.value("url", "about:blank").toString());	//http://gavwood.com/gavcoin.html
	on_urlEdit_returnPressed();
}

std::string Main::getPassword(std::string const& _title, std::string const& _for, std::string* _hint, bool* _ok)
{
	QString password;
	while (true)
	{
		bool ok;
		password = QInputDialog::getText(nullptr, QString::fromStdString(_title), QString::fromStdString(_for), QLineEdit::Password, QString(), &ok);
		if (!ok)
		{
			if (_ok)
				*_ok = false;
			return string();
		}
		if (password.isEmpty())
			break;
		QString confirm = QInputDialog::getText(nullptr, QString::fromStdString(_title), "Confirm this password by typing it again", QLineEdit::Password, QString());
		if (password == confirm)
			break;
		QMessageBox::warning(nullptr, QString::fromStdString(_title), "You entered two different passwords - please enter the same password twice.", QMessageBox::Ok);
	}

	if (!password.isEmpty() && _hint && !m_keyManager.haveHint(password.toStdString()))
		*_hint = QInputDialog::getText(this, "Create Account", "Enter a hint to help you remember this password.").toStdString();

	if (_ok)
		*_ok = true;
	return password.toStdString();
}

void Main::on_importKey_triggered()
{
	QString s = QInputDialog::getText(this, "Import Account Key", "Enter account's secret key", QLineEdit::Password);
	bytes b = fromHex(s.toStdString());
	if (b.size() == 32)
	{
		auto k = KeyPair(h256(b));
		if (!m_keyManager.accounts().count(k.address()))
		{
			QString s = QInputDialog::getText(this, "Import Account Key", "Enter this account's name");
			if (QMessageBox::question(this, "Additional Security?", "Would you like to use additional security for this key? This lets you protect it with a different password to other keys, but also means you must re-enter the key's password every time you wish to use the account.", QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes)
			{
				bool ok;
				std::string hint;
				std::string password = getPassword("Import Account Key", "Enter the password you would like to use for this key. Don't forget it!", &hint, &ok);
				if (!ok)
					return;
				m_keyManager.import(k.secret(), s.toStdString(), password, hint);
			}
			else
				m_keyManager.import(k.secret(), s.toStdString());
			keysChanged();
		}
		else
			QMessageBox::warning(this, "Already Have Key", "Could not import the secret key: we already own this account.");
	}
	else
		QMessageBox::warning(this, "Invalid Entry", "Could not import the secret key; invalid key entered. Make sure it is 64 hex characters (0-9 or A-F).");
}

void Main::on_importKeyFile_triggered()
{
	QString s = QFileDialog::getOpenFileName(this, "Claim Account Contents", QDir::homePath(), "JSON Files (*.json);;All Files (*)");
	h128 uuid = m_keyManager.store().importKey(s.toStdString());
	if (!uuid)
	{
		QMessageBox::warning(this, "Key File Invalid", "Could not find secret key definition. This is probably not an Web3 key file.");
		return;
	}

	QString info = QInputDialog::getText(this, "Import Key File", "Enter a description of this key to help you recognise it in the future.");

	QString pass;
	for (Secret s; !s;)
	{
		s = Secret(m_keyManager.store().secret(uuid, [&](){
			pass = QInputDialog::getText(this, "Import Key File", "Enter the password for the key to complete the import.", QLineEdit::Password);
			return pass.toStdString();
		}, false));
		if (!s && QMessageBox::question(this, "Import Key File", "The password you provided is incorrect. Would you like to try again?", QMessageBox::Retry, QMessageBox::Cancel) == QMessageBox::Cancel)
			return;
	}

	QString hint = QInputDialog::getText(this, "Import Key File", "Enter a hint for this password to help you remember it.");
	m_keyManager.importExisting(uuid, info.toStdString(), pass.toStdString(), hint.toStdString());
}

void Main::on_claimPresale_triggered()
{
	QString s = QFileDialog::getOpenFileName(this, "Claim Account Contents", QDir::homePath(), "JSON Files (*.json);;All Files (*)");
	try
	{
		js::mValue val;
		json_spirit::read_string(asString(dev::contents(s.toStdString())), val);
		auto obj = val.get_obj();
		if (obj["encseed"].type() == js::str_type)
		{
			auto encseed = fromHex(obj["encseed"].get_str());
			KeyPair k;
			for (bool gotit = false; !gotit;)
			{
				gotit = true;
				k = KeyPair::fromEncryptedSeed(&encseed, QInputDialog::getText(this, "Enter Password", "Enter the wallet's passphrase", QLineEdit::Password).toStdString());
				if (obj["ethaddr"].type() == js::str_type)
				{
					Address a(obj["ethaddr"].get_str());
					Address b = k.address();
					if (a != b)
					{
						if (QMessageBox::warning(this, "Password Wrong", "Could not import the secret key: the password you gave appears to be wrong.", QMessageBox::Retry, QMessageBox::Cancel) == QMessageBox::Cancel)
							return;
						else
							gotit = false;
					}
				}
			}

			cnote << k.address();
			if (!m_keyManager.accounts().count(k.address()))
				ethereum()->submitTransaction(k.sec(), ethereum()->balanceAt(k.address()) - gasPrice() * c_txGas, m_beneficiary, {}, c_txGas, gasPrice());
			else
				QMessageBox::warning(this, "Already Have Key", "Could not import the secret key: we already own this account.");
		}
		else
			BOOST_THROW_EXCEPTION(Exception() << errinfo_comment("encseed type is not js::str_type") );

	}
	catch (...)
	{
		cerr << "Unhandled exception!" << endl <<
			boost::current_exception_diagnostic_information();

		QMessageBox::warning(this, "Key File Invalid", "Could not find secret key definition. This is probably not an Ethereum key file.");
	}
}

void Main::on_exportKey_triggered()
{
	if (ui->ourAccounts->currentRow() >= 0)
	{
		auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
		Address h((byte const*)hba.data(), Address::ConstructFromPointer);
		Secret s = retrieveSecret(h);
		QMessageBox::information(this, "Export Account Key", "Secret key to account " + QString::fromStdString(render(h) + " is:\n" + s.hex()));
	}
}

void Main::on_exportState_triggered()
{
	ExportStateDialog dialog(this);
	dialog.exec();
}

void Main::on_usePrivate_triggered()
{
	if (ui->usePrivate->isChecked())
	{
		m_privateChain = QInputDialog::getText(this, "Enter Name", "Enter the name of your private chain", QLineEdit::Normal, QString("NewChain-%1").arg(time(0)));
		if (m_privateChain.isEmpty())
		{
			if (ui->usePrivate->isChecked())
				ui->usePrivate->setChecked(false);
			else
				// was cancelled.
				return;
		}
	}
	else
		m_privateChain.clear();
	on_killBlockchain_triggered();
}

void Main::on_jitvm_triggered()
{
	bool jit = ui->jitvm->isChecked();
	VMFactory::setKind(jit ? VMKind::JIT : VMKind::Interpreter);
}

void Main::on_urlEdit_returnPressed()
{
	QString s = ui->urlEdit->text();
	QUrl url(s);
	if (url.scheme().isEmpty() || url.scheme() == "eth" || url.path().endsWith(".dapp"))
	{
		try
		{
			//try do resolve dapp url
			m_dappLoader->loadDapp(s);
			return;
		}
		catch (...)
		{
			qWarning() << boost::current_exception_diagnostic_information().c_str();
		}
	}

	if (url.scheme().isEmpty())
		if (url.path().indexOf('/') < url.path().indexOf('.'))
			url.setScheme("file");
		else
			url.setScheme("http");
	else {}
	m_dappLoader->loadPage(url.toString());
}

void Main::on_nameReg_textChanged()
{
	string s = ui->nameReg->text().toStdString();
	if (s.size() == 40)
	{
		m_nameReg = Address(fromHex(s));
		refreshAll();
	}
	else
		m_nameReg = Address();
}

void Main::on_preview_triggered()
{
	ethereum()->setDefault(ui->preview->isChecked() ? PendingBlock : LatestBlock);
	refreshAll();
}

void Main::on_prepNextDAG_triggered()
{
	EthashAux::computeFull(
		EthashAux::seedHash(
			ethereum()->blockChain().number() + ETHASH_EPOCH_LENGTH
		)
	);
}

void Main::refreshMining()
{
	pair<uint64_t, unsigned> gp = EthashAux::fullGeneratingProgress();
	QString t;
	if (gp.first != EthashAux::NotGenerating)
		t = QString("DAG for #%1-#%2: %3% complete; ").arg(gp.first).arg(gp.first + ETHASH_EPOCH_LENGTH - 1).arg(gp.second);
	MiningProgress p = ethereum()->miningProgress();
	ui->mineStatus->setText(t + (ethereum()->isMining() ? p.hashes > 0 ? QString("%1s @ %2kH/s").arg(p.ms / 1000).arg(p.ms ? p.hashes / p.ms : 0) : "Awaiting DAG" : "Not mining"));
	if (ethereum()->isMining() && p.hashes > 0)
	{
		if (!ui->miningView->isVisible())
			return;
		list<MineInfo> l = ethereum()->miningHistory();
		static unsigned lh = 0;
		if (p.hashes < lh)
			ui->miningView->resetStats();
		lh = p.hashes;
		ui->miningView->appendStats(l, p);
	}
}

void Main::setBeneficiary(Address const& _b)
{
	for (int i = 0; i < ui->ourAccounts->count(); ++i)
	{
		auto hba = ui->ourAccounts->item(i)->data(Qt::UserRole).toByteArray();
		auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		ui->ourAccounts->item(i)->setCheckState(h == _b ? Qt::Checked : Qt::Unchecked);
	}
	m_beneficiary = _b;
	ethereum()->setAddress(_b);
}

void Main::on_ourAccounts_itemClicked(QListWidgetItem* _i)
{
	auto hba = _i->data(Qt::UserRole).toByteArray();
	setBeneficiary(Address((byte const*)hba.data(), Address::ConstructFromPointer));
}

void Main::refreshBalances()
{
	cwatch << "refreshBalances()";
	// update all the balance-dependent stuff.
	ui->ourAccounts->clear();
	u256 totalBalance = 0;
/*	map<Address, tuple<QString, u256, u256>> altCoins;
	Address coinsAddr = getCurrencies();
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, PendingBlock); ++i)
	{
		auto n = ethereum()->stateAt(coinsAddr, i + 1);
		auto addr = right160(ethereum()->stateAt(coinsAddr, n));
		auto denom = ethereum()->stateAt(coinsAddr, sha3(h256(n).asBytes()));
		if (denom == 0)
			denom = 1;
//		cdebug << n << addr << denom << sha3(h256(n).asBytes());
		altCoins[addr] = make_tuple(fromRaw(n), 0, denom);
	}*/
	for (pair<Address, std::pair<std::string, std::string>> const& i: m_keyManager.accountDetails())
	{
		u256 b = ethereum()->balanceAt(i.first);
		QListWidgetItem* li = new QListWidgetItem(QString("%4 %2: %1 [%3]").arg(formatBalance(b).c_str()).arg(QString::fromStdString(render(i.first))).arg((unsigned)ethereum()->countAt(i.first)).arg(QString::fromStdString(i.second.first)), ui->ourAccounts);
		li->setData(Qt::UserRole, QByteArray((char const*)i.first.data(), Address::size));
		li->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
		li->setCheckState(m_beneficiary == i.first ? Qt::Checked : Qt::Unchecked);
		totalBalance += b;

//		for (auto& c: altCoins)
//			get<1>(c.second) += (u256)ethereum()->stateAt(c.first, (u160)i.address());
	}

	QString b;
/*	for (auto const& c: altCoins)
		if (get<1>(c.second))
		{
			stringstream s;
			s << setw(toString(get<2>(c.second) - 1).size()) << setfill('0') << (get<1>(c.second) % get<2>(c.second));
			b += QString::fromStdString(toString(get<1>(c.second) / get<2>(c.second)) + "." + s.str() + " ") + get<0>(c.second).toUpper() + " | ";
		}*/
	ui->balance->setText(b + QString::fromStdString(formatBalance(totalBalance)));
}

void Main::refreshNetwork()
{
	auto ps = web3()->peers();

	ui->peerCount->setText(QString::fromStdString(toString(ps.size())) + " peer(s)");
	ui->peers->clear();
	ui->nodes->clear();

	if (web3()->haveNetwork())
	{
		map<h512, QString> sessions;
		for (PeerSessionInfo const& i: ps)
			ui->peers->addItem(QString("[%8 %7] %3 ms - %1:%2 - %4 %5 %6")
				.arg(QString::fromStdString(i.host))
				.arg(i.port)
				.arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count())
				.arg(sessions[i.id] = QString::fromStdString(i.clientVersion))
				.arg(QString::fromStdString(toString(i.caps)))
				.arg(QString::fromStdString(toString(i.notes)))
				.arg(i.socketId)
				.arg(QString::fromStdString(i.id.abridged())));

		auto ns = web3()->nodes();
		for (p2p::Peer const& i: ns)
			ui->nodes->insertItem(sessions.count(i.id) ? 0 : ui->nodes->count(), QString("[%1 %3] %2 - ( =%5s | /%4s%6 ) - *%7 $%8")
				.arg(QString::fromStdString(i.id.abridged()))
				.arg(QString::fromStdString(i.endpoint.address.to_string()))
				.arg(i.id == web3()->id() ? "self" : sessions.count(i.id) ? sessions[i.id] : "disconnected")
				.arg(i.isOffline() ? " | " + QString::fromStdString(reasonOf(i.lastDisconnect())) + " | " + QString::number(i.failedAttempts()) + "x" : "")
				.arg(i.rating())
				);
	}
}

void Main::refreshAll()
{
	refreshBlockChain();
	refreshBlockCount();
	refreshPending();
	ui->refreshAccounts->setEnabled(true);
	refreshBalances();
}

void Main::refreshPending()
{
	cwatch << "refreshPending()";
	ui->transactionQueue->clear();
	for (Transaction const& t: ethereum()->pending())
	{
		QString s = t.receiveAddress() ?
			QString("%2 %5> %3: %1 [%4]")
				.arg(formatBalance(t.value()).c_str())
				.arg(QString::fromStdString(render(t.safeSender())))
				.arg(QString::fromStdString(render(t.receiveAddress())))
				.arg((unsigned)t.nonce())
				.arg(ethereum()->codeAt(t.receiveAddress()).size() ? '*' : '-') :
			QString("%2 +> %3: %1 [%4]")
				.arg(formatBalance(t.value()).c_str())
				.arg(QString::fromStdString(render(t.safeSender())))
				.arg(QString::fromStdString(render(right160(sha3(rlpList(t.safeSender(), t.nonce()))))))
				.arg((unsigned)t.nonce());
		ui->transactionQueue->addItem(s);
	}
}

void Main::on_accountsFilter_textChanged()
{
	ui->refreshAccounts->setEnabled(true);
}

void Main::on_showBasic_toggled()
{
	ui->refreshAccounts->setEnabled(true);
}

void Main::on_showContracts_toggled()
{
	ui->refreshAccounts->setEnabled(true);
}

void Main::on_onlyNamed_toggled()
{
	ui->refreshAccounts->setEnabled(true);
}

void Main::on_refreshAccounts_clicked()
{
	refreshAccounts();
}

void Main::refreshAccounts()
{
	DEV_TIMED_FUNCTION;
#if ETH_FATDB || !ETH_TRUE
	cwatch << "refreshAccounts()";
	ui->accounts->clear();
	bool showContract = ui->showContracts->isChecked();
	bool showBasic = ui->showBasic->isChecked();
	bool onlyNamed = ui->onlyNamed->isChecked();
	for (auto const& i: ethereum()->addresses())
	{
		bool isContract = (ethereum()->codeHashAt(i) != EmptySHA3);
		if (!((showContract && isContract) || (showBasic && !isContract)))
			continue;
		string r = render(i);
		if (onlyNamed && !(r.find('"') != string::npos || r.substr(0, 2) == "XE"))
			continue;
		(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(ethereum()->balanceAt(i)).c_str()).arg(QString::fromStdString(r)).arg((unsigned)ethereum()->countAt(i)), ui->accounts))
			->setData(Qt::UserRole, QByteArray((char const*)i.data(), Address::size));
	}
#endif
	ui->refreshAccounts->setEnabled(false);
}

void Main::refreshBlockCount()
{
	auto d = ethereum()->blockChain().details();
	BlockQueueStatus b = ethereum()->blockQueueStatus();
	ui->chainStatus->setText(QString("%3 ready %4 verifying %5 unverified %6 future %7 unknown %8 bad  %1 #%2").arg(m_privateChain.size() ? "[" + m_privateChain + "] " : "testnet").arg(d.number).arg(b.verified).arg(b.verifying).arg(b.unverified).arg(b.future).arg(b.unknown).arg(b.bad));
}

void Main::on_turboMining_triggered()
{
	ethereum()->setTurboMining(ui->turboMining->isChecked());
}

void Main::refreshBlockChain()
{
	if (!ui->blocks->isVisible() && isVisible())
		return;

	DEV_TIMED_FUNCTION_ABOVE(500);
	cwatch << "refreshBlockChain()";

	// TODO: keep the same thing highlighted.
	// TODO: refactor into MVC
	// TODO: use get by hash/number
	// TODO: transactions

	auto const& bc = ethereum()->blockChain();
	QStringList filters = ui->blockChainFilter->text().toLower().split(QRegExp("\\s+"), QString::SkipEmptyParts);

	h256Hash blocks;
	for (QString f: filters)
		if (f.size() == 64)
		{
			h256 h(f.toStdString());
			if (bc.isKnown(h))
				blocks.insert(h);
			for (auto const& b: bc.withBlockBloom(LogBloom().shiftBloom<3>(sha3(h)), 0, -1))
				blocks.insert(bc.numberHash(b));
		}
		else if (f.toLongLong() <= bc.number())
			blocks.insert(bc.numberHash((unsigned)f.toLongLong()));
		else if (f.size() == 40)
		{
			Address h(f.toStdString());
			for (auto const& b: bc.withBlockBloom(LogBloom().shiftBloom<3>(sha3(h)), 0, -1))
				blocks.insert(bc.numberHash(b));
		}

	QByteArray oldSelected = ui->blocks->count() ? ui->blocks->currentItem()->data(Qt::UserRole).toByteArray() : QByteArray();
	ui->blocks->clear();
	auto showBlock = [&](h256 const& h) {
		auto d = bc.details(h);
		QListWidgetItem* blockItem = new QListWidgetItem(QString("#%1 %2").arg(d.number).arg(h.abridged().c_str()), ui->blocks);
		auto hba = QByteArray((char const*)h.data(), h.size);
		blockItem->setData(Qt::UserRole, hba);
		if (oldSelected == hba)
			blockItem->setSelected(true);

		int n = 0;
		try {
			auto b = bc.block(h);
			for (auto const& i: RLP(b)[1])
			{
				Transaction t(i.data(), CheckTransaction::Everything);
				QString s = t.receiveAddress() ?
					QString("    %2 %5> %3: %1 [%4]")
						.arg(formatBalance(t.value()).c_str())
						.arg(QString::fromStdString(render(t.safeSender())))
						.arg(QString::fromStdString(render(t.receiveAddress())))
						.arg((unsigned)t.nonce())
						.arg(ethereum()->codeAt(t.receiveAddress()).size() ? '*' : '-') :
					QString("    %2 +> %3: %1 [%4]")
						.arg(formatBalance(t.value()).c_str())
						.arg(QString::fromStdString(render(t.safeSender())))
						.arg(QString::fromStdString(render(right160(sha3(rlpList(t.safeSender(), t.nonce()))))))
						.arg((unsigned)t.nonce());
				QListWidgetItem* txItem = new QListWidgetItem(s, ui->blocks);
				auto hba = QByteArray((char const*)h.data(), h.size);
				txItem->setData(Qt::UserRole, hba);
				txItem->setData(Qt::UserRole + 1, n);
				if (oldSelected == hba)
					txItem->setSelected(true);
				n++;
			}
		}
		catch (...) {}
	};

	if (filters.empty())
	{
		unsigned i = ui->showAll->isChecked() ? (unsigned)-1 : 10;
		for (auto h = bc.currentHash(); bc.details(h) && i; h = bc.details(h).parent, --i)
		{
			showBlock(h);
			if (h == bc.genesisHash())
				break;
		}
	}
	else
		for (auto const& h: blocks)
			showBlock(h);

	if (!ui->blocks->currentItem())
		ui->blocks->setCurrentRow(0);
}

void Main::on_blockChainFilter_textChanged()
{
	static QTimer* s_delayed = nullptr;
	if (!s_delayed)
	{
		s_delayed = new QTimer(this);
		s_delayed->setSingleShot(true);
		connect(s_delayed, SIGNAL(timeout()), SLOT(refreshBlockChain()));
	}
	s_delayed->stop();
	s_delayed->start(200);
}

void Main::on_refresh_triggered()
{
	refreshAll();
}

static std::string niceUsed(unsigned _n)
{
	static const vector<std::string> c_units = { "bytes", "KB", "MB", "GB", "TB", "PB" };
	unsigned u = 0;
	while (_n > 10240)
	{
		_n /= 1024;
		u++;
	}
	if (_n > 1000)
		return toString(_n / 1000) + "." + toString((min<unsigned>(949, _n % 1000) + 50) / 100) + " " + c_units[u + 1];
	else
		return toString(_n) + " " + c_units[u];
}

void Main::refreshCache()
{
	BlockChain::Statistics s = ethereum()->blockChain().usage();
	QString t;
	auto f = [&](unsigned n, QString l)
	{
		t += ("%1 " + l).arg(QString::fromStdString(niceUsed(n)));
	};
	f(s.memTotal(), "total");
	t += " (";
	f(s.memBlocks, "blocks");
	t += ", ";
	f(s.memReceipts, "receipts");
	t += ", ";
	f(s.memLogBlooms, "blooms");
	t += ", ";
	f(s.memBlockHashes + s.memTransactionAddresses, "hashes");
	t += ", ";
	f(s.memDetails, "family");
	t += ")";
	ui->cacheUsage->setText(t);
}

void Main::timerEvent(QTimerEvent*)
{
	// 7/18, Alex: aggregating timers, prelude to better threading?
	// Runs much faster on slower dual-core processors
	static int interval = 100;

	// refresh mining every 200ms
	if (interval / 100 % 2 == 0)
		refreshMining();

	if ((interval / 100 % 2 == 0 && m_webThree->ethereum()->isSyncing()) || interval == 1000)
		ui->downloadView->update();

	if (m_logChanged)
	{
		m_logLock.lock();
		m_logChanged = false;
		ui->log->appendPlainText(m_logHistory.mid(0, m_logHistory.length() - 1));
		m_logHistory.clear();
		m_logLock.unlock();
	}

	// refresh peer list every 1000ms, reset counter
	if (interval == 1000)
	{
		interval = 0;
		refreshNetwork();
		refreshWhispers();
		refreshCache();
		refreshBlockCount();
		poll();
	}
	else
		interval += 100;

	for (auto const& i: m_handlers)
	{
		auto ls = ethereum()->checkWatchSafe(i.first);
		if (ls.size())
		{
//			cnote << "FIRING WATCH" << i.first << ls.size();
			i.second(ls);
		}
	}
}

string Main::renderDiff(StateDiff const& _d) const
{
	stringstream s;

	auto indent = "<code style=\"white-space: pre\">     </code>";
	for (auto const& i: _d.accounts)
	{
		s << "<hr/>";

		AccountDiff ad = i.second;
		s << "<code style=\"white-space: pre; font-weight: bold\">" << lead(ad.changeType()) << "  </code>" << " <b>" << render(i.first) << "</b>";
		if (!ad.exist.to())
			continue;

		if (ad.balance)
		{
			s << "<br/>" << indent << "Balance " << dec << ad.balance.to() << " [=" << formatBalance(ad.balance.to()) << "]";
			bigint d = (dev::bigint)ad.balance.to() - (dev::bigint)ad.balance.from();
			s << " <b>" << showpos << dec << d << " [=" << formatBalance(d) << "]" << noshowpos << "</b>";
		}
		if (ad.nonce)
		{
			s << "<br/>" << indent << "Count #" << dec << ad.nonce.to();
			s << " <b>" << showpos << (((dev::bigint)ad.nonce.to()) - ((dev::bigint)ad.nonce.from())) << noshowpos << "</b>";
		}
		if (ad.code)
		{
			s << "<br/>" << indent << "Code " << dec << ad.code.to().size() << " bytes";
			if (ad.code.from().size())
				 s << " (" << ad.code.from().size() << " bytes)";
		}

		for (pair<u256, dev::Diff<u256>> const& i: ad.storage)
		{
			s << "<br/><code style=\"white-space: pre\">";
			if (!i.second.from())
				s << " + ";
			else if (!i.second.to())
				s << "XXX";
			else
				s << " * ";
			s << "  </code>";

			s << prettyU256(i.first);
/*			if (i.first > u256(1) << 246)
				s << (h256)i.first;
			else if (i.first > u160(1) << 150)
				s << (h160)(u160)i.first;
			else
				s << hex << i.first;
*/
			if (!i.second.from())
				s << ": " << prettyU256(i.second.to());
			else if (!i.second.to())
				s << " (" << prettyU256(i.second.from()) << ")";
			else
				s << ": " << prettyU256(i.second.to()) << " (" << prettyU256(i.second.from()) << ")";
		}
	}
	return s.str();
}

void Main::on_transactionQueue_currentItemChanged()
{
	ui->pendingInfo->clear();

	stringstream s;
	int i = ui->transactionQueue->currentRow();
	if (i >= 0 && i < (int)ethereum()->pending().size())
	{
		Transaction tx(ethereum()->pending()[i]);
		TransactionReceipt receipt(ethereum()->postState().receipt(i));
		auto ss = tx.safeSender();
		h256 th = sha3(rlpList(ss, tx.nonce()));
		s << "<h3>" << th << "</h3>";
		s << "From: <b>" << pretty(ss) << "</b> " << ss;
		if (tx.isCreation())
			s << "<br/>Creates: <b>" << pretty(right160(th)) << "</b> " << right160(th);
		else
			s << "<br/>To: <b>" << pretty(tx.receiveAddress()) << "</b> " << tx.receiveAddress();
		s << "<br/>Value: <b>" << formatBalance(tx.value()) << "</b>";
		s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce() << "</b>";
		s << "<br/>Gas price: <b>" << formatBalance(tx.gasPrice()) << "</b>";
		s << "<br/>Gas: <b>" << tx.gas() << "</b>";
		if (tx.isCreation())
		{
			if (tx.data().size())
				s << "<h4>Code</h4>" << disassemble(tx.data());
		}
		else
		{
			if (tx.data().size())
				s << dev::memDump(tx.data(), 16, true);
		}
		s << "<div>Hex: " Span(Mono) << toHex(tx.rlp()) << "</span></div>";
		s << "<hr/>";
		if (!!receipt.bloom())
			s << "<div>Log Bloom: " << receipt.bloom() << "</div>";
		else
			s << "<div>Log Bloom: <b><i>Uneventful</i></b></div>";
		s << "<div>Gas Used: <b>" << receipt.gasUsed() << "</b></div>";
		s << "<div>End State: <b>" << receipt.stateRoot().abridged() << "</b></div>";
		auto r = receipt.rlp();
		s << "<div>Receipt: " << toString(RLP(r)) << "</div>";
		s << "<div>Receipt-Hex: " Span(Mono) << toHex(receipt.rlp()) << "</span></div>";
		s << renderDiff(ethereum()->diff(i, PendingBlock));
//		s << "Pre: " << fs.rootHash() << "<br/>";
//		s << "Post: <b>" << ts.rootHash() << "</b>";
	}

	ui->pendingInfo->setHtml(QString::fromStdString(s.str()));
	ui->pendingInfo->moveCursor(QTextCursor::Start);
}

void Main::on_inject_triggered()
{
	QString s = QInputDialog::getText(this, "Inject Transaction", "Enter transaction dump in hex");
	try
	{
		bytes b = fromHex(s.toStdString(), WhenError::Throw);
		ethereum()->injectTransaction(b);
	}
	catch (BadHexCharacter& _e)
	{
		cwarn << "invalid hex character, transaction rejected";
		cwarn << boost::diagnostic_information(_e);
	}
	catch (...)
	{
		cwarn << "transaction rejected";
	}
}

void Main::on_injectBlock_triggered()
{
	QString s = QInputDialog::getText(this, "Inject Block", "Enter block dump in hex");
	try
	{
		bytes b = fromHex(s.toStdString(), WhenError::Throw);
		ethereum()->injectBlock(b);
	}
	catch (BadHexCharacter& _e)
	{
		cwarn << "invalid hex character, transaction rejected";
		cwarn << boost::diagnostic_information(_e);
	}
	catch (...)
	{
		cwarn << "block rejected";
	}
}

static string htmlEscaped(string const& _s)
{
	return QString::fromStdString(_s).toHtmlEscaped().toStdString();
}

void Main::on_blocks_currentItemChanged()
{
	ui->info->clear();
	ui->debugCurrent->setEnabled(false);
	ui->debugDumpState->setEnabled(false);
	ui->debugDumpStatePre->setEnabled(false);
	if (auto item = ui->blocks->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 32);
		auto h = h256((byte const*)hba.data(), h256::ConstructFromPointer);
		auto details = ethereum()->blockChain().details(h);
		auto blockData = ethereum()->blockChain().block(h);
		auto block = RLP(blockData);
		BlockInfo info(blockData);

		stringstream s;

		if (item->data(Qt::UserRole + 1).isNull())
		{
			char timestamp[64];
			time_t rawTime = (time_t)(uint64_t)info.timestamp;
			strftime(timestamp, 64, "%c", localtime(&rawTime));
			s << "<h3>" << h << "</h3>";
			s << "<h4>#" << info.number;
			s << "&nbsp;&emsp;&nbsp;<b>" << timestamp << "</b></h4>";
			s << "<div>D/TD: <b>" << info.difficulty << "</b>/<b>" << details.totalDifficulty << "</b> = 2^" << log2((double)info.difficulty) << "/2^" << log2((double)details.totalDifficulty) << "</div>";
			s << "&nbsp;&emsp;&nbsp;Children: <b>" << details.children.size() << "</b></div>";
			s << "<div>Gas used/limit: <b>" << info.gasUsed << "</b>/<b>" << info.gasLimit << "</b>" << "</div>";
			s << "<div>Beneficiary: <b>" << htmlEscaped(pretty(info.coinbaseAddress)) << " " << info.coinbaseAddress << "</b>" << "</div>";
			s << "<div>Seed hash: <b>" << info.seedHash() << "</b>" << "</div>";
			s << "<div>Mix hash: <b>" << info.mixHash << "</b>" << "</div>";
			s << "<div>Nonce: <b>" << info.nonce << "</b>" << "</div>";
			s << "<div>Hash w/o nonce: <b>" << info.headerHash(WithoutNonce) << "</b>" << "</div>";
			s << "<div>Difficulty: <b>" << info.difficulty << "</b>" << "</div>";
			if (info.number)
			{
				auto e = EthashAux::eval(info);
				s << "<div>Proof-of-Work: <b>" << e.value << " &lt;= " << (h256)u256((bigint(1) << 256) / info.difficulty) << "</b> (mixhash: " << e.mixHash.abridged() << ")" << "</div>";
				s << "<div>Parent: <b>" << info.parentHash << "</b>" << "</div>";
			}
			else
			{
				s << "<div>Proof-of-Work: <b><i>Phil has nothing to prove</i></b></div>";
				s << "<div>Parent: <b><i>It was a virgin birth</i></b></div>";
			}
//			s << "<div>Bloom: <b>" << details.bloom << "</b>";
			if (!!info.logBloom)
				s << "<div>Log Bloom: " << info.logBloom << "</div>";
			else
				s << "<div>Log Bloom: <b><i>Uneventful</i></b></div>";
			s << "<div>Transactions: <b>" << block[1].itemCount() << "</b> @<b>" << info.transactionsRoot << "</b>" << "</div>";
			s << "<div>Uncles: <b>" << block[2].itemCount() << "</b> @<b>" << info.sha3Uncles << "</b>" << "</div>";
			for (auto u: block[2])
			{
				BlockInfo uncle = BlockInfo::fromHeader(u.data());
				char const* line = "<div><span style=\"margin-left: 2em\">&nbsp;</span>";
				s << line << "Hash: <b>" << uncle.hash() << "</b>" << "</div>";
				s << line << "Parent: <b>" << uncle.parentHash << "</b>" << "</div>";
				s << line << "Number: <b>" << uncle.number << "</b>" << "</div>";
				s << line << "Coinbase: <b>" << htmlEscaped(pretty(uncle.coinbaseAddress)) << " " << uncle.coinbaseAddress << "</b>" << "</div>";
				s << line << "Seed hash: <b>" << uncle.seedHash() << "</b>" << "</div>";
				s << line << "Mix hash: <b>" << uncle.mixHash << "</b>" << "</div>";
				s << line << "Nonce: <b>" << uncle.nonce << "</b>" << "</div>";
				s << line << "Hash w/o nonce: <b>" << uncle.headerHash(WithoutNonce) << "</b>" << "</div>";
				s << line << "Difficulty: <b>" << uncle.difficulty << "</b>" << "</div>";
				auto e = EthashAux::eval(uncle);
				s << line << "Proof-of-Work: <b>" << e.value << " &lt;= " << (h256)u256((bigint(1) << 256) / uncle.difficulty) << "</b> (mixhash: " << e.mixHash.abridged() << ")" << "</div>";
			}
			if (info.parentHash)
				s << "<div>Pre: <b>" << BlockInfo(ethereum()->blockChain().block(info.parentHash)).stateRoot << "</b>" << "</div>";
			else
				s << "<div>Pre: <b><i>Nothing is before Phil</i></b>" << "</div>";

			s << "<div>Receipts: @<b>" << info.receiptsRoot << "</b>:" << "</div>";
			BlockReceipts receipts = ethereum()->blockChain().receipts(h);
			unsigned ii = 0;
			for (auto const& i: block[1])
			{
				s << "<div>" << sha3(i.data()).abridged() << ": <b>" << receipts.receipts[ii].stateRoot() << "</b> [<b>" << receipts.receipts[ii].gasUsed() << "</b> used]" << "</div>";
				++ii;
			}
			s << "<div>Post: <b>" << info.stateRoot << "</b>" << "</div>";
			s << "<div>Dump: " Span(Mono) << toHex(block[0].data()) << "</span>" << "</div>";
			s << "<div>Receipts-Hex: " Span(Mono) << toHex(receipts.rlp()) << "</span></div>";
		}
		else
		{
			unsigned txi = item->data(Qt::UserRole + 1).toInt();
			Transaction tx(block[1][txi].data(), CheckTransaction::Everything);
			auto ss = tx.safeSender();
			h256 th = sha3(rlpList(ss, tx.nonce()));
			TransactionReceipt receipt = ethereum()->blockChain().receipts(h).receipts[txi];
			s << "<h3>" << th << "</h3>";
			s << "<h4>" << h << "[<b>" << txi << "</b>]</h4>";
			s << "<div>From: <b>" << htmlEscaped(pretty(ss)) << " " << ss << "</b>" << "</div>";
			if (tx.isCreation())
				s << "<div>Creates: <b>" << htmlEscaped(pretty(right160(th))) << "</b> " << right160(th) << "</div>";
			else
				s << "<div>To: <b>" << htmlEscaped(pretty(tx.receiveAddress())) << "</b> " << tx.receiveAddress() << "</div>";
			s << "<div>Value: <b>" << formatBalance(tx.value()) << "</b>" << "</div>";
			s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce() << "</b>" << "</div>";
			s << "<div>Gas price: <b>" << formatBalance(tx.gasPrice()) << "</b>" << "</div>";
			s << "<div>Gas: <b>" << tx.gas() << "</b>" << "</div>";
			s << "<div>V: <b>" << hex << nouppercase << (int)tx.signature().v << " + 27</b>" << "</div>";
			s << "<div>R: <b>" << hex << nouppercase << tx.signature().r << "</b>" << "</div>";
			s << "<div>S: <b>" << hex << nouppercase << tx.signature().s << "</b>" << "</div>";
			s << "<div>Msg: <b>" << tx.sha3(eth::WithoutSignature) << "</b>" << "</div>";
			if (!tx.data().empty())
			{
				if (tx.isCreation())
					s << "<h4>Code</h4>" << disassemble(tx.data());
				else
					s << "<h4>Data</h4>" << dev::memDump(tx.data(), 16, true);
			}
			s << "<div>Hex: " Span(Mono) << toHex(block[1][txi].data()) << "</span></div>";
			s << "<hr/>";
			if (!!receipt.bloom())
				s << "<div>Log Bloom: " << receipt.bloom() << "</div>";
			else
				s << "<div>Log Bloom: <b><i>Uneventful</i></b></div>";
			s << "<div>Gas Used: <b>" << receipt.gasUsed() << "</b></div>";
			s << "<div>End State: <b>" << receipt.stateRoot().abridged() << "</b></div>";
			auto r = receipt.rlp();
			s << "<div>Receipt: " << toString(RLP(r)) << "</div>";
			s << "<div>Receipt-Hex: " Span(Mono) << toHex(receipt.rlp()) << "</span></div>";
			s << "<h4>Diff</h4>" << renderDiff(ethereum()->diff(txi, h));
			ui->debugCurrent->setEnabled(true);
			ui->debugDumpState->setEnabled(true);
			ui->debugDumpStatePre->setEnabled(true);
		}

		ui->info->appendHtml(QString::fromStdString(s.str()));
		ui->info->moveCursor(QTextCursor::Start);
	}
}

void Main::on_debugCurrent_triggered()
{
	if (auto item = ui->blocks->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 32);
		auto h = h256((byte const*)hba.data(), h256::ConstructFromPointer);

		if (!item->data(Qt::UserRole + 1).isNull())
		{
			unsigned txi = item->data(Qt::UserRole + 1).toInt();
			bytes t = ethereum()->blockChain().transaction(h, txi);
			State s(ethereum()->state(txi, h));
			Executive e(s, ethereum()->blockChain());
			Debugger dw(this, this);
			dw.populate(e, Transaction(t, CheckTransaction::Everything));
			dw.exec();
		}
	}
}

void Main::debugDumpState(int _add)
{
	if (auto item = ui->blocks->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 32);
		auto h = h256((byte const*)hba.data(), h256::ConstructFromPointer);

		if (!item->data(Qt::UserRole + 1).isNull())
		{
			QString fn = QFileDialog::getSaveFileName(this, "Select file to output state dump");
			ofstream f(fn.toStdString());
			if (f.is_open())
			{
				unsigned txi = item->data(Qt::UserRole + 1).toInt();
				f << ethereum()->state(txi + _add, h) << endl;
			}
		}
	}
}

void Main::on_accounts_currentItemChanged()
{
	ui->accountInfo->clear();
	if (auto item = ui->accounts->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 20);
		auto address = h160((byte const*)hba.data(), h160::ConstructFromPointer);

		stringstream s;
		try
		{
			auto storage = ethereum()->storageAt(address);
			for (auto const& i: storage)
				s << "@" << showbase << hex << prettyU256(i.first) << "&nbsp;&nbsp;&nbsp;&nbsp;" << showbase << hex << prettyU256(i.second) << "<br/>";
			s << "<h4>Body Code (" << sha3(ethereum()->codeAt(address)).abridged() << ")</h4>" << disassemble(ethereum()->codeAt(address));
			s << Div(Mono) << toHex(ethereum()->codeAt(address)) << "</div>";
			ui->accountInfo->appendHtml(QString::fromStdString(s.str()));
		}
		catch (dev::InvalidTrie)
		{
			ui->accountInfo->appendHtml("Corrupted trie.");
		}
		ui->accountInfo->moveCursor(QTextCursor::Start);
	}
}

void Main::on_idealPeers_valueChanged(int)
{
	m_webThree->setIdealPeerCount(ui->idealPeers->value());
}

void Main::on_ourAccounts_doubleClicked()
{
	auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

/*void Main::on_log_doubleClicked()
{
	ui->log->setPlainText("");
	m_logHistory.clear();
}*/

void Main::on_accounts_doubleClicked()
{
	if (ui->accounts->count())
	{
		auto hba = ui->accounts->currentItem()->data(Qt::UserRole).toByteArray();
		auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
	}
}

static shh::Topics topicFromText(QString _s)
{
	shh::BuildTopic ret;
	while (_s.size())
	{
		QRegExp r("(@|\\$)?\"([^\"]*)\"(\\s.*)?");
		QRegExp d("(@|\\$)?([0-9]+)(\\s*(ether)|(finney)|(szabo))?(\\s.*)?");
		QRegExp h("(@|\\$)?(0x)?(([a-fA-F0-9])+)(\\s.*)?");
		bytes part;
		if (r.exactMatch(_s))
		{
			for (auto i: r.cap(2))
				part.push_back((byte)i.toLatin1());
			if (r.cap(1) != "$")
				for (int i = r.cap(2).size(); i < 32; ++i)
					part.push_back(0);
			else
				part.push_back(0);
			_s = r.cap(3);
		}
		else if (d.exactMatch(_s))
		{
			u256 v(d.cap(2).toStdString());
			if (d.cap(6) == "szabo")
				v *= szabo;
			else if (d.cap(5) == "finney")
				v *= finney;
			else if (d.cap(4) == "ether")
				v *= ether;
			bytes bs = dev::toCompactBigEndian(v);
			if (d.cap(1) != "$")
				for (auto i = bs.size(); i < 32; ++i)
					part.push_back(0);
			for (auto b: bs)
				part.push_back(b);
			_s = d.cap(7);
		}
		else if (h.exactMatch(_s))
		{
			bytes bs = fromHex((((h.cap(3).size() & 1) ? "0" : "") + h.cap(3)).toStdString());
			if (h.cap(1) != "$")
				for (auto i = bs.size(); i < 32; ++i)
					part.push_back(0);
			for (auto b: bs)
				part.push_back(b);
			_s = h.cap(5);
		}
		else
			_s = _s.mid(1);
		ret.shift(part);
	}
	return ret;
}

void Main::on_clearPending_triggered()
{
	writeSettings();
	ui->mine->setChecked(false);
	ui->net->setChecked(false);
	web3()->stopNetwork();
	ethereum()->clearPending();
	readSettings(true);
	installWatches();
	refreshAll();
}

void Main::on_retryUnknown_triggered()
{
	ethereum()->retryUnkonwn();
}

void Main::on_killBlockchain_triggered()
{
	writeSettings();
	ui->mine->setChecked(false);
	ui->net->setChecked(false);
	web3()->stopNetwork();
	ethereum()->killChain();
	readSettings(true);
	installWatches();
	refreshAll();
}

void Main::on_net_triggered()
{
	ui->port->setEnabled(!ui->net->isChecked());
	ui->clientName->setEnabled(!ui->net->isChecked());
	string n = string("AlethZero/v") + dev::Version;
	if (ui->clientName->text().size())
		n += "/" + ui->clientName->text().toStdString();
	n +=  "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM);
	web3()->setClientVersion(n);
	if (ui->net->isChecked())
	{
		web3()->setIdealPeerCount(ui->idealPeers->value());
		web3()->setNetworkPreferences(netPrefs(), ui->dropPeers->isChecked());
		ethereum()->setNetworkId(m_privateChain.size() ? sha3(m_privateChain.toStdString()) : h256());
		web3()->startNetwork();
		ui->downloadView->setDownloadMan(ethereum()->downloadMan());
		ui->enode->setText(QString::fromStdString(web3()->enode()));
	}
	else
	{
		ui->downloadView->setDownloadMan(nullptr);
		writeSettings();
		web3()->stopNetwork();
	}
}

void Main::on_connect_triggered()
{
	if (!ui->net->isChecked())
	{
		ui->net->setChecked(true);
		on_net_triggered();
	}

	m_connect.setEnvironment(m_servers);
	if (m_connect.exec() == QDialog::Accepted)
	{
		bool required = m_connect.required();
		string host(m_connect.host().toStdString());
		NodeId nodeID;
		try
		{
			nodeID = NodeId(fromHex(m_connect.nodeId().toStdString()));
		}
		catch (BadHexCharacter&) {}

		m_connect.reset();

		if (required)
			web3()->requirePeer(nodeID, host);
		else
			web3()->addNode(nodeID, host);
	}
}

void Main::on_verbosity_valueChanged()
{
	g_logVerbosity = ui->verbosity->value();
	ui->verbosityLabel->setText(QString::number(g_logVerbosity));
}

void Main::on_mine_triggered()
{
	if (ui->mine->isChecked())
	{
//		EthashAux::computeFull(ethereum()->blockChain().number());
		ethereum()->setAddress(m_beneficiary);
		ethereum()->startMining();
	}
	else
		ethereum()->stopMining();
}

void Main::keysChanged()
{
	onBalancesChange();
}

bool beginsWith(Address _a, bytes const& _b)
{
	for (unsigned i = 0; i < min<unsigned>(20, _b.size()); ++i)
		if (_a[i] != _b[i])
			return false;
	return true;
}

void Main::on_newAccount_triggered()
{
	bool ok = true;
	enum { NoVanity = 0, DirectICAP, FirstTwo, FirstTwoNextTwo, FirstThree, FirstFour, StringMatch };
	QStringList items = {"No vanity (instant)", "Direct ICAP address", "Two pairs first (a few seconds)", "Two pairs first and second (a few minutes)", "Three pairs first (a few minutes)", "Four pairs first (several hours)", "Specific hex string"};
	unsigned v = items.QList<QString>::indexOf(QInputDialog::getItem(this, "Vanity Key?", "Would you a vanity key? This could take several hours.", items, 1, false, &ok));
	if (!ok)
		return;

	bytes bs;
	if (v == StringMatch)
	{
		QString s = QInputDialog::getText(this, "Vanity Beginning?", "Enter some hex digits that it should begin with.\nNOTE: The more you enter, the longer generation will take.", QLineEdit::Normal, QString(), &ok);
		if (!ok)
			return;
		bs = fromHex(s.toStdString());
	}

	KeyPair p;
	bool keepGoing = true;
	unsigned done = 0;
	function<void()> f = [&]() {
		KeyPair lp;
		while (keepGoing)
		{
			done++;
			if (done % 1000 == 0)
				cnote << "Tried" << done << "keys";
			lp = KeyPair::create();
			auto a = lp.address();
			if (v == NoVanity ||
				(v == DirectICAP && !a[0]) ||
				(v == FirstTwo && a[0] == a[1]) ||
				(v == FirstTwoNextTwo && a[0] == a[1] && a[2] == a[3]) ||
				(v == FirstThree && a[0] == a[1] && a[1] == a[2]) ||
				(v == FirstFour && a[0] == a[1] && a[1] == a[2] && a[2] == a[3]) ||
				(v == StringMatch && beginsWith(lp.address(), bs))
			)
				break;
		}
		if (keepGoing)
			p = lp;
		keepGoing = false;
	};
	vector<std::thread*> ts;
	for (unsigned t = 0; t < std::thread::hardware_concurrency() - 1; ++t)
		ts.push_back(new std::thread(f));
	f();
	for (std::thread* t: ts)
	{
		t->join();
		delete t;
	}

	QString s = QInputDialog::getText(this, "Create Account", "Enter this account's name");
	if (QMessageBox::question(this, "Create Account", "Would you like to use additional security for this key? This lets you protect it with a different password to other keys, but also means you must re-enter the key's password every time you wish to use the account.", QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes)
	{
		bool ok = false;
		std::string hint;
		std::string password = getPassword("Create Account", "Enter the password you would like to use for this key. Don't forget it!", &hint, &ok);
		if (!ok)
			return;
		m_keyManager.import(p.secret(), s.toStdString(), password, hint);
	}
	else
		m_keyManager.import(p.secret(), s.toStdString());
	keysChanged();
}

void Main::on_killAccount_triggered()
{
	if (ui->ourAccounts->currentRow() >= 0)
	{
		auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
		Address h((byte const*)hba.data(), Address::ConstructFromPointer);
		auto k = m_keyManager.accountDetails()[h];
		QString s = QInputDialog::getText(this, QString::fromStdString("Kill Account " + k.first + "?!"),
			QString::fromStdString("Account " + k.first + " (" + render(h) + ") has " + formatBalance(ethereum()->balanceAt(h)) + " in it.\r\nIt, and any contract that this account can access, will be lost forever if you continue. Do NOT continue unless you know what you are doing.\n"
			"Are you sure you want to continue? \r\n If so, type 'YES' to confirm."),
			QLineEdit::Normal, "NO");
		if (s != "YES")
			return;
		m_keyManager.kill(h);
		if (m_keyManager.accounts().empty())
			m_keyManager.import(Secret::random(), "Default account");
		m_beneficiary = *m_keyManager.accounts().begin();
		keysChanged();
		if (m_beneficiary == h)
			setBeneficiary(*m_keyManager.accounts().begin());
	}
}

void Main::on_reencryptKey_triggered()
{
	if (ui->ourAccounts->currentRow() >= 0)
	{
		auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
		Address a((byte const*)hba.data(), Address::ConstructFromPointer);
		QStringList kdfs = {"PBKDF2-SHA256", "Scrypt"};
		bool ok = true;
		KDF kdf = (KDF)kdfs.indexOf(QInputDialog::getItem(this, "Re-Encrypt Key", "Select a key derivation function to use for storing your key:", kdfs, kdfs.size() - 1, false, &ok));
		if (!ok)
			return;
		std::string hint;
		std::string password = getPassword("Create Account", "Enter the password you would like to use for this key. Don't forget it!\nEnter nothing to use your Master password.", &hint, &ok);
		if (!ok)
			return;
		try {
			auto pw = [&](){
				auto p = QInputDialog::getText(this, "Re-Encrypt Key", "Enter the original password for this key.\nHint: " + QString::fromStdString(m_keyManager.hint(a)), QLineEdit::Password, QString()).toStdString();
				if (p.empty())
					throw UnknownPassword();
				return p;
			};
			while (!(password.empty() ? m_keyManager.recode(a, SemanticPassword::Master, pw, kdf) : m_keyManager.recode(a, password, hint, pw, kdf)))
				if (QMessageBox::question(this, "Re-Encrypt Key", "Password given is incorrect. Would you like to try again?", QMessageBox::Retry, QMessageBox::Cancel) == QMessageBox::Cancel)
					return;
		}
		catch (UnknownPassword&) {}
	}
}

void Main::on_reencryptAll_triggered()
{
	QStringList kdfs = {"PBKDF2-SHA256", "Scrypt"};
	bool ok = false;
	QString kdf = QInputDialog::getItem(this, "Re-Encrypt Key", "Select a key derivation function to use for storing your keys:", kdfs, kdfs.size() - 1, false, &ok);
	if (!ok)
		return;
	try {
		for (Address const& a: m_keyManager.accounts())
			while (!m_keyManager.recode(a, SemanticPassword::Existing, [&](){
				auto p = QInputDialog::getText(nullptr, "Re-Encrypt Key", QString("Enter the original password for key %1.\nHint: %2").arg(QString::fromStdString(pretty(a))).arg(QString::fromStdString(m_keyManager.hint(a))), QLineEdit::Password, QString()).toStdString();
				if (p.empty())
					throw UnknownPassword();
				return p;
			}, (KDF)kdfs.indexOf(kdf)))
				if (QMessageBox::question(this, "Re-Encrypt Key", "Password given is incorrect. Would you like to try again?", QMessageBox::Retry, QMessageBox::Cancel) == QMessageBox::Cancel)
					return;
	}
	catch (UnknownPassword&) {}
}

void Main::on_go_triggered()
{
	if (!ui->net->isChecked())
	{
		ui->net->setChecked(true);
		on_net_triggered();
	}
	for (auto const& i: Host::pocHosts())
		web3()->requirePeer(i.first, i.second);
}

std::string Main::prettyU256(dev::u256 const& _n) const
{
	unsigned inc = 0;
	string raw;
	ostringstream s;
	if (_n > szabo && _n < 1000000 * ether)
		s << "<span style=\"color: #215\">" << formatBalance(_n) << "</span> <span style=\"color: #448\">(0x" << hex << (uint64_t)_n << ")</span>";
	else if (!(_n >> 64))
		s << "<span style=\"color: #008\">" << (uint64_t)_n << "</span> <span style=\"color: #448\">(0x" << hex << (uint64_t)_n << ")</span>";
	else if (!~(_n >> 64))
		s << "<span style=\"color: #008\">" << (int64_t)_n << "</span> <span style=\"color: #448\">(0x" << hex << (int64_t)_n << ")</span>";
	else if ((_n >> 160) == 0)
	{
		Address a = right160(_n);
		string n = pretty(a);
		if (n.empty())
			s << "<span style=\"color: #844\">0x</span><span style=\"color: #800\">" << a << "</span>";
		else
			s << "<span style=\"font-weight: bold; color: #800\">" << htmlEscaped(n) << "</span> (<span style=\"color: #844\">0x</span><span style=\"color: #800\">" << a.abridged() << "</span>)";
	}
	else if ((raw = fromRaw((h256)_n, &inc)).size())
		return "<span style=\"color: #484\">\"</span><span style=\"color: #080\">" + htmlEscaped(raw) + "</span><span style=\"color: #484\">\"" + (inc ? " + " + toString(inc) : "") + "</span>";
	else
		s << "<span style=\"color: #466\">0x</span><span style=\"color: #044\">" << (h256)_n << "</span>";
	return s.str();
}

void Main::on_post_clicked()
{
	return;
	shh::Message m;
	m.setTo(stringToPublic(ui->shhTo->currentText()));
	m.setPayload(parseData(ui->shhData->toPlainText().toStdString()));
	Public f = stringToPublic(ui->shhFrom->currentText());
	Secret from;
	if (m_server->ids().count(f))
		from = m_server->ids().at(f);
	whisper()->inject(m.seal(from, topicFromText(ui->shhTopic->toPlainText()), ui->shhTtl->value(), ui->shhWork->value()));
}

int Main::authenticate(QString _title, QString _text)
{
	QMessageBox userInput(this);
	userInput.setText(_title);
	userInput.setInformativeText(_text);
	userInput.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	userInput.button(QMessageBox::Ok)->setText("Allow");
	userInput.button(QMessageBox::Cancel)->setText("Reject");
	userInput.setDefaultButton(QMessageBox::Cancel);
	return userInput.exec();
}

void Main::refreshWhispers()
{
	return;
	ui->whispers->clear();
	for (auto const& w: whisper()->all())
	{
		shh::Envelope const& e = w.second;
		shh::Message m;
		for (pair<Public, Secret> const& i: m_server->ids())
			if (!!(m = e.open(shh::Topics(), i.second)))
				break;
		if (!m)
			m = e.open(shh::Topics());

		QString msg;
		if (m.from())
			// Good message.
			msg = QString("{%1 -> %2} %3").arg(m.from() ? m.from().abridged().c_str() : "???").arg(m.to() ? m.to().abridged().c_str() : "*").arg(toHex(m.payload()).c_str());
		else if (m)
			// Maybe message.
			msg = QString("{%1 -> %2} %3 (?)").arg(m.from() ? m.from().abridged().c_str() : "???").arg(m.to() ? m.to().abridged().c_str() : "*").arg(toHex(m.payload()).c_str());

		time_t ex = e.expiry();
		QString t(ctime(&ex));
		t.chop(1);
		QString item = QString("[%1 - %2s] *%3 %5 %4").arg(t).arg(e.ttl()).arg(e.workProved()).arg(toString(e.topic()).c_str()).arg(msg);
		ui->whispers->addItem(item);
	}
}

void Main::dappLoaded(Dapp& _dapp)
{
	QUrl url = m_dappHost->hostDapp(std::move(_dapp));
	ui->webView->page()->setUrl(url);
}

void Main::pageLoaded(QByteArray const& _content, QString const& _mimeType, QUrl const& _uri)
{
	ui->webView->page()->setContent(_content, _mimeType, _uri);
}
