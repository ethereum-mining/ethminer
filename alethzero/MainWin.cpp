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
#include <QtWebKit/QWebSettings>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
#include <boost/algorithm/string.hpp>
#include <test/JsonSpiritHeaders.h>
#ifndef _MSC_VER
#include <libserpent/funcs.h>
#include <libserpent/util.h>
#endif
#include <libdevcrypto/FileSystem.h>
#include <libethcore/CommonJS.h>
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
#include "MainWin.h"
#include "DownloadView.h"
#include "MiningView.h"
#include "BuildInfo.h"
#include "OurWebThreeStubServer.h"
#include "ui_Main.h"
using namespace std;
using namespace dev;
using namespace dev::p2p;
using namespace dev::eth;
namespace js = json_spirit;

#define Small "font-size: small; "
#define Mono "font-family: Ubuntu Mono, Monospace, Lucida Console, Courier New; font-weight: bold; "
#define Div(S) "<div style=\"" S "\">"
#define Span(S) "<span style=\"" S "\">"

static void initUnits(QComboBox* _b)
{
	for (auto n = (unsigned)units().size(); n-- != 0; )
		_b->addItem(QString::fromStdString(units()[n].second), n);
}

QString Main::fromRaw(h256 _n, unsigned* _inc)
{
	if (_n)
	{
		string s((char const*)_n.data(), 32);
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

static vector<KeyPair> keysAsVector(QList<KeyPair> const& keys)
{
	auto list = keys.toStdList();
	return {begin(list), end(list)};
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

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	g_logPost = [=](string const& s, char const* c)
	{
		simpleDebugOut(s, c);
		m_logLock.lock();
		m_logHistory.append(QString::fromStdString(s) + "\n");
		m_logChanged = true;
		m_logLock.unlock();
//		ui->log->addItem(QString::fromStdString(s));
	};

#if ETH_DEBUG
	m_servers.append("localhost:30300");
#endif
	m_servers.append(QString::fromStdString(Host::pocHost() + ":30303"));

	cerr << "State root: " << CanonBlockChain::genesis().stateRoot << endl;
	auto block = CanonBlockChain::createGenesisBlock();
	cerr << "Block Hash: " << CanonBlockChain::genesis().hash << endl;
	cerr << "Block RLP: " << RLP(block) << endl;
	cerr << "Block Hex: " << toHex(block) << endl;
	cerr << "Network protocol version: " << c_protocolVersion << endl;
	cerr << "Client database version: " << c_databaseVersion << endl;

	ui->configDock->close();
	on_verbosity_valueChanged();
	initUnits(ui->gasPriceUnits);
	initUnits(ui->valueUnits);
	ui->valueUnits->setCurrentIndex(6);
	ui->gasPriceUnits->setCurrentIndex(4);
	ui->gasPrice->setValue(10);
	on_destination_currentTextChanged();

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->mineStatus);
	statusBar()->addPermanentWidget(ui->blockCount);

	connect(ui->ourAccounts->model(), SIGNAL(rowsMoved(const QModelIndex &, int, int, const QModelIndex &, int)), SLOT(ourAccountsRowsMoved()));
	
	QSettings s("ethereum", "alethzero");
	m_networkConfig = s.value("peers").toByteArray();
	bytesConstRef network((byte*)m_networkConfig.data(), m_networkConfig.size());
	m_webThree.reset(new WebThreeDirect(string("AlethZero/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM), getDataDir() + "/AlethZero", false, {"eth", "shh"}, p2p::NetworkPreferences(), network));

	m_qwebConnector.reset(new QWebThreeConnector());
	m_server.reset(new OurWebThreeStubServer(*m_qwebConnector, *web3(), keysAsVector(m_myKeys), this));
	connect(&*m_server, SIGNAL(onNewId(QString)), SLOT(addNewId(QString)));
	m_server->setIdentities(keysAsVector(owned()));
	m_server->StartListening();

	connect(ui->webView, &QWebView::loadStarted, [this]()
	{
		// NOTE: no need to delete as QETH_INSTALL_JS_NAMESPACE adopts it.
		m_qweb = new QWebThree(this);
		auto qweb = m_qweb;
		m_qwebConnector->setQWeb(qweb);

		QWebSettings::globalSettings()->setAttribute(QWebSettings::DeveloperExtrasEnabled, true);
		QWebFrame* f = ui->webView->page()->mainFrame();
		f->disconnect(SIGNAL(javaScriptWindowObjectCleared()));
		
		connect(f, &QWebFrame::javaScriptWindowObjectCleared, QETH_INSTALL_JS_NAMESPACE(f, this, qweb));
		connect(m_qweb, SIGNAL(onNewId(QString)), this, SLOT(addNewId(QString)));
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
	writeSettings();
	// Must do this here since otherwise m_ethereum'll be deleted (and therefore clearWatches() called by the destructor)
	// *after* the client is dead.
	m_qweb->clientDieing();
	g_logPost = simpleDebugOut;
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
	return NetworkPreferences(ui->port->value(), ui->forceAddress->text().toStdString(), ui->upnp->isChecked(), ui->localNetworking->isChecked());
}

void Main::onKeysChanged()
{
	installBalancesWatch();
}

unsigned Main::installWatch(LogFilter const& _tf, WatchHandler const& _f)
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

void Main::uninstallWatch(unsigned _w)
{
	ethereum()->uninstallWatch(_w);
	m_handlers.erase(_w);
}

void Main::installWatches()
{
	installWatch(LogFilter().address(c_newConfig), [=](LocalisedLogEntries const&) { installNameRegWatch(); });
	installWatch(LogFilter().address(c_newConfig), [=](LocalisedLogEntries const&) { installCurrenciesWatch(); });
	installWatch(PendingChangedFilter, [=](LocalisedLogEntries const&){ onNewPending(); });
	installWatch(ChainChangedFilter, [=](LocalisedLogEntries const&){ onNewBlock(); });
}

Address Main::getNameReg() const
{
	return abiOut<Address>(ethereum()->call(c_newConfig, abiIn("lookup(uint256)", (u256)1)));
}

Address Main::getCurrencies() const
{
	return abiOut<Address>(ethereum()->call(c_newConfig, abiIn("lookup(uint256)", (u256)3)));
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
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, 0); ++i)
		altCoins.push_back(right160(ethereum()->stateAt(coinsAddr, i + 1)));
	for (auto i: m_myKeys)
		for (auto c: altCoins)
			tf.address(c).topic(0, h256(i.address(), h256::AlignRight));

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
	refreshAccounts();

	// We must update balances since we can't filter updates to basic accounts.
	refreshBalances();
}

void Main::onNewPending()
{
	cwatch << "Pending transactions changed!";

	// update any pending-transaction dependent views.
	refreshPending();
	refreshAccounts();
}

void Main::on_forceMining_triggered()
{
	ethereum()->setForceMining(ui->forceMining->isChecked());
}

void Main::on_enableOptimizer_triggered()
{
	m_enableOptimizer = ui->enableOptimizer->isChecked();
	on_data_textChanged();
}

QString Main::contents(QString _s)
{
	return QString::fromStdString(dev::asString(dev::contents(_s.toStdString())));
}

void Main::load(QString _s)
{
	QString contents = QString::fromStdString(dev::asString(dev::contents(_s.toStdString())));
	ui->webView->page()->currentFrame()->evaluateJavaScript(contents);
	/*
	QFile fin(_s);
	if (!fin.open(QFile::ReadOnly))
		return;
	QString line;
	while (!fin.atEnd())
	{
		QString l = QString::fromUtf8(fin.readLine());
		line.append(l);
		if (line.count('"') % 2)
		{
			line.chop(1);
		}
		else if (line.endsWith("\\\n"))
			line.chop(2);
		else
		{
			ui->webView->page()->currentFrame()->evaluateJavaScript(line);
			//eval(line);
			line.clear();
		}
	}*/
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

QVariant Main::evalRaw(QString const& _js)
{
	return ui->webView->page()->currentFrame()->evaluateJavaScript(_js);
}

void Main::eval(QString const& _js)
{
	if (_js.trimmed().isEmpty())
		return;
	QVariant ev = ui->webView->page()->currentFrame()->evaluateJavaScript((_js.startsWith("{") || _js.startsWith("if ") || _js.startsWith("if(")) ? _js : ("___RET=(" + _js + ")"));
	QVariant jsonEv = ui->webView->page()->currentFrame()->evaluateJavaScript("JSON.stringify(___RET)");
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
	m_consoleHistory.push_back(qMakePair(_js, s));
	s = "<html><body style=\"margin: 0;\">" Div(Mono "position: absolute; bottom: 0; border: 0px; margin: 0px; width: 100%");
	for (auto const& i: m_consoleHistory)
		s +=	"<div style=\"border-bottom: 1 solid #eee; width: 100%\"><span style=\"float: left; width: 1em; color: #888; font-weight: bold\">&gt;</span><span style=\"color: #35d\">" + i.first.toHtmlEscaped() + "</span></div>"
				"<div style=\"border-bottom: 1 solid #eee; width: 100%\"><span style=\"float: left; width: 1em\">&nbsp;</span><span>" + i.second + "</span></div>";
	s += "</div></body></html>";
	ui->jsConsole->setHtml(s);
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

//static Address g_newNameReg;

QString Main::pretty(dev::Address _a) const
{
/*	static map<Address, QString> s_memos;

	if (!s_memos.count(_a))
	{*/
//		if (!g_newNameReg)
			auto g_newNameReg = getNameReg();

		if (g_newNameReg)
		{
			QString s = QString::fromStdString(toString(abiOut<string32>(ethereum()->call(g_newNameReg, abiIn("nameOf(address)", _a)))));
//			s_memos[_a] = s;
			if (s.size())
				return s;
		}
/*	}
	else
		if (s_memos[_a].size())
			return s_memos[_a];*/

	h256 n;
/*
	if (h160 nameReg = (u160)ethereum()->stateAt(c_config, 0))
		n = ethereum()->stateAt(nameReg, (u160)(_a));

	if (!n)
		n = ethereum()->stateAt(m_nameReg, (u160)(_a));
*/
	return fromRaw(n);
}

QString Main::render(dev::Address _a) const
{
	QString p = pretty(_a);
	if (!p.isNull())
		return p + " (" + QString::fromStdString(_a.abridged()) + ")";
	return QString::fromStdString(_a.abridged());
}

string32 fromString(string const& _s)
{
	string32 ret;
	for (unsigned i = 0; i < 32 && i <= _s.size(); ++i)
		ret[i] = i < _s.size() ? _s[i] : 0;
	return ret;
}

Address Main::fromString(QString const& _n) const
{
	if (_n == "(Create Contract)")
		return Address();

/*	static map<QString, Address> s_memos;

	if (!s_memos.count(_n))
	{*/
//		if (!g_newNameReg)
			auto g_newNameReg = getNameReg();

		if (g_newNameReg)
		{
			Address a = abiOut<Address>(ethereum()->call(g_newNameReg, abiIn("addressOf(string32)", ::fromString(_n.toStdString()))));
//			s_memos[_n] = a;
			if (a)
				return a;
		}
/*	}
	else
		if (s_memos[_n])
			return s_memos[_n];

	string sn = _n.toStdString();
	if (sn.size() > 32)
		sn.resize(32);
	h256 n;
	memcpy(n.data(), sn.data(), sn.size());
	memset(n.data() + sn.size(), 0, 32 - sn.size());
	if (_n.size())
	{
		if (h160 nameReg = (u160)ethereum()->stateAt(c_config, 0))
			if (h256 a = ethereum()->stateAt(nameReg, n))
				return right160(a);

		if (h256 a = ethereum()->stateAt(m_nameReg, n))
			return right160(a);
	}*/

	if (_n.size() == 40)
	{
		try
		{
			return Address(fromHex(_n.toStdString(), ThrowType::Throw));
		}
		catch (BadHexCharacter& _e)
		{
			cwarn << "invalid hex character, address rejected";
			cwarn << boost::diagnostic_information(_e);
			return Address();
		}
		catch (...)
		{
			cwarn << "address rejected";
			return Address();
		}
	}
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
	// TODO: fix with the new DNSreg contract
//	if (h160 dnsReg = (u160)ethereum()->stateAt(c_config, 4, 0))
//		ret = ethereum()->stateAt(dnsReg, n);
/*	if (!ret)
		if (h160 nameReg = (u160)ethereum()->stateAt(c_config, 0, 0))
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
	QMessageBox::about(this, "About AlethZero PoC-" + QString(dev::Version).section('.', 1, 1), QString("AlethZero/v") + dev::Version + "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM) "\n" DEV_QUOTED(ETH_COMMIT_HASH) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBerlin ÐΞV team, 2014.\nOriginally by Gav Wood. Based on a design by Vitalik Buterin.\n\nThanks to the various contributors including: Tim Hughes, caktux, Eric Lombrozo, Marko Simovic.");
}

void Main::on_paranoia_triggered()
{
	ethereum()->setParanoia(ui->paranoia->isChecked());
}

void Main::writeSettings()
{
	QSettings s("ethereum", "alethzero");
	{
		QByteArray b;
		b.resize(sizeof(Secret) * m_myKeys.size());
		auto p = b.data();
		for (auto i: m_myKeys)
		{
			memcpy(p, &(i.secret()), sizeof(Secret));
			p += sizeof(Secret);
		}
		s.setValue("address", b);
	}
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

	s.setValue("upnp", ui->upnp->isChecked());
	s.setValue("forceAddress", ui->forceAddress->text());
	s.setValue("usePast", ui->usePast->isChecked());
	s.setValue("localNetworking", ui->localNetworking->isChecked());
	s.setValue("forceMining", ui->forceMining->isChecked());
	s.setValue("paranoia", ui->paranoia->isChecked());
	s.setValue("showAll", ui->showAll->isChecked());
	s.setValue("showAllAccounts", ui->showAllAccounts->isChecked());
	s.setValue("enableOptimizer", m_enableOptimizer);
	s.setValue("clientName", ui->clientName->text());
	s.setValue("idealPeers", ui->idealPeers->value());
	s.setValue("port", ui->port->value());
	s.setValue("url", ui->urlEdit->text());
	s.setValue("privateChain", m_privateChain);
	s.setValue("verbosity", ui->verbosity->value());
	s.setValue("jitvm", ui->jitvm->isChecked());

	bytes d = m_webThree->saveNetwork();
	if (d.size())
		m_networkConfig = QByteArray((char*)d.data(), (int)d.size());
	s.setValue("peers", m_networkConfig);
	s.setValue("nameReg", ui->nameReg->text());

	s.setValue("geometry", saveGeometry());
	s.setValue("windowState", saveState());
}

void Main::readSettings(bool _skipGeometry)
{
	QSettings s("ethereum", "alethzero");

	if (!_skipGeometry)
		restoreGeometry(s.value("geometry").toByteArray());
	restoreState(s.value("windowState").toByteArray());

	{
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
		ethereum()->setAddress(m_myKeys.back().address());
		m_server->setAccounts(keysAsVector(m_myKeys));
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

	ui->upnp->setChecked(s.value("upnp", true).toBool());
	ui->forceAddress->setText(s.value("forceAddress", "").toString());
	ui->usePast->setChecked(s.value("usePast", true).toBool());
	ui->localNetworking->setChecked(s.value("localNetworking", true).toBool());
	ui->forceMining->setChecked(s.value("forceMining", false).toBool());
	on_forceMining_triggered();
	ui->paranoia->setChecked(s.value("paranoia", false).toBool());
	ui->showAll->setChecked(s.value("showAll", false).toBool());
	ui->showAllAccounts->setChecked(s.value("showAllAccounts", false).toBool());
	m_enableOptimizer = s.value("enableOptimizer", true).toBool();
	ui->enableOptimizer->setChecked(m_enableOptimizer);
	ui->clientName->setText(s.value("clientName", "").toString());
	if (ui->clientName->text().isEmpty())
		ui->clientName->setText(QInputDialog::getText(nullptr, "Enter identity", "Enter a name that will identify you on the peer network"));
	ui->idealPeers->setValue(s.value("idealPeers", ui->idealPeers->value()).toInt());
	ui->port->setValue(s.value("port", ui->port->value()).toInt());
	ui->nameReg->setText(s.value("nameReg", "").toString());
	m_privateChain = s.value("privateChain", "").toString();
	ui->usePrivate->setChecked(m_privateChain.size());
	ui->verbosity->setValue(s.value("verbosity", 1).toInt());
	ui->jitvm->setChecked(s.value("jitvm", true).toBool());

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
			if (std::find(m_myKeys.begin(), m_myKeys.end(), k) == m_myKeys.end())
			{
				if (m_myKeys.empty())
				{
					m_myKeys.push_back(KeyPair::create());
					keysChanged();
				}
				ethereum()->transact(k.sec(), ethereum()->balanceAt(k.address()) - gasPrice() * c_txGas, m_myKeys.back().address(), {}, c_txGas, gasPrice());
			}
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
	if (ui->ourAccounts->currentRow() >= 0 && ui->ourAccounts->currentRow() < m_myKeys.size())
	{
		auto k = m_myKeys[ui->ourAccounts->currentRow()];
		QMessageBox::information(this, "Export Account Key", "Secret key to account " + render(k.address()) + " is:\n" + QString::fromStdString(toHex(k.sec().ref())));
	}
}

void Main::on_usePrivate_triggered()
{
	if (ui->usePrivate->isChecked())
	{
		m_privateChain = QInputDialog::getText(this, "Enter Name", "Enter the name of your private chain", QLineEdit::Normal, QString("NewChain-%1").arg(time(0)));
		if (m_privateChain.isEmpty())
			ui->usePrivate->setChecked(false);
	}
	else
	{
		m_privateChain.clear();
	}
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
	ethereum()->setDefault(ui->preview->isChecked() ? 0 : -1);
	refreshAll();
}

void Main::refreshMining()
{
	MineProgress p = ethereum()->miningProgress();
	ui->mineStatus->setText(ethereum()->isMining() ? QString("%1s @ %2kH/s").arg(p.ms / 1000).arg(p.ms ? p.hashes / p.ms : 0) : "Not mining");
	if (!ui->miningView->isVisible())
		return;
	list<MineInfo> l = ethereum()->miningHistory();
	static unsigned lh = 0;
	if (p.hashes < lh)
		ui->miningView->resetStats();
	lh = p.hashes;
	ui->miningView->appendStats(l, p);
/*	if (p.ms)
		for (MineInfo const& i: l)
			cnote << i.hashes * 10 << "h/sec, need:" << i.requirement << " best:" << i.best << " best-so-far:" << p.best << " avg-speed:" << (p.hashes * 1000 / p.ms) << "h/sec";
*/
}

void Main::refreshBalances()
{
	cwatch << "refreshBalances()";
	// update all the balance-dependent stuff.
	ui->ourAccounts->clear();
	u256 totalBalance = 0;
/*	map<Address, tuple<QString, u256, u256>> altCoins;
	Address coinsAddr = getCurrencies();
	for (unsigned i = 0; i < ethereum()->stateAt(coinsAddr, 0); ++i)
	{
		auto n = ethereum()->stateAt(coinsAddr, i + 1);
		auto addr = right160(ethereum()->stateAt(coinsAddr, n));
		auto denom = ethereum()->stateAt(coinsAddr, sha3(h256(n).asBytes()));
		if (denom == 0)
			denom = 1;
//		cdebug << n << addr << denom << sha3(h256(n).asBytes());
		altCoins[addr] = make_tuple(fromRaw(n), 0, denom);
	}*/
	for (auto i: m_myKeys)
	{
		u256 b = ethereum()->balanceAt(i.address());
		(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(b).c_str()).arg(render(i.address())).arg((unsigned)ethereum()->countAt(i.address())), ui->ourAccounts))
			->setData(Qt::UserRole, QByteArray((char const*)i.address().data(), Address::size));
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
							   .arg(i.socket)
							   .arg(QString::fromStdString(i.id.abridged())));

		auto ns = web3()->nodes();
		for (p2p::Peer const& i: ns)
			ui->nodes->insertItem(sessions.count(i.id) ? 0 : ui->nodes->count(), QString("[%1 %3] %2 - ( =%5s | /%4s%6 ) - *%7 $%8")
						   .arg(QString::fromStdString(i.id.abridged()))
						   .arg(QString::fromStdString(i.peerEndpoint().address().to_string()))
						   .arg(i.id == web3()->id() ? "self" : sessions.count(i.id) ? sessions[i.id] : "disconnected")
						   .arg(i.isOffline() ? " | " + QString::fromStdString(reasonOf(i.lastDisconnect())) + " | " + QString::number(i.failedAttempts()) + "x" : "")
						   .arg(i.rating())
						   );
	}
}

void Main::refreshAll()
{
	refreshDestination();
	refreshBlockChain();
	refreshBlockCount();
	refreshPending();
	refreshAccounts();
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
				.arg(render(t.safeSender()))
				.arg(render(t.receiveAddress()))
				.arg((unsigned)t.nonce())
				.arg(ethereum()->codeAt(t.receiveAddress()).size() ? '*' : '-') :
			QString("%2 +> %3: %1 [%4]")
				.arg(formatBalance(t.value()).c_str())
				.arg(render(t.safeSender()))
				.arg(render(right160(sha3(rlpList(t.safeSender(), t.nonce())))))
				.arg((unsigned)t.nonce());
		ui->transactionQueue->addItem(s);
	}
}

void Main::refreshAccounts()
{
	cwatch << "refreshAccounts()";
	ui->accounts->clear();
	ui->contracts->clear();
	for (auto n = 0; n < 2; ++n)
		for (auto i: ethereum()->addresses())
		{
			auto r = render(i);
			if (r.contains('(') == !n)
			{
				if (n == 0 || ui->showAllAccounts->isChecked())
					(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(ethereum()->balanceAt(i)).c_str()).arg(r).arg((unsigned)ethereum()->countAt(i)), ui->accounts))
						->setData(Qt::UserRole, QByteArray((char const*)i.data(), Address::size));
				if (ethereum()->codeAt(i).size())
					(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(ethereum()->balanceAt(i)).c_str()).arg(r).arg((unsigned)ethereum()->countAt(i)), ui->contracts))
						->setData(Qt::UserRole, QByteArray((char const*)i.data(), Address::size));
			}
		}
}

void Main::refreshDestination()
{
	cwatch << "refreshDestination()";
	QString s;
	for (auto i: ethereum()->addresses())
		if ((s = pretty(i)).size())
			// A namereg address
			if (ui->destination->findText(s, Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
				ui->destination->addItem(s);
	for (int i = 0; i < ui->destination->count(); ++i)
		if (ui->destination->itemText(i) != "(Create Contract)" && !fromString(ui->destination->itemText(i)))
			ui->destination->removeItem(i--);
}

void Main::refreshBlockCount()
{
	cwatch << "refreshBlockCount()";
	auto d = ethereum()->blockChain().details();
	auto diff = BlockInfo(ethereum()->blockChain().block()).difficulty;
	ui->blockCount->setText(QString("%6 #%1 @%3 T%2 PV%4 D%5").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)).arg(c_protocolVersion).arg(c_databaseVersion).arg(m_privateChain.size() ? "[" + m_privateChain + "] " : "testnet"));
}

static bool blockMatch(string const& _f, BlockDetails const& _b, h256 _h, CanonBlockChain const& _bc)
{
	try
	{
		if (_f.size() > 1 && _f.size() < 10 && _f[0] == '#' && stoul(_f.substr(1)) == _b.number)
			return true;
	}
	catch (...) {}
	if (toHex(_h.ref()).find(_f) != string::npos)
		return true;
	BlockInfo bi(_bc.block(_h));
	string info = toHex(bi.stateRoot.ref()) + " " + toHex(bi.coinbaseAddress.ref()) + " " + toHex(bi.transactionsRoot.ref()) + " " + toHex(bi.sha3Uncles.ref());
	if (info.find(_f) != string::npos)
		return true;
	return false;
}

static bool transactionMatch(string const& _f, Transaction const& _t)
{
	string info = toHex(_t.receiveAddress().ref()) + " " + toHex(_t.sha3().ref()) + " " + toHex(_t.sha3(eth::WithoutSignature).ref()) + " " + toHex(_t.sender().ref());
	if (info.find(_f) != string::npos)
		return true;
	return false;
}

void Main::on_turboMining_triggered()
{
	ethereum()->setTurboMining(ui->turboMining->isChecked());
}

void Main::refreshBlockChain()
{
	cwatch << "refreshBlockChain()";

	QByteArray oldSelected = ui->blocks->count() ? ui->blocks->currentItem()->data(Qt::UserRole).toByteArray() : QByteArray();
	ui->blocks->clear();

	string filter = ui->blockChainFilter->text().toLower().toStdString();
	auto const& bc = ethereum()->blockChain();
	unsigned i = (ui->showAll->isChecked() || !filter.empty()) ? (unsigned)-1 : 10;
	for (auto h = bc.currentHash(); bc.details(h) && i; h = bc.details(h).parent, --i)
	{
		auto d = bc.details(h);
		auto bm = blockMatch(filter, d, h, bc);
		if (bm)
		{
			QListWidgetItem* blockItem = new QListWidgetItem(QString("#%1 %2").arg(d.number).arg(h.abridged().c_str()), ui->blocks);
			auto hba = QByteArray((char const*)h.data(), h.size);
			blockItem->setData(Qt::UserRole, hba);
			if (oldSelected == hba)
				blockItem->setSelected(true);
		}
		int n = 0;
		auto b = bc.block(h);
		for (auto const& i: RLP(b)[1])
		{
			Transaction t(i.data(), CheckSignature::Sender);
			if (bm || transactionMatch(filter, t))
			{
				QString s = t.receiveAddress() ?
					QString("    %2 %5> %3: %1 [%4]")
						.arg(formatBalance(t.value()).c_str())
						.arg(render(t.safeSender()))
						.arg(render(t.receiveAddress()))
						.arg((unsigned)t.nonce())
						.arg(ethereum()->codeAt(t.receiveAddress()).size() ? '*' : '-') :
					QString("    %2 +> %3: %1 [%4]")
						.arg(formatBalance(t.value()).c_str())
						.arg(render(t.safeSender()))
						.arg(render(right160(sha3(rlpList(t.safeSender(), t.nonce())))))
						.arg((unsigned)t.nonce());
				QListWidgetItem* txItem = new QListWidgetItem(s, ui->blocks);
				auto hba = QByteArray((char const*)h.data(), h.size);
				txItem->setData(Qt::UserRole, hba);
				txItem->setData(Qt::UserRole + 1, n);
				if (oldSelected == hba)
					txItem->setSelected(true);
			}
			n++;
		}
		if (h == bc.genesisHash())
			break;
	}

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
	}
	else
		interval += 100;

	for (auto const& i: m_handlers)
	{
		auto ls = ethereum()->checkWatch(i.first);
		if (ls.size())
		{
			cnote << "FIRING WATCH" << i.first << ls.size();
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
		s << "<code style=\"white-space: pre; font-weight: bold\">" << lead(ad.changeType()) << "  </code>" << " <b>" << render(i.first).toStdString() << "</b>";
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

			s << prettyU256(i.first).toStdString();
/*			if (i.first > u256(1) << 246)
				s << (h256)i.first;
			else if (i.first > u160(1) << 150)
				s << (h160)(u160)i.first;
			else
				s << hex << i.first;
*/
			if (!i.second.from())
				s << ": " << prettyU256(i.second.to()).toStdString();
			else if (!i.second.to())
				s << " (" << prettyU256(i.second.from()).toStdString() << ")";
			else
				s << ": " << prettyU256(i.second.to()).toStdString() << " (" << prettyU256(i.second.from()).toStdString() << ")";
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
		s << "From: <b>" << pretty(ss).toStdString() << "</b> " << ss;
		if (tx.isCreation())
			s << "<br/>Creates: <b>" << pretty(right160(th)).toStdString() << "</b> " << right160(th);
		else
			s << "<br/>To: <b>" << pretty(tx.receiveAddress()).toStdString() << "</b> " << tx.receiveAddress();
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
		s << "<div>Log Bloom: " << receipt.bloom() << "</div>";
		auto r = receipt.rlp();
		s << "<div>Receipt: " << toString(RLP(r)) << "</div>";
		s << "<div>Receipt-Hex: " Span(Mono) << toHex(receipt.rlp()) << "</span></div>";
		s << renderDiff(ethereum()->diff(i, 0));
//		s << "Pre: " << fs.rootHash() << "<br/>";
//		s << "Post: <b>" << ts.rootHash() << "</b>";
	}

	ui->pendingInfo->setHtml(QString::fromStdString(s.str()));
	ui->pendingInfo->moveCursor(QTextCursor::Start);
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
		m_server->setAccounts(keysAsVector(m_myKeys));
}

void Main::on_inject_triggered()
{
	QString s = QInputDialog::getText(this, "Inject Transaction", "Enter transaction dump in hex");
	try
	{
		bytes b = fromHex(s.toStdString(), ThrowType::Throw);
		ethereum()->inject(&b);
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
			s << "<br/>D/TD: <b>2^" << log2((double)info.difficulty) << "</b>/<b>2^" << log2((double)details.totalDifficulty) << "</b>";
			s << "&nbsp;&emsp;&nbsp;Children: <b>" << details.children.size() << "</b></h5>";
			s << "<br/>Gas used/limit: <b>" << info.gasUsed << "</b>/<b>" << info.gasLimit << "</b>";
			s << "<br/>Coinbase: <b>" << pretty(info.coinbaseAddress).toHtmlEscaped().toStdString() << "</b> " << info.coinbaseAddress;
			s << "<br/>Nonce: <b>" << info.nonce << "</b>";
			s << "<br/>Hash w/o nonce: <b>" << info.headerHash(WithoutNonce) << "</b>";
			s << "<br/>Difficulty: <b>" << info.difficulty << "</b>";
			if (info.number)
				s << "<br/>Proof-of-Work: <b>" << ProofOfWork::eval(info.headerHash(WithoutNonce), info.nonce) << " &lt;= " << (h256)u256((bigint(1) << 256) / info.difficulty) << "</b>";
			else
				s << "<br/>Proof-of-Work: <i>Phil has nothing to prove</i>";
			s << "<br/>Parent: <b>" << info.parentHash << "</b>";
//			s << "<br/>Bloom: <b>" << details.bloom << "</b>";
			s << "<br/>Log Bloom: <b>" << info.logBloom << "</b>";
			s << "<br/>Transactions: <b>" << block[1].itemCount() << "</b> @<b>" << info.transactionsRoot << "</b>";
			s << "<br/>Receipts: @<b>" << info.receiptsRoot << "</b>:";
			s << "<br/>Uncles: <b>" << block[2].itemCount() << "</b> @<b>" << info.sha3Uncles << "</b>";
			for (auto u: block[2])
			{
				BlockInfo uncle = BlockInfo::fromHeader(u.data());
				s << "<br/><span style=\"margin-left: 2em\">&nbsp;</span>Hash: <b>" << uncle.hash << "</b>";
				s << "<br/><span style=\"margin-left: 2em\">&nbsp;</span>Parent: <b>" << uncle.parentHash << "</b>";
				s << "<br/><span style=\"margin-left: 2em\">&nbsp;</span>Number: <b>" << uncle.number << "</b>";
			}
			if (info.parentHash)
				s << "<br/>Pre: <b>" << BlockInfo(ethereum()->blockChain().block(info.parentHash)).stateRoot << "</b>";
			else
				s << "<br/>Pre: <i>Nothing is before Phil</i>";
			for (auto const& i: block[1])
				s << "<br/>" << sha3(i.data()).abridged();// << ": <b>" << i[1].toHash<h256>() << "</b> [<b>" << i[2].toInt<u256>() << "</b> used]";
			s << "<br/>Post: <b>" << info.stateRoot << "</b>";
			s << "<br/>Dump: " Span(Mono) << toHex(block[0].data()) << "</span>";
			s << "<div>Receipts-Hex: " Span(Mono) << toHex(ethereum()->blockChain().receipts(h).rlp()) << "</span></div>";
		}
		else
		{
			unsigned txi = item->data(Qt::UserRole + 1).toInt();
			Transaction tx(block[1][txi].data(), CheckSignature::Sender);
			auto ss = tx.safeSender();
			h256 th = sha3(rlpList(ss, tx.nonce()));
			TransactionReceipt receipt = ethereum()->blockChain().receipts(h).receipts[txi];
			s << "<h3>" << th << "</h3>";
			s << "<h4>" << h << "[<b>" << txi << "</b>]</h4>";
			s << "<br/>From: <b>" << pretty(ss).toHtmlEscaped().toStdString() << "</b> " << ss;
			if (tx.isCreation())
				s << "<br/>Creates: <b>" << pretty(right160(th)).toHtmlEscaped().toStdString() << "</b> " << right160(th);
			else
				s << "<br/>To: <b>" << pretty(tx.receiveAddress()).toHtmlEscaped().toStdString() << "</b> " << tx.receiveAddress();
			s << "<br/>Value: <b>" << formatBalance(tx.value()) << "</b>";
			s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce() << "</b>";
			s << "<br/>Gas price: <b>" << formatBalance(tx.gasPrice()) << "</b>";
			s << "<br/>Gas: <b>" << tx.gas() << "</b>";
			s << "<br/>V: <b>" << hex << nouppercase << (int)tx.signature().v << " + 27</b>";
			s << "<br/>R: <b>" << hex << nouppercase << tx.signature().r << "</b>";
			s << "<br/>S: <b>" << hex << nouppercase << tx.signature().s << "</b>";
			s << "<br/>Msg: <b>" << tx.sha3(eth::WithoutSignature) << "</b>";
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
			s << "<div>Hex: " Span(Mono) << toHex(block[1][txi].data()) << "</span></div>";
			s << "<hr/>";
			s << "<div>Log Bloom: " << receipt.bloom() << "</div>";
			auto r = receipt.rlp();
			s << "<div>Receipt: " << toString(RLP(r)) << "</div>";
			s << "<div>Receipt-Hex: " Span(Mono) << toHex(receipt.rlp()) << "</span></div>";
			s << renderDiff(ethereum()->diff(txi, h));
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
			m_executiveState = ethereum()->state(txi + 1, h);
			m_currentExecution = unique_ptr<Executive>(new Executive(m_executiveState, ethereum()->blockChain(), 0));
			Transaction t = m_executiveState.pending()[txi];
			m_executiveState = m_executiveState.fromPending(txi);
			auto r = t.rlp();
			populateDebugger(&r);
			m_currentExecution.reset();
		}
	}
}

void Main::on_debugDumpState_triggered(int _add)
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

void Main::on_debugDumpStatePre_triggered()
{
	on_debugDumpState_triggered(0);
}

void Main::populateDebugger(dev::bytesConstRef _r)
{
	bool done = m_currentExecution->setup(_r);
	if (!done)
	{
		debugFinished();
		vector<WorldState const*> levels;
		m_codes.clear();
		bytes lastExtCode;
		bytesConstRef lastData;
		h256 lastHash;
		h256 lastDataHash;
		auto onOp = [&](uint64_t steps, Instruction inst, dev::bigint newMemSize, dev::bigint gasCost, VM* voidVM, ExtVMFace const* voidExt)
		{
			VM& vm = *voidVM;
			ExtVM const& ext = *static_cast<ExtVM const*>(voidExt);
			if (ext.code != lastExtCode)
			{
				lastExtCode = ext.code;
				lastHash = sha3(lastExtCode);
				if (!m_codes.count(lastHash))
					m_codes[lastHash] = ext.code;
			}
			if (ext.data != lastData)
			{
				lastData = ext.data;
				lastDataHash = sha3(lastData);
				if (!m_codes.count(lastDataHash))
					m_codes[lastDataHash] = ext.data.toBytes();
			}
			if (levels.size() < ext.depth)
				levels.push_back(&m_history.back());
			else
				levels.resize(ext.depth);
			m_history.append(WorldState({steps, ext.myAddress, vm.curPC(), inst, newMemSize, vm.gas(), lastHash, lastDataHash, vm.stack(), vm.memory(), gasCost, ext.state().storage(ext.myAddress), levels}));
		};
		m_currentExecution->go(onOp);
		m_currentExecution->finalize();
		initDebugger();
		updateDebugger();
	}
}

void Main::on_contracts_currentItemChanged()
{
	ui->contractInfo->clear();
	if (auto item = ui->contracts->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 20);
		auto address = h160((byte const*)hba.data(), h160::ConstructFromPointer);

		stringstream s;
		try
		{
			auto storage = ethereum()->storageAt(address);
			for (auto const& i: storage)
				s << "@" << showbase << hex << prettyU256(i.first).toStdString() << "&nbsp;&nbsp;&nbsp;&nbsp;" << showbase << hex << prettyU256(i.second).toStdString() << "<br/>";
			s << "<h4>Body Code</h4>" << disassemble(ethereum()->codeAt(address));
			s << Div(Mono) << toHex(ethereum()->codeAt(address)) << "</div>";
			ui->contractInfo->appendHtml(QString::fromStdString(s.str()));
		}
		catch (dev::InvalidTrie)
		{
			ui->contractInfo->appendHtml("Corrupted trie.");
		}
		ui->contractInfo->moveCursor(QTextCursor::Start);
	}
}

void Main::on_idealPeers_valueChanged()
{
	m_webThree->setIdealPeerCount(ui->idealPeers->value());
}

void Main::on_ourAccounts_doubleClicked()
{
	auto hba = ui->ourAccounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::on_log_doubleClicked()
{
	ui->log->setPlainText("");
	m_logHistory.clear();
}

void Main::on_accounts_doubleClicked()
{
	auto hba = ui->accounts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::on_contracts_doubleClicked()
{
	auto hba = ui->contracts->currentItem()->data(Qt::UserRole).toByteArray();
	auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::on_destination_currentTextChanged()
{
	if (ui->destination->currentText().size() && ui->destination->currentText() != "(Create Contract)")
		if (Address a = fromString(ui->destination->currentText()))
			ui->calculatedName->setText(render(a));
		else
			ui->calculatedName->setText("Unknown Address");
	else
		ui->calculatedName->setText("Create Contract");
	on_data_textChanged();
//	updateFee();
}

static shh::FullTopic topicFromText(QString _s)
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

bool Main::sourceIsSolidity(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 8) == "contract" || _source.substr(0, 5) == "//sol");
}

static bool sourceIsSerpent(string const& _source)
{
	// TODO: Improve this heuristic
	return (_source.substr(0, 5) == "//ser");
}

string const Main::getFunctionHashes(dev::solidity::CompilerStack const &_compiler,
									 string const& _contractName)
{
	string ret = "";
	auto const& contract = _compiler.getContractDefinition(_contractName);
	auto interfaceFunctions = contract.getInterfaceFunctions();

	for (auto const& it: interfaceFunctions)
	{
		ret += it.first.abridged();
		ret += " :";
		ret += it.second->getDeclaration().getName() + "\n";
	}
	return ret;
}

void Main::on_data_textChanged()
{
	m_pcWarp.clear();
	if (isCreation())
	{
		string src = ui->data->toPlainText().toStdString();
		vector<string> errors;
		QString lll;
		QString solidity;
		if (src.find_first_not_of("1234567890abcdefABCDEF") == string::npos && src.size() % 2 == 0)
		{
			m_data = fromHex(src);
		}
		else if (sourceIsSolidity(src))
		{
			dev::solidity::CompilerStack compiler;
			try
			{
//				compiler.addSources(dev::solidity::StandardSources);
				m_data = compiler.compile(src, m_enableOptimizer);
				solidity = "<h4>Solidity</h4>";
				solidity += "<pre>var " + QString::fromStdString(compiler.getContractNames().front()) + " = web3.eth.contractFromAbi(" + QString::fromStdString(compiler.getInterface()).replace(QRegExp("\\s"), "").toHtmlEscaped() + ");</pre>";
				solidity += "<pre>" + QString::fromStdString(compiler.getSolidityInterface()).toHtmlEscaped() + "</pre>";
				solidity += "<pre>" + QString::fromStdString(getFunctionHashes(compiler)).toHtmlEscaped() + "</pre>";
			}
			catch (dev::Exception const& exception)
			{
				ostringstream error;
				solidity::SourceReferenceFormatter::printExceptionInformation(error, exception, "Error", compiler);
				solidity = "<h4>Solidity</h4><pre>" + QString::fromStdString(error.str()).toHtmlEscaped() + "</pre>";
			}
			catch (...)
			{
				solidity = "<h4>Solidity</h4><pre>Uncaught exception.</pre>";
			}
		}
#ifndef _MSC_VER
		else if (sourceIsSerpent(src))
		{
			try
			{
				m_data = dev::asBytes(::compile(src));
				for (auto& i: errors)
					i = "(LLL " + i + ")";
			}
			catch (string err)
			{
				errors.push_back("Serpent " + err);
			}
		}
#endif
		else
		{
			m_data = compileLLL(src, m_enableOptimizer, &errors);
			if (errors.empty())
			{
				auto asmcode = compileLLLToAsm(src, false);
				lll = "<h4>Pre</h4><pre>" + QString::fromStdString(asmcode).toHtmlEscaped() + "</pre>";
				if (m_enableOptimizer)
				{
					asmcode = compileLLLToAsm(src, true);
					lll = "<h4>Opt</h4><pre>" + QString::fromStdString(asmcode).toHtmlEscaped() + "</pre>" + lll;
				}
			}
		}
		QString errs;
		if (errors.size())
		{
			errs = "<h4>Errors</h4>";
			for (auto const& i: errors)
				errs.append("<div style=\"border-left: 6px solid #c00; margin-top: 2px\">" + QString::fromStdString(i).toHtmlEscaped() + "</div>");
		}
		ui->code->setHtml(errs + lll + solidity + "<h4>Code</h4>" + QString::fromStdString(disassemble(m_data)).toHtmlEscaped() + "<h4>Hex</h4>" Div(Mono) + QString::fromStdString(toHex(m_data)) + "</div>");
		ui->gas->setMinimum((qint64)Client::txGas(m_data, 0));
		if (!ui->gas->isEnabled())
			ui->gas->setValue(m_backupGas);
		ui->gas->setEnabled(true);
	}
	else
	{
		m_data = parseData(ui->data->toPlainText().toStdString());
		ui->code->setHtml(QString::fromStdString(dev::memDump(m_data, 8, true)));
		if (ethereum()->codeAt(fromString(ui->destination->currentText()), 0).size())
		{
			ui->gas->setMinimum((qint64)Client::txGas(m_data, 1));
			if (!ui->gas->isEnabled())
				ui->gas->setValue(m_backupGas);
			ui->gas->setEnabled(true);
		}
		else
		{
			if (ui->gas->isEnabled())
				m_backupGas = ui->gas->value();
			ui->gas->setValue((qint64)Client::txGas(m_data));
			ui->gas->setEnabled(false);
		}
	}
	updateFee();
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

bool Main::isCreation() const
{
	return ui->destination->currentText().isEmpty() || ui->destination->currentText() == "(Create Contract)";
}

u256 Main::fee() const
{
	return ui->gas->value() * gasPrice();
}

u256 Main::value() const
{
	if (ui->valueUnits->currentIndex() == -1)
		return 0;
	return ui->value->value() * units()[units().size() - 1 - ui->valueUnits->currentIndex()].first;
}

u256 Main::gasPrice() const
{
	if (ui->gasPriceUnits->currentIndex() == -1)
		return 0;
	return ui->gasPrice->value() * units()[units().size() - 1 - ui->gasPriceUnits->currentIndex()].first;
}

u256 Main::total() const
{
	return value() + fee();
}

void Main::updateFee()
{
	ui->fee->setText(QString("(gas sub-total: %1)").arg(formatBalance(fee()).c_str()));
	auto totalReq = total();
	ui->total->setText(QString("Total: %1").arg(formatBalance(totalReq).c_str()));

	bool ok = false;
	for (auto i: m_myKeys)
		if (ethereum()->balanceAt(i.address()) >= totalReq)
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
	string n = string("AlethZero/v") + dev::Version;
	if (ui->clientName->text().size())
		n += "/" + ui->clientName->text().toStdString();
	n +=  "/" DEV_QUOTED(ETH_BUILD_TYPE) "/" DEV_QUOTED(ETH_BUILD_PLATFORM);
	web3()->setClientVersion(n);
	if (ui->net->isChecked())
	{
		web3()->setIdealPeerCount(ui->idealPeers->value());
		web3()->setNetworkPreferences(netPrefs());
		ethereum()->setNetworkId(m_privateChain.size() ? sha3(m_privateChain.toStdString()) : 0);
		// TODO: p2p
//		if (m_networkConfig.size()/* && ui->usePast->isChecked()*/)
//			web3()->restoreNetwork(bytesConstRef((byte*)m_networkConfig.data(), m_networkConfig.size()));
		web3()->startNetwork();
		ui->downloadView->setDownloadMan(ethereum()->downloadMan());
	}
	else
	{
		ui->downloadView->setDownloadMan(nullptr);
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
	bool ok = false;
	QString s = QInputDialog::getItem(this, "Connect to a Network Peer", "Enter a peer to which a connection may be made:", m_servers, m_servers.count() ? rand() % m_servers.count() : 0, true, &ok);
	if (ok && s.contains(":"))
	{
		string host = s.section(":", 0, 0).toStdString();
		unsigned short port = s.section(":", 1).toInt();
		web3()->connect(host, port);
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
		ethereum()->setAddress(m_myKeys.last().address());
		ethereum()->startMining();
	}
	else
		ethereum()->stopMining();
}

void Main::on_send_clicked()
{
	u256 totalReq = value() + fee();
	for (auto i: m_myKeys)
		if (ethereum()->balanceAt(i.address(), 0) >= totalReq)
		{
			debugFinished();
			Secret s = i.secret();
			if (isCreation())
			{
				// If execution is a contract creation, add Natspec to
				// a local Natspec LEVELDB
				ethereum()->transact(s, value(), m_data, ui->gas->value(), gasPrice());
				string src = ui->data->toPlainText().toStdString();
				if (sourceIsSolidity(src))
					try
					{
						dev::solidity::CompilerStack compiler;
						m_data = compiler.compile(src, m_enableOptimizer);
						for (string& s: compiler.getContractNames())
						{
							h256 contractHash = compiler.getContractCodeHash(s);
							m_natspecDB.add(contractHash,
											compiler.getMetadata(s, dev::solidity::DocumentationType::NatspecUser));
						}
					}
					catch (...)
					{
						statusBar()->showMessage("Couldn't compile Solidity Contract.");
					}
			}
			else
				ethereum()->transact(s, value(), fromString(ui->destination->currentText()), m_data, ui->gas->value(), gasPrice());
			return;
		}
	statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
}

void Main::keysChanged()
{
	onBalancesChange();
	m_server->setAccounts(keysAsVector(m_myKeys));
}

void Main::on_debug_clicked()
{
	debugFinished();
	try
	{
		u256 totalReq = value() + fee();
		for (auto i: m_myKeys)
			if (ethereum()->balanceAt(i.address()) >= totalReq)
			{
				Secret s = i.secret();
				m_executiveState = ethereum()->postState();
				m_currentExecution = unique_ptr<Executive>(new Executive(m_executiveState, ethereum()->blockChain(), 0));
				Transaction t = isCreation() ?
					Transaction(value(), gasPrice(), ui->gas->value(), m_data, m_executiveState.transactionsFrom(dev::toAddress(s)), s) :
					Transaction(value(), gasPrice(), ui->gas->value(), fromString(ui->destination->currentText()), m_data, m_executiveState.transactionsFrom(dev::toAddress(s)), s);
				auto r = t.rlp();
				populateDebugger(&r);
				m_currentExecution.reset();
				return;
			}
		statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
	}
	catch (dev::Exception const& _e)
	{
		statusBar()->showMessage("Error running transaction: " + QString::fromStdString(diagnostic_information(_e)));
		// this output is aimed at developers, reconsider using _e.what for more user friendly output.
	}
}

bool beginsWith(Address _a, bytes const& _b)
{
	for (unsigned i = 0; i < min<unsigned>(20, _b.size()); ++i)
		if (_a[i] != _b[i])
			return false;
	return true;
}

void Main::on_create_triggered()
{
	bool ok = true;
	enum { NoVanity = 0, FirstTwo, FirstTwoNextTwo, FirstThree, FirstFour, StringMatch };
	QStringList items = {"No vanity (instant)", "Two pairs first (a few seconds)", "Two pairs first and second (a few minutes)", "Three pairs first (a few minutes)", "Four pairs first (several hours)", "Specific hex string"};
	unsigned v = items.QList<QString>::indexOf(QInputDialog::getItem(this, "Vanity Key?", "Would you a vanity key? This could take several hours.", items, 0, false, &ok));
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
	m_myKeys.append(p);
	keysChanged();
}

void Main::on_killAccount_triggered()
{
	if (ui->ourAccounts->currentRow() >= 0 && ui->ourAccounts->currentRow() < m_myKeys.size())
	{
		auto k = m_myKeys[ui->ourAccounts->currentRow()];
		if (ethereum()->balanceAt(k.address()) != 0 && QMessageBox::critical(this, "Kill Account?!", "Account " + render(k.address()) + " has " + QString::fromStdString(formatBalance(ethereum()->balanceAt(k.address()))) + " in it. It, and any contract that this account can access, will be lost forever if you continue. Do NOT continue unless you know what you are doing.\nAre you sure you want to continue?", QMessageBox::Yes, QMessageBox::No) == QMessageBox::No)
			return;
		m_myKeys.erase(m_myKeys.begin() + ui->ourAccounts->currentRow());
		keysChanged();
	}
}

void Main::on_debugStep_triggered()
{
	if (ui->debugTimeline->value() < m_history.size()) {
		auto l = m_history[ui->debugTimeline->value()].levels.size();
		if ((ui->debugTimeline->value() + 1) < m_history.size() && m_history[ui->debugTimeline->value() + 1].levels.size() > l)
		{
			on_debugStepInto_triggered();
			if (m_history[ui->debugTimeline->value()].levels.size() > l)
				on_debugStepOut_triggered();
		}
		else
			on_debugStepInto_triggered();
	}
}

void Main::on_debugStepInto_triggered()
{
	ui->debugTimeline->setValue(ui->debugTimeline->value() + 1);
	ui->callStack->setCurrentRow(0);
}

void Main::on_debugStepOut_triggered()
{
	if (ui->debugTimeline->value() < m_history.size())
	{
		auto ls = m_history[ui->debugTimeline->value()].levels.size();
		auto l = ui->debugTimeline->value();
		for (; l < m_history.size() && m_history[l].levels.size() >= ls; ++l) {}
		ui->debugTimeline->setValue(l);
		ui->callStack->setCurrentRow(0);
	}
}

void Main::on_debugStepBackInto_triggered()
{
	ui->debugTimeline->setValue(ui->debugTimeline->value() - 1);
	ui->callStack->setCurrentRow(0);
}

void Main::on_debugStepBack_triggered()
{
	auto l = m_history[ui->debugTimeline->value()].levels.size();
	if (ui->debugTimeline->value() > 0 && m_history[ui->debugTimeline->value() - 1].levels.size() > l)
	{
		on_debugStepBackInto_triggered();
		if (m_history[ui->debugTimeline->value()].levels.size() > l)
			on_debugStepBackOut_triggered();
	}
	else
		on_debugStepBackInto_triggered();
}

void Main::on_debugStepBackOut_triggered()
{
	if (ui->debugTimeline->value() > 0 && m_history.size() > 0)
	{
		auto ls = m_history[min(ui->debugTimeline->value(), m_history.size() - 1)].levels.size();
		int l = ui->debugTimeline->value();
		for (; l > 0 && m_history[l].levels.size() >= ls; --l) {}
		ui->debugTimeline->setValue(l);
		ui->callStack->setCurrentRow(0);
	}
}

void Main::on_dumpTrace_triggered()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_history)
			f << ws.cur << " " << hex << toHex(dev::toCompactBigEndian(ws.curPC, 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)ws.inst, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)ws.gas, 1)) << endl;
}

void Main::on_dumpTracePretty_triggered()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_history)
		{
			f << endl << "    STACK" << endl;
			for (auto i: ws.stack)
				f << (h256)i << endl;
			f << "    MEMORY" << endl << dev::memDump(ws.memory);
			f << "    STORAGE" << endl;
			for (auto const& i: ws.storage)
				f << showbase << hex << i.first << ": " << i.second << endl;
			f << dec << ws.levels.size() << " | " << ws.cur << " | #" << ws.steps << " | " << hex << setw(4) << setfill('0') << ws.curPC << " : " << instructionInfo(ws.inst).name << " | " << dec << ws.gas << " | -" << dec << ws.gasCost << " | " << ws.newMemSize << "x32";
		}
}

void Main::on_dumpTraceStorage_triggered()
{
	QString fn = QFileDialog::getSaveFileName(this, "Select file to output EVM trace");
	ofstream f(fn.toStdString());
	if (f.is_open())
		for (WorldState const& ws: m_history)
		{
			if (ws.inst == Instruction::STOP || ws.inst == Instruction::RETURN || ws.inst == Instruction::SUICIDE)
				for (auto i: ws.storage)
					f << toHex(dev::toCompactBigEndian(i.first, 1)) << " " << toHex(dev::toCompactBigEndian(i.second, 1)) << endl;
			f << ws.cur << " " << hex << toHex(dev::toCompactBigEndian(ws.curPC, 1)) << " " << hex << toHex(dev::toCompactBigEndian((int)(byte)ws.inst, 1)) << " " << hex << toHex(dev::toCompactBigEndian((uint64_t)ws.gas, 1)) << endl;
		}
}

void Main::on_go_triggered()
{
	if (!ui->net->isChecked())
	{
		ui->net->setChecked(true);
		on_net_triggered();
	}
	web3()->connect(Host::pocHost());
}

void Main::on_callStack_currentItemChanged()
{
	updateDebugger();
}

void Main::alterDebugStateGroup(bool _enable) const
{
	ui->debugStep->setEnabled(_enable);
	ui->debugStepInto->setEnabled(_enable);
	ui->debugStepOut->setEnabled(_enable);
	ui->debugStepBackInto->setEnabled(_enable);
	ui->debugStepBackOut->setEnabled(_enable);
	ui->dumpTrace->setEnabled(_enable);
	ui->dumpTraceStorage->setEnabled(_enable);
	ui->dumpTracePretty->setEnabled(_enable);
	ui->debugStepBack->setEnabled(_enable);
	ui->debugPanel->setEnabled(_enable);
}

void Main::debugFinished()
{
	m_codes.clear();
	m_pcWarp.clear();
	m_history.clear();
	m_lastLevels.clear();
	m_lastCode = h256();
	ui->callStack->clear();
	ui->debugCode->clear();
	ui->debugStack->clear();
	ui->debugMemory->setHtml("");
	ui->debugStorage->setHtml("");
	ui->debugStateInfo->setText("");
	alterDebugStateGroup(false);
//	ui->send->setEnabled(true);
}

void Main::initDebugger()
{
//	ui->send->setEnabled(false);
	if (m_history.size())
	{
		alterDebugStateGroup(true);
		ui->debugCode->setEnabled(false);
		ui->debugTimeline->setMinimum(0);
		ui->debugTimeline->setMaximum(m_history.size());
		ui->debugTimeline->setValue(0);
	}
}

void Main::on_debugTimeline_valueChanged()
{
	updateDebugger();
}

QString Main::prettyU256(dev::u256 _n) const
{
	unsigned inc = 0;
	QString raw;
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
		QString n = pretty(a);
		if (n.isNull())
			s << "<span style=\"color: #844\">0x</span><span style=\"color: #800\">" << a << "</span>";
		else
			s << "<span style=\"font-weight: bold; color: #800\">" << n.toHtmlEscaped().toStdString() << "</span> (<span style=\"color: #844\">0x</span><span style=\"color: #800\">" << a.abridged() << "</span>)";
	}
	else if ((raw = fromRaw((h256)_n, &inc)).size())
		return "<span style=\"color: #484\">\"</span><span style=\"color: #080\">" + raw.toHtmlEscaped() + "</span><span style=\"color: #484\">\"" + (inc ? " + " + QString::number(inc) : "") + "</span>";
	else
		s << "<span style=\"color: #466\">0x</span><span style=\"color: #044\">" << (h256)_n << "</span>";
	return QString::fromStdString(s.str());
}

void Main::updateDebugger()
{
	if (m_history.size())
	{
		WorldState const& nws = m_history[min((int)m_history.size() - 1, ui->debugTimeline->value())];
		WorldState const& ws = ui->callStack->currentRow() > 0 ? *nws.levels[nws.levels.size() - ui->callStack->currentRow()] : nws;

		if (ui->debugTimeline->value() >= m_history.size())
		{
			if (ws.gasCost > ws.gas)
				ui->debugMemory->setHtml("<h3>OUT-OF-GAS</h3>");
			else if (ws.inst == Instruction::RETURN && ws.stack.size() >= 2)
			{
				unsigned from = (unsigned)ws.stack.back();
				unsigned size = (unsigned)ws.stack[ws.stack.size() - 2];
				unsigned o = 0;
				bytes out(size, 0);
				for (; o < size && from + o < ws.memory.size(); ++o)
					out[o] = ws.memory[from + o];
				ui->debugMemory->setHtml("<h3>RETURN</h3>" + QString::fromStdString(dev::memDump(out, 16, true)));
			}
			else if (ws.inst == Instruction::STOP)
				ui->debugMemory->setHtml("<h3>STOP</h3>");
			else if (ws.inst == Instruction::SUICIDE && ws.stack.size() >= 1)
				ui->debugMemory->setHtml("<h3>SUICIDE</h3>0x" + QString::fromStdString(toString(right160(ws.stack.back()))));
			else
				ui->debugMemory->setHtml("<h3>EXCEPTION</h3>");

			ostringstream ss;
			ss << dec << "EXIT  |  GAS: " << dec << max<dev::bigint>(0, (dev::bigint)ws.gas - ws.gasCost);
			ui->debugStateInfo->setText(QString::fromStdString(ss.str()));
			ui->debugStorage->setHtml("");
			ui->debugCallData->setHtml("");
			m_lastData = h256();
			ui->callStack->clear();
			m_lastLevels.clear();
			ui->debugCode->clear();
			m_lastCode = h256();
			ui->debugStack->setHtml("");
		}
		else
		{
			if (m_lastLevels != nws.levels || !ui->callStack->count())
			{
				m_lastLevels = nws.levels;
				ui->callStack->clear();
				for (unsigned i = 0; i <= nws.levels.size(); ++i)
				{
					WorldState const& s = i ? *nws.levels[nws.levels.size() - i] : nws;
					ostringstream out;
					out << s.cur.abridged();
					if (i)
						out << " " << instructionInfo(s.inst).name << " @0x" << hex << s.curPC;
					ui->callStack->addItem(QString::fromStdString(out.str()));
				}
			}

			if (ws.code != m_lastCode)
			{
				bytes const& code = m_codes[ws.code];
				QListWidget* dc = ui->debugCode;
				dc->clear();
				m_pcWarp.clear();
				for (unsigned i = 0; i <= code.size(); ++i)
				{
					byte b = i < code.size() ? code[i] : 0;
					try
					{
						QString s = QString::fromStdString(instructionInfo((Instruction)b).name);
						ostringstream out;
						out << hex << setw(4) << setfill('0') << i;
						m_pcWarp[i] = dc->count();
						if (b >= (byte)Instruction::PUSH1 && b <= (byte)Instruction::PUSH32)
						{
							unsigned bc = b - (byte)Instruction::PUSH1 + 1;
							s = "PUSH 0x" + QString::fromStdString(toHex(bytesConstRef(&code[i + 1], bc)));
							i += bc;
						}
						dc->addItem(QString::fromStdString(out.str()) + "  "  + s);
					}
					catch (...)
					{
						cerr << "Unhandled exception!" << endl <<
									boost::current_exception_diagnostic_information();
						break;	// probably hit data segment
					}
				}
				m_lastCode = ws.code;
			}

			if (ws.callData != m_lastData)
			{
				m_lastData = ws.callData;
				if (ws.callData)
				{
					assert(m_codes.count(ws.callData));
					ui->debugCallData->setHtml(QString::fromStdString(dev::memDump(m_codes[ws.callData], 16, true)));
				}
				else
					ui->debugCallData->setHtml("");
			}

			QString stack;
			for (auto i: ws.stack)
				stack.prepend("<div>" + prettyU256(i) + "</div>");
			ui->debugStack->setHtml(stack);
			ui->debugMemory->setHtml(QString::fromStdString(dev::memDump(ws.memory, 16, true)));
			assert(m_codes.count(ws.code));

			if (m_codes[ws.code].size() >= (unsigned)ws.curPC)
			{
				int l = m_pcWarp[(unsigned)ws.curPC];
				ui->debugCode->setCurrentRow(max(0, l - 5));
				ui->debugCode->setCurrentRow(min(ui->debugCode->count() - 1, l + 5));
				ui->debugCode->setCurrentRow(l);
			}
			else
				cwarn << "PC (" << (unsigned)ws.curPC << ") is after code range (" << m_codes[ws.code].size() << ")";

			ostringstream ss;
			ss << dec << "STEP: " << ws.steps << "  |  PC: 0x" << hex << ws.curPC << "  :  " << instructionInfo(ws.inst).name << "  |  ADDMEM: " << dec << ws.newMemSize << " words  |  COST: " << dec << ws.gasCost <<  "  |  GAS: " << dec << ws.gas;
			ui->debugStateInfo->setText(QString::fromStdString(ss.str()));
			stringstream s;
			for (auto const& i: ws.storage)
				s << "@" << prettyU256(i.first).toStdString() << "&nbsp;&nbsp;&nbsp;&nbsp;" << prettyU256(i.second).toStdString() << "<br/>";
			ui->debugStorage->setHtml(QString::fromStdString(s.str()));
		}
	}
}

void Main::on_post_clicked()
{
	shh::Message m;
	m.setTo(stringToPublic(ui->shhTo->currentText()));
	m.setPayload(parseData(ui->shhData->toPlainText().toStdString()));
	Public f = stringToPublic(ui->shhFrom->currentText());
	Secret from;
	if (m_server->ids().count(f))
		from = m_server->ids().at(f);
	whisper()->inject(m.seal(from, topicFromText(ui->shhTopic->toPlainText()), ui->shhTtl->value(), ui->shhWork->value()));
}

string Main::lookupNatSpec(dev::h256 const& _contractHash) const
{
	return m_natspecDB.retrieve(_contractHash);
}

string Main::lookupNatSpecUserNotice(dev::h256 const& _contractHash, dev::bytes const& _transactionData)
{
	return m_natspecDB.getUserNotice(_contractHash, _transactionData);
}

void Main::refreshWhispers()
{
	ui->whispers->clear();
	for (auto const& w: whisper()->all())
	{
		shh::Envelope const& e = w.second;
		shh::Message m;
		for (pair<Public, Secret> const& i: m_server->ids())
			if (!!(m = e.open(shh::FullTopic(), i.second)))
				break;
		if (!m)
			m = e.open(shh::FullTopic());

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
