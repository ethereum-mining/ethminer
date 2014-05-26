#include <fstream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma warning(push)
#pragma warning(disable:4100)
#include <boost/process.hpp>
#pragma GCC diagnostic pop
#pragma warning(pop)
#include <QtNetwork/QNetworkReply>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWebKitWidgets/QWebFrame>
#include <QtGui/QClipboard>
#include <QtCore/QtCore>
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

// Horrible global for the mainwindow. Needed for the QEthereums to find the Main window which acts as multiplexer for now.
// Can get rid of this once we've sorted out ITC for signalling & multiplexed querying.
Main* g_main = nullptr;

static void initUnits(QComboBox* _b)
{
	for (auto n = (::uint)units().size(); n-- != 0; )
		_b->addItem(QString::fromStdString(units()[n].second), n);
}

string htmlDump(bytes const& _b, unsigned _w = 8)
{
	stringstream ret;
	ret << "<pre style=\"font-family: Monospace, sans-serif; font-size: small\">";
	for (unsigned i = 0; i < _b.size(); i += _w)
	{
		ret << hex << setw(4) << setfill('0') << i << " ";
		for (unsigned j = i; j < i + _w; ++j)
			if (j < _b.size())
				if (_b[j] >= 32 && _b[j] < 128)
					ret << (char)_b[j];
				else ret << '?';
			else
				ret << ' ';
		ret << " ";
		for (unsigned j = i; j < i + _w && j < _b.size(); ++j)
			ret << setfill('0') << setw(2) << hex << (unsigned)_b[j] << " ";
		ret << "\n";
	}
	ret << "</pre>";
	return ret.str();
}

Address c_config = Address("ccdeac59d35627b7de09332e819d5159e7bb7250");

using namespace boost::process;

string findSerpent()
{
	string ret;
	vector<string> paths = { ".", "..", "../cpp-ethereum", "../../cpp-ethereum", "/usr/local/bin", "/usr/bin/" };
	for (auto i: paths)
		if (!ifstream(i + "/serpent_cli.py").fail())
		{
			ret = i + "/serpent_cli.py";
			break;
		}
	if (ret.empty())
		cwarn << "Serpent compiler not found. Please install into the same path as this executable.";
	return ret;
}

bytes compileSerpent(string const& _code)
{
	static const string serpent = findSerpent();
	if (serpent.empty())
		return bytes();

#ifdef _WIN32
	vector<string> args = vector<string>({"python", serpent, "-b", "compile"});
	string exec = "";
#else
	vector<string> args = vector<string>({"serpent_cli.py", "-b", "compile"});
	string exec = serpent;
#endif

	context ctx;
	ctx.environment = self::get_environment();
	ctx.stdin_behavior = capture_stream();
	ctx.stdout_behavior = capture_stream();
	try
	{
		child c = launch(exec, args, ctx);
		postream& os = c.get_stdin();
		pistream& is = c.get_stdout();

		os << _code << "\n";
		os.close();

		string hex;
		int i;
		while ((i = is.get()) != -1 && i != '\r' && i != '\n')
			hex.push_back(i);

		return fromHex(hex);
	}
	catch (boost::system::system_error&)
	{
		cwarn << "Serpent compiler failed to launch.";
		return bytes();
	}
}

Main::Main(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::Main)
{
	g_main = this;

	setWindowFlags(Qt::Window);
	ui->setupUi(this);
	g_logPost = [=](std::string const& s, char const* c) { simpleDebugOut(s, c); ui->log->addItem(QString::fromStdString(s)); };
	m_client.reset(new Client("AlethZero"));

	m_refresh = new QTimer(this);
	connect(m_refresh, SIGNAL(timeout()), SLOT(refresh()));
	m_refresh->start(100);
	m_refreshNetwork = new QTimer(this);
	connect(m_refreshNetwork, SIGNAL(timeout()), SLOT(refreshNetwork()));
	m_refreshNetwork->start(1000);

	connect(ui->ourAccounts->model(), SIGNAL(rowsMoved(const QModelIndex &, int, int, const QModelIndex &, int)), SLOT(ourAccountsRowsMoved()));

#if ETH_DEBUG
	m_servers.append("192.168.0.10:30301");
#else
	int pocnumber = QString(ETH_QUOTED(ETH_VERSION)).section('.', 1, 1).toInt();
	if (pocnumber == 4)
		m_servers.push_back("54.72.31.55:30303");
	else if (pocnumber == 5)
		m_servers.push_back("54.201.28.117:30303");
	else
	{
		connect(&m_webCtrl, &QNetworkAccessManager::finished, [&](QNetworkReply* _r)
		{
			m_servers = QString::fromUtf8(_r->readAll()).split("\n", QString::SkipEmptyParts);
		});
		QNetworkRequest r(QUrl("http://www.ethereum.org/servers.poc" + QString::number(pocnumber) + ".txt"));
		r.setHeader(QNetworkRequest::UserAgentHeader, "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1712.0 Safari/537.36");
		m_webCtrl.get(r);
		srand(time(0));
	}
#endif

	cerr << "State root: " << BlockChain::genesis().stateRoot << endl << "Block Hash: " << sha3(BlockChain::createGenesisBlock()) << endl << "Block RLP: " << RLP(BlockChain::createGenesisBlock()) << endl << "Block Hex: " << toHex(BlockChain::createGenesisBlock()) << endl;
	cerr << "Network protocol version: " << eth::c_protocolVersion << endl;

	ui->configDock->close();

	on_verbosity_sliderMoved();
	initUnits(ui->gasPriceUnits);
	initUnits(ui->valueUnits);
	ui->valueUnits->setCurrentIndex(6);
	ui->gasPriceUnits->setCurrentIndex(4);
	ui->gasPrice->setValue(10);
	on_destination_currentTextChanged();

	statusBar()->addPermanentWidget(ui->balance);
	statusBar()->addPermanentWidget(ui->peerCount);
	statusBar()->addPermanentWidget(ui->blockCount);

	connect(ui->webView, &QWebView::titleChanged, [=]()
	{
		ui->tabWidget->setTabText(0, ui->webView->title());
	});

	connect(ui->webView, &QWebView::loadFinished, [=]()
	{
		this->changed();
	});

	QWebFrame* f = ui->webView->page()->currentFrame();
	connect(f, &QWebFrame::javaScriptWindowObjectCleared, [=](){
		auto qe = new QEthereum(this, m_client.get(), owned());
		qe->setup(f);
	});

	readSettings();
	refresh();

	{
		QSettings s("ethereum", "alethzero");
		if (s.value("splashMessage", true).toBool())
		{
			QMessageBox::information(this, "Here Be Dragons!", "This is proof-of-concept software. The project as a whole is not even at the alpha-testing stage. It here to show you, if you have a technical bent, the sort of thing that might be possible down the line.\nPlease don't blame us if it does something unexpected or if you're underwhelmed with the user-experience. We have great plans for it in terms of UX down the line but right now we just want to get the groundwork sorted. We welcome contributions, be they in code, testing or documentation!\nAfter you close this message it won't appear again.");
			s.setValue("splashMessage", false);
		}
	}
}

Main::~Main()
{
	g_logPost = simpleDebugOut;
	writeSettings();
}

void Main::on_jsInput_returnPressed()
{
	ui->jsInput->setText(ui->webView->page()->currentFrame()->evaluateJavaScript(ui->jsInput->text()).toString());
	ui->jsInput->setSelection(0, ui->jsInput->text().size());
}

QString Main::pretty(eth::Address _a) const
{
	h256 n;

	if (h160 nameReg = (u160)state().storage(c_config, 0))
		n = state().storage(nameReg, (u160)(_a));

	if (!n)
		n = state().storage(m_nameReg, (u160)(_a));

	if (n)
	{
		std::string s((char const*)n.data(), 32);
		if (s.find_first_of('\0') != string::npos)
			s.resize(s.find_first_of('\0'));
		return QString::fromStdString(s);
	}
	return QString();
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
		if (h160 nameReg = (u160)state().storage(c_config, 0))
			if (h256 a = state().storage(nameReg, n))
				return right160(a);

		if (h256 a = state().storage(m_nameReg, n))
			return right160(a);
	}
	if (_a.size() == 40)
		return Address(fromHex(_a.toStdString()));
	else
		return Address();
}

void Main::on_about_triggered()
{
	QMessageBox::about(this, "About AlethZero PoC-" + QString(ETH_QUOTED(ETH_VERSION)).section('.', 1, 1), QString("AlethZero/v" ETH_QUOTED(ETH_VERSION) "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM) "\n" ETH_QUOTED(ETH_COMMIT_HASH)) + (ETH_CLEAN_REPO ? "\nCLEAN" : "\n+ LOCAL CHANGES") + "\n\nBy Gav Wood, 2014.\nBased on a design by Vitalik Buterin.\n\nTeam Ethereum++ includes: Eric Lombrozo, Marko Simovic, Alex Leverington, Tim Hughes and several others.");
}

void Main::on_paranoia_triggered()
{
	m_client->setParanoia(ui->paranoia->isChecked());
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
	s.setValue("paranoia", ui->paranoia->isChecked());
	s.setValue("clientName", ui->clientName->text());
	s.setValue("idealPeers", ui->idealPeers->value());
	s.setValue("port", ui->port->value());

	if (m_client->peerServer())
	{
		bytes d = m_client->peerServer()->savePeers();
		m_peers = QByteArray((char*)d.data(), (int)d.size());

	}
	s.setValue("peers", m_peers);
	s.setValue("nameReg", ui->nameReg->text());

	s.setValue("geometry", saveGeometry());
	s.setValue("windowState", saveState());
}

void Main::readSettings()
{
	QSettings s("ethereum", "alethzero");

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
	ui->upnp->setChecked(s.value("upnp", true).toBool());
	ui->upnp->setChecked(s.value("paranoia", false).toBool());
	ui->clientName->setText(s.value("clientName", "").toString());
	ui->idealPeers->setValue(s.value("idealPeers", ui->idealPeers->value()).toInt());
	ui->port->setValue(s.value("port", ui->port->value()).toInt());
	ui->nameReg->setText(s.value("NameReg", "").toString());
	ui->urlEdit->setText(s.value("url", "http://gavwood.com/gavcoin.html").toString());
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
			m_keysChanged = true;
			update();
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
	ui->webView->setUrl(ui->urlEdit->text());
}

void Main::on_nameReg_textChanged()
{
	string s = ui->nameReg->text().toStdString();
	if (s.size() == 40)
	{
		m_nameReg = Address(fromHex(s));
		refresh(true);
	}
	else
		m_nameReg = Address();
}

void Main::refreshNetwork()
{
	auto ps = m_client->peers();

	ui->peerCount->setText(QString::fromStdString(toString(ps.size())) + " peer(s)");
	ui->peers->clear();
	for (PeerInfo const& i: ps)
		ui->peers->addItem(QString("%3 ms - %1:%2 - %4").arg(i.host.c_str()).arg(i.port).arg(chrono::duration_cast<chrono::milliseconds>(i.lastPing).count()).arg(i.clientVersion.c_str()));
}

eth::State const& Main::state() const
{
	return ui->preview->isChecked() ? m_client->postState() : m_client->state();
}

void Main::refresh(bool _override)
{
	eth::ClientGuard g(m_client.get());
	auto const& st = state();

	bool c = m_client->changed();
	if (c || _override)
	{
		changed();

		auto d = m_client->blockChain().details();
		auto diff = BlockInfo(m_client->blockChain().block()).difficulty;
		ui->blockCount->setText(QString("#%1 @%3 T%2").arg(d.number).arg(toLog2(d.totalDifficulty)).arg(toLog2(diff)));

		auto acs = st.addresses();
		ui->accounts->clear();
		ui->contracts->clear();
		for (auto n = 0; n < 2; ++n)
			for (auto i: acs)
			{
				auto r = render(i.first);
				if (r.contains('(') == !n)
				{
					(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(i.second).c_str()).arg(r).arg((unsigned)state().transactionsFrom(i.first)), ui->accounts))
						->setData(Qt::UserRole, QByteArray((char const*)i.first.data(), Address::size));
					if (st.addressHasCode(i.first))
						(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(i.second).c_str()).arg(r).arg((unsigned)st.transactionsFrom(i.first)), ui->contracts))
							->setData(Qt::UserRole, QByteArray((char const*)i.first.data(), Address::size));

					if (r.contains('('))
					{
						// A namereg address
						QString s = pretty(i.first);
						if (ui->destination->findText(s, Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
							ui->destination->addItem(s);
					}
				}
			}

		for (int i = 0; i < ui->destination->count(); ++i)
			if (ui->destination->itemText(i) != "(Create Contract)" && !fromString(ui->destination->itemText(i)))
				ui->destination->removeItem(i--);

		ui->transactionQueue->clear();
		for (Transaction const& t: m_client->pending())
		{
			QString s = t.receiveAddress ?
				QString("%2 %5> %3: %1 [%4]")
					.arg(formatBalance(t.value).c_str())
					.arg(render(t.safeSender()))
					.arg(render(t.receiveAddress))
					.arg((unsigned)t.nonce)
					.arg(st.addressHasCode(t.receiveAddress) ? '*' : '-') :
				QString("%2 +> %3: %1 [%4]")
					.arg(formatBalance(t.value).c_str())
					.arg(render(t.safeSender()))
					.arg(render(right160(sha3(rlpList(t.safeSender(), t.nonce)))))
					.arg((unsigned)t.nonce);
			ui->transactionQueue->addItem(s);
		}

		ui->blocks->clear();
		auto const& bc = m_client->blockChain();
		for (auto h = bc.currentHash(); h != bc.genesisHash(); h = bc.details(h).parent)
		{
			auto d = bc.details(h);
			QListWidgetItem* blockItem = new QListWidgetItem(QString("#%1 %2").arg(d.number).arg(h.abridged().c_str()), ui->blocks);
			blockItem->setData(Qt::UserRole, QByteArray((char const*)h.data(), h.size));
			int n = 0;
			for (auto const& i: RLP(bc.block(h))[1])
			{
				Transaction t(i[0].data());
				QString s = t.receiveAddress ?
					QString("    %2 %5> %3: %1 [%4]")
						.arg(formatBalance(t.value).c_str())
						.arg(render(t.safeSender()))
						.arg(render(t.receiveAddress))
						.arg((unsigned)t.nonce)
						.arg(st.addressHasCode(t.receiveAddress) ? '*' : '-') :
					QString("    %2 +> %3: %1 [%4]")
						.arg(formatBalance(t.value).c_str())
						.arg(render(t.safeSender()))
						.arg(render(right160(sha3(rlpList(t.safeSender(), t.nonce)))))
						.arg((unsigned)t.nonce);
				QListWidgetItem* txItem = new QListWidgetItem(s, ui->blocks);
				txItem->setData(Qt::UserRole, QByteArray((char const*)h.data(), h.size));
				txItem->setData(Qt::UserRole + 1, n);
				n++;
			}
		}
	}

	if (c || m_keysChanged || _override)
	{
		m_keysChanged = false;
		ui->ourAccounts->clear();
		u256 totalBalance = 0;
		u256 totalGavCoinBalance = 0;
		Address gavCoin = fromString("GavCoin");
		for (auto i: m_myKeys)
		{
			u256 b = st.balance(i.address());
			(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(b).c_str()).arg(render(i.address())).arg((unsigned)st.transactionsFrom(i.address())), ui->ourAccounts))
				->setData(Qt::UserRole, QByteArray((char const*)i.address().data(), Address::size));
			totalBalance += b;

			totalGavCoinBalance += st.storage(gavCoin, (u160)i.address());
		}

		ui->balance->setText(QString::fromStdString(toString(totalGavCoinBalance) + " GAV | " + formatBalance(totalBalance)));
	}
}

void Main::on_transactionQueue_currentItemChanged()
{
	ui->pendingInfo->clear();
	eth::ClientGuard g(m_client.get());

	stringstream s;
	int i = ui->transactionQueue->currentRow();
	if (i >= 0)
	{
		Transaction tx(m_client->postState().pending()[i]);
		auto ss = tx.safeSender();
		h256 th = sha3(rlpList(ss, tx.nonce));
		s << "<h3>" << th << "</h3>";
		s << "From: <b>" << pretty(ss).toStdString() << "</b> " << ss;
		if (tx.isCreation())
			s << "<br/>Creates: <b>" << pretty(right160(th)).toStdString() << "</b> " << right160(th);
		else
			s << "<br/>To: <b>" << pretty(tx.receiveAddress).toStdString() << "</b> " << tx.receiveAddress;
		s << "<br/>Value: <b>" << formatBalance(tx.value) << "</b>";
		s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce << "</b>";
		s << "<br/>Gas price: <b>" << formatBalance(tx.gasPrice) << "</b>";
		s << "<br/>Gas: <b>" << tx.gas << "</b>";
		if (tx.isCreation())
		{
			if (tx.data.size())
				s << "<h4>Code</h4>" << disassemble(tx.data);
		}
		else
		{
			if (tx.data.size())
				s << htmlDump(tx.data, 16);
		}
		s << "<hr/>";

		eth::State fs = m_client->postState().fromPending(i);
		eth::State ts = m_client->postState().fromPending(i + 1);
		eth::StateDiff d = fs.diff(ts);

		s << "Pre: " << fs.rootHash() << "<br/>";
		s << "Post: <b>" << ts.rootHash() << "</b>";

		auto indent = "<code style=\"white-space: pre\">     </code>";
		for (auto const& i: d.accounts)
		{
			s << "<hr/>";

			eth::AccountDiff const& ad = i.second;
			s << "<code style=\"white-space: pre; font-weight: bold\">" << ad.lead() << "  </code>" << " <b>" << render(i.first).toStdString() << "</b>";
			if (!ad.exist.to())
				continue;

			if (ad.balance)
			{
				s << "<br/>" << indent << "Balance " << std::dec << formatBalance(ad.balance.to());
				s << " <b>" << std::showpos << (((eth::bigint)ad.balance.to()) - ((eth::bigint)ad.balance.from())) << std::noshowpos << "</b>";
			}
			if (ad.nonce)
			{
				s << "<br/>" << indent << "Count #" << std::dec << ad.nonce.to();
				s << " <b>" << std::showpos << (((eth::bigint)ad.nonce.to()) - ((eth::bigint)ad.nonce.from())) << std::noshowpos << "</b>";
			}
			if (ad.code)
			{
				s << "<br/>" << indent << "Code " << std::hex << ad.code.to();
				if (ad.code.from().size())
					 s << " (" << ad.code.from() << ")";
			}

			for (pair<u256, eth::Diff<u256>> const& i: ad.storage)
			{
				s << "<br/><code style=\"white-space: pre\">";
				if (!i.second.from())
					s << " + ";
				else if (!i.second.to())
					s << "XXX";
				else
					s << " * ";
				s << "  </code>";

				if (i.first > u256(1) << 246)
					s << (h256)i.first;
				else if (i.first > u160(1) << 150)
					s << (h160)(u160)i.first;
				else
					s << std::hex << i.first;

				if (!i.second.from())
					s << ": " << std::hex << i.second.to();
				else if (!i.second.to())
					s << " (" << std::hex << i.second.from() << ")";
				else
					s << ": " << std::hex << i.second.to() << " (" << i.second.from() << ")";
			}
		}
	}

	ui->pendingInfo->setHtml(QString::fromStdString(s.str()));
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
}

void Main::on_inject_triggered()
{
	QString s = QInputDialog::getText(this, "Inject Transaction", "Enter transaction dump in hex");
	bytes b = fromHex(s.toStdString());
	m_client->inject(&b);
	refresh();
}

void Main::on_blocks_currentItemChanged()
{
	ui->info->clear();
	eth::ClientGuard g(m_client.get());
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
			s << "<h4>#" << info.number;
			s << "&nbsp;&emsp;&nbsp;<b>" << timestamp << "</b></h4>";
			s << "<br/>D/TD: <b>2^" << log2((double)info.difficulty) << "</b>/<b>2^" << log2((double)details.totalDifficulty) << "</b>";
			s << "&nbsp;&emsp;&nbsp;Children: <b>" << details.children.size() << "</b></h5>";
			s << "<br/>Gas used/limit: <b>" << info.gasUsed << "</b>/<b>" << info.gasLimit << "</b>";
			s << "&nbsp;&emsp;&nbsp;Minimum gas price: <b>" << formatBalance(info.minGasPrice) << "</b>";
			s << "<br/>Coinbase: <b>" << pretty(info.coinbaseAddress).toStdString() << "</b> " << info.coinbaseAddress;
			s << "<br/>Nonce: <b>" << info.nonce << "</b>";
			s << "<br/>Transactions: <b>" << block[1].itemCount() << "</b> @<b>" << info.transactionsRoot << "</b>";
			s << "<br/>Uncles: <b>" << block[2].itemCount() << "</b> @<b>" << info.sha3Uncles << "</b>";
			s << "<br/>Pre: <b>" << BlockInfo(m_client->blockChain().block(info.parentHash)).stateRoot << "</b>";
			for (auto const& i: block[1])
				s << "<br/>" << sha3(i[0].data()).abridged() << ": <b>" << i[1].toHash<h256>() << "</b> [<b>" << i[2].toInt<u256>() << "</b> used]";
			s << "<br/>Post: <b>" << info.stateRoot << "</b>";
		}
		else
		{
			unsigned txi = item->data(Qt::UserRole + 1).toInt();
			Transaction tx(block[1][txi][0].data());
			auto ss = tx.safeSender();
			h256 th = sha3(rlpList(ss, tx.nonce));
			s << "<h3>" << th << "</h3>";
			s << "<h4>" << h << "[<b>" << txi << "</b>]</h4>";
			s << "<br/>From: <b>" << pretty(ss).toStdString() << "</b> " << ss;
			if (tx.isCreation())
				s << "<br/>Creates: <b>" << pretty(right160(th)).toStdString() << "</b> " << right160(th);
			else
				s << "<br/>To: <b>" << pretty(tx.receiveAddress).toStdString() << "</b> " << tx.receiveAddress;
			s << "<br/>Value: <b>" << formatBalance(tx.value) << "</b>";
			s << "&nbsp;&emsp;&nbsp;#<b>" << tx.nonce << "</b>";
			s << "<br/>Gas price: <b>" << formatBalance(tx.gasPrice) << "</b>";
			s << "<br/>Gas: <b>" << tx.gas << "</b>";
			s << "<br/>V: <b>" << hex << (int)tx.vrs.v << "</b>";
			s << "<br/>R: <b>" << hex << tx.vrs.r << "</b>";
			s << "<br/>S: <b>" << hex << tx.vrs.s << "</b>";
			s << "<br/>Msg: <b>" << tx.sha3(false) << "</b>";
			if (tx.isCreation())
			{
				if (tx.data.size())
					s << "<h4>Code</h4>" << disassemble(tx.data);
			}
			else
			{
				if (tx.data.size())
					s << htmlDump(tx.data, 16);
			}
		}


		ui->info->appendHtml(QString::fromStdString(s.str()));
	}
}

void Main::on_contracts_currentItemChanged()
{
	ui->contractInfo->clear();
	eth::ClientGuard l(&*m_client);
	if (auto item = ui->contracts->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 20);
		auto h = h160((byte const*)hba.data(), h160::ConstructFromPointer);

		stringstream s;
		try
		{
			auto storage = state().storage(h);
			for (auto const& i: storage)
				s << "@" << showbase << hex << i.first << "&nbsp;&nbsp;&nbsp;&nbsp;" << showbase << hex << i.second << "<br/>";
			s << "<h4>Body Code</h4>" << disassemble(state().code(h));
			ui->contractInfo->appendHtml(QString::fromStdString(s.str()));
		}
		catch (eth::InvalidTrie)
		{
			ui->contractInfo->appendHtml("Corrupted trie.");
		}
	}
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
	qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
}

void Main::on_log_doubleClicked()
{
	qApp->clipboard()->setText(ui->log->currentItem()->text());
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

void Main::on_data_textChanged()
{
	if (isCreation())
	{
		QString code = ui->data->toPlainText();
		bytes initBytes;
		bytes bodyBytes;
		auto init = code.indexOf("init:");
		auto body = code.indexOf("body:");
		if (body == -1)
			body = code.indexOf("code:");

		if (body == -1 && init == -1)
		{
			vector<string> errors;
			initBytes = compileLLL(code.toStdString(), &errors);
			for (auto const& i: errors)
				cwarn << i;
		}
		else
		{
			init = (init == -1 ? 0 : (init + 5));
			int initSize = (body == -1 ? code.size() : (body - init));
			body = (body == -1 ? code.size() : (body + 5));
			auto initCode = code.mid(init, initSize).trimmed();
			auto bodyCode = code.mid(body).trimmed();
			if (QRegExp("[^0-9a-fA-F]").indexIn(initCode) == -1)
				initBytes = fromHex(initCode.toStdString());
			else
				initBytes = compileSerpent(initCode.toStdString());
			if (QRegExp("[^0-9a-zA-Z]").indexIn(bodyCode) == -1)
				bodyBytes = fromHex(bodyCode.toStdString());
			else
				bodyBytes = compileSerpent(bodyCode.toStdString());
		}

		m_data.clear();
		if (initBytes.size())
			m_data = initBytes;
		if (bodyBytes.size())
		{
			eth::CodeFragment c(bodyBytes);

			unsigned s = bodyBytes.size();
			unsigned ss = c.appendPush(s);
			unsigned p = m_data.size() + 4 + 2 + 1 + ss + 2 + 1;
			c.appendPush(p);
			c.appendPush(0);
			c.appendInstruction(Instruction::CODECOPY);
			c.appendPush(s);
			c.appendPush(0);
			c.appendInstruction(Instruction::RETURN);
			while (c.size() < p)
				c.appendInstruction(Instruction::STOP);
			for (auto b: c.code())
				m_data.push_back(b);
		}

		ui->code->setHtml("<h4>Code</h4>" + QString::fromStdString(disassemble(m_data)).toHtmlEscaped());
		ui->gas->setMinimum((qint64)state().createGas(m_data.size(), 0));
		if (!ui->gas->isEnabled())
			ui->gas->setValue(m_backupGas);
		ui->gas->setEnabled(true);
	}
	else
	{
		m_data.clear();
		QString s = ui->data->toPlainText();
		while (s.size())
		{
			QRegExp r("(@|\\$)?\"(.*)\"(.*)");
			QRegExp h("(@|\\$)?(0x)?(([a-fA-F0-9])+)(.*)");
			if (r.exactMatch(s))
			{
				for (auto i: r.cap(2))
					m_data.push_back((byte)i.toLatin1());
				if (r.cap(1) != "$")
					for (int i = r.cap(2).size(); i < 32; ++i)
						m_data.push_back(0);
				else
					m_data.push_back(0);
				s = r.cap(3);
			}
			else if (h.exactMatch(s))
			{
				bytes bs = fromHex((((h.cap(3).size() & 1) ? "0" : "") + h.cap(3)).toStdString());
				if (h.cap(1) != "$")
					for (auto i = bs.size(); i < 32; ++i)
						m_data.push_back(0);
				for (auto b: bs)
					m_data.push_back(b);
				s = h.cap(5);
			}
			else
				s = s.mid(1);
		}
		ui->code->setHtml(QString::fromStdString(htmlDump(m_data)));
		if (m_client->postState().addressHasCode(fromString(ui->destination->currentText())))
		{
			ui->gas->setMinimum((qint64)state().callGas(m_data.size(), 1));
			if (!ui->gas->isEnabled())
				ui->gas->setValue(m_backupGas);
			ui->gas->setEnabled(true);
		}
		else
		{
			if (ui->gas->isEnabled())
				m_backupGas = ui->gas->value();
			ui->gas->setValue((qint64)state().callGas(m_data.size()));
			ui->gas->setEnabled(false);
		}
	}
	updateFee();
}

void Main::on_killBlockchain_triggered()
{
	writeSettings();
	ui->mine->setChecked(false);
	ui->net->setChecked(false);
	m_client.reset();
	m_client.reset(new Client("AlethZero", Address(), string(), true));
	readSettings();
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
		if (state().balance(i.address()) >= totalReq)
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
	string n = "AlethZero/v" ETH_QUOTED(ETH_VERSION);
	if (ui->clientName->text().size())
		n += "/" + ui->clientName->text().toStdString();
	n +=  "/" ETH_QUOTED(ETH_BUILD_TYPE) "/" ETH_QUOTED(ETH_BUILD_PLATFORM);
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
	debugFinished();
	u256 totalReq = value() + fee();
	eth::ClientGuard l(&*m_client);
	for (auto i: m_myKeys)
		if (m_client->postState().balance(i.address()) >= totalReq)
		{
			m_client->unlock();
			Secret s = i.secret();
			if (isCreation())
				m_client->transact(s, value(), m_data, ui->gas->value(), gasPrice());
			else
				m_client->transact(s, value(), fromString(ui->destination->currentText()), m_data, ui->gas->value(), gasPrice());
			refresh();
			return;
		}
	statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
}

void Main::on_debug_clicked()
{
	debugFinished();
	u256 totalReq = value() + fee();
	eth::ClientGuard l(&*m_client);
	for (auto i: m_myKeys)
		if (m_client->state().balance(i.address()) >= totalReq)
		{
			m_client->unlock();
			Secret s = i.secret();
			m_client->lock();
			m_executiveState = state();
			m_client->unlock();
			m_currentExecution = unique_ptr<Executive>(new Executive(m_executiveState));
			Transaction t;
			t.nonce = m_executiveState.transactionsFrom(toAddress(s));
			t.value = value();
			t.gasPrice = gasPrice();
			t.gas = ui->gas->value();
			t.data = m_data;
			t.receiveAddress = isCreation() ? Address() : fromString(ui->destination->currentText());
			t.sign(s);
			auto r = t.rlp();
			m_currentExecution->setup(&r);

			m_pcWarp.clear();
			m_history.clear();
			bool ok = true;
			while (ok)
			{
				m_history.append(WorldState({m_currentExecution->vm().curPC(), m_currentExecution->vm().gas(), m_currentExecution->vm().stack(), m_currentExecution->vm().memory(), m_currentExecution->state().storage(m_currentExecution->ext().myAddress)}));
				ok = !m_currentExecution->go(1);
			}
			initDebugger();
			m_currentExecution.reset();
			updateDebugger();
			return;
		}
	statusBar()->showMessage("Couldn't make transaction: no single account contains at least the required amount.");
}

void Main::on_create_triggered()
{
	m_myKeys.append(KeyPair::create());
	m_keysChanged = true;
}

void Main::on_debugStep_triggered()
{
	ui->debugTimeline->setValue(ui->debugTimeline->value() + 1);
}

void Main::debugFinished()
{
	m_pcWarp.clear();
	m_history.clear();
	ui->debugCode->clear();
	ui->debugStack->clear();
	ui->debugMemory->setHtml("");
	ui->debugStorage->setHtml("");
	ui->debugStateInfo->setText("");
//	ui->send->setEnabled(true);
	ui->debugStep->setEnabled(false);
	ui->debugPanel->setEnabled(false);
}

void Main::initDebugger()
{
//	ui->send->setEnabled(false);
	ui->debugStep->setEnabled(true);
	ui->debugPanel->setEnabled(true);
	ui->debugCode->setEnabled(false);
	ui->debugTimeline->setMinimum(0);
	ui->debugTimeline->setMaximum(m_history.size() - 1);
	ui->debugTimeline->setValue(0);

	QListWidget* dc = ui->debugCode;
	dc->clear();
	if (m_currentExecution)
	{
		for (unsigned i = 0; i <= m_currentExecution->ext().code.size(); ++i)
		{
			byte b = i < m_currentExecution->ext().code.size() ? m_currentExecution->ext().code[i] : 0;
			QString s = c_instructionInfo.at((Instruction)b).name;
			m_pcWarp[i] = dc->count();
			ostringstream out;
			out << hex << setw(4) << setfill('0') << i;
			if (b >= (byte)Instruction::PUSH1 && b <= (byte)Instruction::PUSH32)
			{
				unsigned bc = b - (byte)Instruction::PUSH1 + 1;
				s = "PUSH 0x" + QString::fromStdString(toHex(bytesConstRef(&m_currentExecution->ext().code[i + 1], bc)));
				i += bc;
			}
			dc->addItem(QString::fromStdString(out.str()) + "  "  + s);
		}

	}
}

void Main::on_debugTimeline_valueChanged()
{
	updateDebugger();
}

void Main::updateDebugger()
{
	QListWidget* ds = ui->debugStack;
	ds->clear();

	WorldState const& ws = m_history[ui->debugTimeline->value()];

	for (auto i: ws.stack)
		ds->insertItem(0, QString::fromStdString(toHex(((h256)i).asArray())));
	ui->debugMemory->setHtml(QString::fromStdString(htmlDump(ws.memory, 16)));
	ui->debugCode->setCurrentRow(m_pcWarp[(unsigned)ws.curPC]);
	ostringstream ss;
	ss << hex << "PC: 0x" << ws.curPC << "  |  GAS: 0x" << ws.gas;
	ui->debugStateInfo->setText(QString::fromStdString(ss.str()));

	stringstream s;
	for (auto const& i: ws.storage)
		s << "@" << showbase << hex << i.first << "&nbsp;&nbsp;&nbsp;&nbsp;" << showbase << hex << i.second << "<br/>";
	ui->debugStorage->setHtml(QString::fromStdString(s.str()));
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_MainWin.cpp"

#endif
