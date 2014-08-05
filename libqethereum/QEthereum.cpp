#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libethcore/FileSystem.h>
#include <libethcore/Dagger.h>
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/PeerServer.h>
#include "QEthereum.h"
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

eth::bytes toBytes(QString const& _s)
{
	if (_s.startsWith("0x"))
		// Hex
		return eth::fromHex(_s.mid(2).toStdString());
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Decimal
		return eth::toCompactBigEndian(eth::bigint(_s.toStdString()));
	else
		// Binary
		return asBytes(_s);
}

QString padded(QString const& _s, unsigned _l, unsigned _r)
{
	eth::bytes b = toBytes(_s);
	while (b.size() < _l)
		b.insert(b.begin(), 0);
	while (b.size() < _r)
		b.push_back(0);
	return asQString(eth::asBytes(eth::asString(b).substr(b.size() - max(_l, _r))));
}

//"0xff".bin().unbin()

QString padded(QString const& _s, unsigned _l)
{
	if (_s.startsWith("0x") || !_s.contains(QRegExp("[^0-9]")))
		// Numeric: pad to right
		return padded(_s, _l, _l);
	else
		// Text: pad to the left
		return padded(_s, 0, _l);
}

QString unpadded(QString _s)
{
	while (_s.size() && _s.endsWith(QChar(0)))
		_s.chop(1);
	return _s;
}

QEthereum::QEthereum(QObject* _p, Client* _c, QList<eth::KeyPair> _accounts): QObject(_p), m_client(_c), m_accounts(_accounts)
{
	// required to prevent crash on osx when performing addto/evaluatejavascript calls
	moveToThread(_p->thread());
}

QEthereum::~QEthereum()
{
	clearWatches();
}

void QEthereum::clientDieing()
{
	clearWatches();
	m_client = nullptr;
}

void QEthereum::clearWatches()
{
	if (m_client)
		for (auto i: m_watches)
			m_client->uninstallWatch(i);
	m_watches.clear();
}

QString QEthereum::secretToAddress(QString _s) const
{
	return toQJS(KeyPair(toSecret(_s)).address());
}

Client* QEthereum::client() const
{
	return m_client;
}

QString QEthereum::lll(QString _s) const
{
	return asQString(eth::compileLLL(_s.toStdString()));
}

QString QEthereum::sha3(QString _s) const
{
	return toQJS(eth::sha3(asBytes(_s)));
}

QString QEthereum::offset(QString _s, int _i) const
{
	return toQJS(toU256(_s) + _i);
}

QString QEthereum::coinbase() const
{
	return m_client ? toQJS(client()->address()) : "";
}

QString QEthereum::number() const
{
	return m_client ? QString::number(client()->blockChain().number() + 1) : "";
}

QString QEthereum::account() const
{
	if (m_accounts.empty())
		return toQJS(Address());
	return toQJS(m_accounts[0].address());
}

QStringList QEthereum::accounts() const
{
	QStringList ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.address()));
	return ret;
}

QString QEthereum::key() const
{
	if (m_accounts.empty())
		return toQJS(KeyPair().sec());
	return toQJS(m_accounts[0].sec());
}

QStringList QEthereum::keys() const
{
	QStringList ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.sec()));
	return ret;
}

void QEthereum::setCoinbase(QString _a)
{
	if (m_client && client()->address() != toAddress(_a))
	{
		client()->setAddress(toAddress(_a));
		coinbaseChanged();
	}
}

void QEthereum::setDefault(int _block)
{
	if (m_client)
		m_client->setDefault(_block);
}

int QEthereum::getDefault() const
{
	return m_client ? m_client->getDefault() : 0;
}

QString QEthereum::balanceAt(QString _a) const
{
	return m_client ? toQJS(client()->balanceAt(toAddress(_a))) : "";
}

QString QEthereum::balanceAt(QString _a, int _block) const
{
	return m_client ? toQJS(client()->balanceAt(toAddress(_a), _block)) : "";
}

QString QEthereum::stateAt(QString _a, QString _p) const
{
	return m_client ? toQJS(client()->stateAt(toAddress(_a), toU256(_p))) : "";
}

QString QEthereum::stateAt(QString _a, QString _p, int _block) const
{
	return m_client ? toQJS(client()->stateAt(toAddress(_a), toU256(_p), _block)) : "";
}

QString QEthereum::codeAt(QString _a) const
{
	return m_client ? ::fromBinary(client()->codeAt(toAddress(_a))) : "";
}

QString QEthereum::codeAt(QString _a, int _block) const
{
	return m_client ? ::fromBinary(client()->codeAt(toAddress(_a), _block)) : "";
}

double QEthereum::countAt(QString _a) const
{
	return m_client ? (double)(uint64_t)client()->countAt(toAddress(_a)) : 0;
}

double QEthereum::countAt(QString _a, int _block) const
{
	return m_client ? (double)(uint64_t)client()->countAt(toAddress(_a), _block) : 0;
}

static eth::MessageFilter toMessageFilter(QString _json)
{
	eth::MessageFilter filter;

	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	if (f.contains("earliest"))
		filter.withEarliest(f["earliest"].toInt());
	if (f.contains("latest"))
		filter.withLatest(f["latest"].toInt());
	if (f.contains("max"))
		filter.withMax(f["max"].toInt());
	if (f.contains("skip"))
		filter.withSkip(f["skip"].toInt());
	if (f.contains("from"))
	{
		if (f["from"].isArray())
			for (auto i: f["from"].toArray())
				filter.from(toAddress(i.toString()));
		else
			filter.from(toAddress(f["from"].toString()));
	}
	if (f.contains("to"))
	{
		if (f["to"].isArray())
			for (auto i: f["to"].toArray())
				filter.to(toAddress(i.toString()));
		else
			filter.to(toAddress(f["to"].toString()));
	}
	if (f.contains("altered"))
	{
		if (f["altered"].isArray())
			for (auto i: f["altered"].toArray())
				if (i.isObject())
					filter.altered(toAddress(i.toObject()["id"].toString()), toU256(i.toObject()["at"].toString()));
				else
					filter.altered(toAddress(i.toString()));
		else
			if (f["altered"].isObject())
				filter.altered(toAddress(f["altered"].toObject()["id"].toString()), toU256(f["altered"].toObject()["at"].toString()));
			else
				filter.altered(toAddress(f["altered"].toString()));
	}
	return filter;
}

struct TransactionSkeleton
{
	Secret from;
	Address to;
	u256 value;
	bytes data;
	u256 gas;
	u256 gasPrice;
};

static TransactionSkeleton toTransaction(QString _json)
{
	TransactionSkeleton ret;

	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	if (f.contains("from"))
		ret.from = toSecret(f["from"].toString());
	if (f.contains("to"))
		ret.to = toAddress(f["to"].toString());
	if (f.contains("value"))
		ret.value = toU256(f["value"].toString());
	if (f.contains("gas"))
		ret.gas = toU256(f["gas"].toString());
	if (f.contains("gasPrice"))
		ret.gasPrice = toU256(f["gasPrice"].toString());
	if (f.contains("data"))
	{
		if (f["data"].isString())
			ret.data = toBytes(f["data"].toString());
		else if (f["data"].isArray())
			for (auto i: f["data"].toArray())
				eth::operator +=(ret.data, toBytes(padded(i.toString(), 32)));
		else if (f["dataclose"].isArray())
			for (auto i: f["dataclose"].toArray())
				eth::operator +=(ret.data, toBytes(toBinary(i.toString())));
	}
	return ret;
}

static QString toJson(eth::PastMessages const& _pms)
{
	QJsonArray jsonArray;
	for (eth::PastMessage const& t: _pms)
	{
		QJsonObject v;
		v["input"] = ::fromBinary(t.input);
		v["output"] = ::fromBinary(t.output);
		v["to"] = toQJS(t.to);
		v["from"] = toQJS(t.from);
		v["origin"] = toQJS(t.origin);
		v["timestamp"] = (int)t.timestamp;
		v["coinbase"] = toQJS(t.coinbase);
		v["block"] = toQJS(t.block);
		QJsonArray path;
		for (int i: t.path)
			path.append(i);
		v["path"] = path;
		v["number"] = (int)t.number;
		jsonArray.append(v);
	}
	return QString::fromUtf8(QJsonDocument(jsonArray).toJson());
}

QString QEthereum::getMessages(QString _json) const
{
	return m_client ? toJson(m_client->messages(toMessageFilter(_json))) : "";
}

bool QEthereum::isMining() const
{
	return m_client ? client()->isMining() : false;
}

bool QEthereum::isListening() const
{
	return m_client ? client()->haveNetwork() : false;
}

void QEthereum::setMining(bool _l)
{
	if (m_client)
	{
		if (_l)
			client()->startMining();
		else
			client()->stopMining();
	}
}

void QEthereum::setListening(bool _l)
{
	if (!m_client)
		return;
	if (_l)
		client()->startNetwork();
	else
		client()->stopNetwork();
}

unsigned QEthereum::peerCount() const
{
	return m_client ? (unsigned)client()->peerCount() : 0;
}

QString QEthereum::doCreate(QString _secret, QString _amount, QString _init, QString _gas, QString _gasPrice)
{
	if (!m_client)
		return "";
	auto ret = toQJS(client()->transact(toSecret(_secret), toU256(_amount), toBytes(_init), toU256(_gas), toU256(_gasPrice)));
	client()->flushTransactions();
	return ret;
}

void QEthereum::doTransact(QString _secret, QString _amount, QString _dest, QString _data, QString _gas, QString _gasPrice)
{
	if (!m_client)
		return;
	client()->transact(toSecret(_secret), toU256(_amount), toAddress(_dest), toBytes(_data), toU256(_gas), toU256(_gasPrice));
	client()->flushTransactions();
}

void QEthereum::doTransact(QString _json)
{
	if (!m_client)
		return;
	TransactionSkeleton t = toTransaction(_json);
	if (!t.from && m_accounts.size())
		t.from = m_accounts[0].secret();
	if (!t.gasPrice)
		t.gasPrice = 10 * eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(KeyPair(t.from).address()) / t.gasPrice);
	if (t.to)
		client()->transact(t.from, t.value, t.to, t.data, t.gas, t.gasPrice);
	else
		client()->transact(t.from, t.value, t.data, t.gas, t.gasPrice);
	client()->flushTransactions();
}

QString QEthereum::doCall(QString _json)
{
	if (!m_client)
		return QString();
	TransactionSkeleton t = toTransaction(_json);
	if (!t.to)
		return QString();
	if (!t.from && m_accounts.size())
		t.from = m_accounts[0].secret();
	if (!t.gasPrice)
		t.gasPrice = 10 * eth::szabo;
	if (!t.gas)
		t.gas = client()->balanceAt(KeyPair(t.from).address()) / t.gasPrice;
	bytes out = client()->call(t.from, t.value, t.to, t.data, t.gas, t.gasPrice);
	return asQString(out);
}

unsigned QEthereum::newWatch(QString _json)
{
	if (!m_client)
		return (unsigned)-1;
	unsigned ret;
	if (_json == "chainChanged")
		ret = m_client->installWatch(eth::ChainChangedFilter);
	else if (_json == "pendingChanged")
		ret = m_client->installWatch(eth::PendingChangedFilter);
	else
		ret = m_client->installWatch(toMessageFilter(_json));
	m_watches.push_back(ret);
	return ret;
}

QString QEthereum::watchMessages(unsigned _w)
{
	if (!m_client)
		return "";
	return toJson(m_client->messages(_w));
}

void QEthereum::killWatch(unsigned _w)
{
	if (!m_client)
		return;
	m_client->uninstallWatch(_w);
	std::remove(m_watches.begin(), m_watches.end(), _w);
}

void QEthereum::poll()
{
	if (!m_client)
		return;
	for (auto w: m_watches)
		if (m_client->checkWatch(w))
			emit watchChanged(w);
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
