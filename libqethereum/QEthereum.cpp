#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libdevcrypto/FileSystem.h>
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include "QEthereum.h"
using namespace std;
using namespace dev;
using namespace dev::eth;

dev::bytes toBytes(QString const& _s)
{
	if (_s.startsWith("0x"))
		// Hex
		return dev::fromHex(_s.mid(2).toStdString());
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Decimal
		return dev::toCompactBigEndian(dev::bigint(_s.toStdString()));
	else
	{
		// Binary
		cwarn << "THIS FUNCTIONALITY IS DEPRECATED. DO NOT ASSUME ASCII/BINARY-STRINGS WILL BE ACCEPTED. USE eth.fromAscii().";
		return asBytes(_s);
	}
}

QString padded(QString const& _s, unsigned _l, unsigned _r)
{
	dev::bytes b = toBytes(_s);
	while (b.size() < _l)
		b.insert(b.begin(), 0);
	while (b.size() < _r)
		b.push_back(0);
	return asQString(dev::asBytes(dev::asString(b).substr(b.size() - max(_l, _r))));
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

QEthereum::QEthereum(QObject* _p, eth::Interface* _c, QList<dev::KeyPair> _accounts):
	QObject(_p), m_client(_c)
{
	// required to prevent crash on osx when performing addto/evaluatejavascript calls
	moveToThread(_p->thread());
	setAccounts(_accounts);
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

eth::Interface* QEthereum::client() const
{
	return m_client;
}

QString QEthereum::lll(QString _s) const
{
	return toQJS(dev::eth::compileLLL(_s.toStdString()));
}

QString QDev::sha3(QString _s) const
{
	return toQJS(dev::sha3(toBytes(_s)));
}

QString QDev::sha3(QString _s1, QString _s2) const
{
	return toQJS(dev::sha3(asBytes(padded(_s1, 32)) + asBytes(padded(_s2, 32))));
}

QString QDev::sha3(QString _s1, QString _s2, QString _s3) const
{
	return toQJS(dev::sha3(asBytes(padded(_s1, 32)) + asBytes(padded(_s2, 32)) + asBytes(padded(_s3, 32))));
}

QString QDev::offset(QString _s, int _i) const
{
	return toQJS(toU256(_s) + _i);
}

QString QEthereum::coinbase() const
{
	return m_client ? toQJS(client()->address()) : "";
}

QString QEthereum::number() const
{
	return m_client ? QString::number(client()->number() + 1) : "";
}

QStringList QEthereum::accounts() const
{
	QStringList ret;
	for (auto i: m_accounts)
		ret.push_back(toQJS(i.first));
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

static dev::eth::MessageFilter toMessageFilter(QString _json)
{
	dev::eth::MessageFilter filter;

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
	Address from;
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
		ret.from = toAddress(f["from"].toString());
	if (f.contains("to"))
		ret.to = toAddress(f["to"].toString());
	if (f.contains("value"))
		ret.value = toU256(f["value"].toString());
	if (f.contains("gas"))
		ret.gas = toU256(f["gas"].toString());
	if (f.contains("gasPrice"))
		ret.gasPrice = toU256(f["gasPrice"].toString());
	if (f.contains("data") || f.contains("code") || f.contains("dataclose"))
	{
		if (f["data"].isString())
			ret.data = toBytes(f["data"].toString());
		else if (f["code"].isString())
			ret.data = toBytes(f["code"].toString());
		else if (f["data"].isArray())
			for (auto i: f["data"].toArray())
				dev::operator +=(ret.data, asBytes(padded(i.toString(), 32)));
		else if (f["code"].isArray())
			for (auto i: f["code"].toArray())
				dev::operator +=(ret.data, asBytes(padded(i.toString(), 32)));
		else if (f["dataclose"].isArray())
			for (auto i: f["dataclose"].toArray())
				dev::operator +=(ret.data, toBytes(i.toString()));
	}
	return ret;
}

static QString toJson(dev::eth::PastMessages const& _pms)
{
	QJsonArray jsonArray;
	for (dev::eth::PastMessage const& t: _pms)
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

static QString toJson(dev::eth::BlockInfo const& _bi, dev::eth::BlockDetails const& _bd)
{
	QJsonObject v;
	v["hash"] = toQJS(_bi.hash);

	v["parentHash"] = toQJS(_bi.parentHash);
	v["sha3Uncles"] = toQJS(_bi.sha3Uncles);
	v["miner"] = toQJS(_bi.coinbaseAddress);
	v["stateRoot"] = toQJS(_bi.stateRoot);
	v["transactionsRoot"] = toQJS(_bi.transactionsRoot);
	v["difficulty"] = toQJS(_bi.difficulty);
	v["number"] = (int)_bi.number;
	v["minGasPrice"] = toQJS(_bi.minGasPrice);
	v["gasLimit"] = (int)_bi.gasLimit;
	v["gasUsed"] = (int)_bi.gasUsed;
	v["timestamp"] = (int)_bi.timestamp;
	v["extraData"] = ::fromBinary(_bi.extraData);
	v["nonce"] = toQJS(_bi.nonce);

	QJsonArray children;
	for (auto c: _bd.children)
		children.append(toQJS(c));
	v["children"] = children;
	v["totalDifficulty"] = toQJS(_bd.totalDifficulty);
	v["bloom"] = toQJS(_bd.bloom);
	return QString::fromUtf8(QJsonDocument(v).toJson());
}

static QString toJson(dev::eth::BlockInfo const& _bi)
{
	QJsonObject v;
	v["hash"] = toQJS(_bi.hash);

	v["parentHash"] = toQJS(_bi.parentHash);
	v["sha3Uncles"] = toQJS(_bi.sha3Uncles);
	v["miner"] = toQJS(_bi.coinbaseAddress);
	v["stateRoot"] = toQJS(_bi.stateRoot);
	v["transactionsRoot"] = toQJS(_bi.transactionsRoot);
	v["difficulty"] = toQJS(_bi.difficulty);
	v["number"] = (int)_bi.number;
	v["minGasPrice"] = toQJS(_bi.minGasPrice);
	v["gasLimit"] = (int)_bi.gasLimit;
	v["gasUsed"] = (int)_bi.gasUsed;
	v["timestamp"] = (int)_bi.timestamp;
	v["extraData"] = ::fromBinary(_bi.extraData);
	v["nonce"] = toQJS(_bi.nonce);

	return QString::fromUtf8(QJsonDocument(v).toJson());
}

static QString toJson(dev::eth::Transaction const& _bi)
{
	QJsonObject v;
	v["hash"] = toQJS(_bi.sha3());

	v["input"] = ::fromBinary(_bi.data);
	v["to"] = toQJS(_bi.receiveAddress);
	v["from"] = toQJS(_bi.sender());
	v["gas"] = (int)_bi.gas;
	v["gasPrice"] = toQJS(_bi.gasPrice);
	v["nonce"] = (int)_bi.nonce;
	v["value"] = toQJS(_bi.value);

	return QString::fromUtf8(QJsonDocument(v).toJson());
}

QString QEthereum::getUncle(QString _numberOrHash, int _i) const
{
	auto n = toU256(_numberOrHash);
	auto h = n < m_client->number() ? m_client->hashFromNumber((unsigned)n) : ::toFixed<32>(_numberOrHash);
	return m_client ? toJson(m_client->uncle(h, _i)) : "";
}

QString QEthereum::getTransaction(QString _numberOrHash, int _i) const
{
	auto n = toU256(_numberOrHash);
	auto h = n < m_client->number() ? m_client->hashFromNumber((unsigned)n) : ::toFixed<32>(_numberOrHash);
	return m_client ? toJson(m_client->transaction(h, _i)) : "";
}

QString QEthereum::getBlock(QString _numberOrHash) const
{
	auto n = toU256(_numberOrHash);
	auto h = n < m_client->number() ? m_client->hashFromNumber((unsigned)n) : ::toFixed<32>(_numberOrHash);
	return m_client ? toJson(m_client->blockInfo(h), m_client->blockDetails(h)) : "";
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
	return /*m_client ? client()->haveNetwork() :*/ false;
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

void QEthereum::setListening(bool)
{
	if (!m_client)
		return;
/*	if (_l)
		client()->startNetwork();
	else
		client()->stopNetwork();*/
}

void QEthereum::setAccounts(QList<dev::KeyPair> const& _l)
{
	m_accounts.clear();
	for (auto i: _l)
		m_accounts[i.address()] = i.secret();
	keysChanged();
}

unsigned QEthereum::peerCount() const
{
	return /*m_client ? (unsigned)client()->peerCount() :*/ 0;
}

QString QEthereum::doTransact(QString _json)
{
	QString ret;
	if (!m_client)
		return ret;
	TransactionSkeleton t = toTransaction(_json);
	if (!t.from && m_accounts.size())
	{
		auto b = m_accounts.begin()->first;
		for (auto a: m_accounts)
			if (client()->balanceAt(a.first) > client()->balanceAt(b))
				b = a.first;
		t.from = b;
	}
	if (!m_accounts.count(t.from))
		return QString();
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);

	cwarn << "Silently signing transaction from address" << t.from.abridged() << ": User validation hook goes here.";
	if (t.to)
		// TODO: insert validification hook here.
		client()->transact(m_accounts[t.from].secret(), t.value, t.to, t.data, t.gas, t.gasPrice);
	else
		ret = toQJS(client()->transact(t.from, t.value, t.data, t.gas, t.gasPrice));
	client()->flushTransactions();
	return ret;
}

QString QEthereum::doCall(QString _json)
{
	if (!m_client)
		return QString();
	TransactionSkeleton t = toTransaction(_json);
	if (!t.from && m_accounts.size())
	{
		auto b = m_accounts.begin()->first;
		for (auto a: m_accounts)
			if (client()->balanceAt(a.first) > client()->balanceAt(b))
				b = a.first;
		t.from = b;
	}
	if (!m_accounts.count(t.from))
		return QString();
	if (!t.gasPrice)
		t.gasPrice = 10 * dev::eth::szabo;
	if (!t.gas)
		t.gas = min<u256>(client()->gasLimitRemaining(), client()->balanceAt(t.from) / t.gasPrice);
	bytes out = client()->call(m_accounts[t.from].secret(), t.value, t.to, t.data, t.gas, t.gasPrice);
	return asQString(out);
}

unsigned QEthereum::newWatch(QString _json)
{
	if (!m_client)
		return (unsigned)-1;
	unsigned ret;
	if (_json == "chain")
		ret = m_client->installWatch(dev::eth::ChainChangedFilter);
	else if (_json == "pending")
		ret = m_client->installWatch(dev::eth::PendingChangedFilter);
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

// TODO: repot and hook all these up.

QWhisper::QWhisper(QObject* _p, std::shared_ptr<dev::shh::Interface> const& _c): QObject(_p), m_face(_c)
{
}

QWhisper::~QWhisper()
{
}

// probably want a better way of doing this. somehow guarantee that the face() will always be available as long as this object is.
struct NoInterface: public Exception {};

std::shared_ptr<dev::shh::Interface> QWhisper::face() const
{
	auto ret = m_face.lock();
	if (!ret)
		throw NoInterface();
	return ret;
}

void QWhisper::faceDieing()
{

}

void QWhisper::send(QString /*dev::Address*/ _dest, QString /*ev::KeyPair*/ _from, QString /*dev::h256 const&*/ _topic, QString /*dev::bytes const&*/ _payload)
{
	(void)_dest;
	(void)_from;
	(void)_topic;
	(void)_payload;
}

unsigned QWhisper::newWatch(QString _json)
{
	(void)_json;
	return 0;
}

QString QWhisper::watchMessages(unsigned _w)
{
	(void)_w;
	return "";
}

void QWhisper::killWatch(unsigned _w)
{
	(void)_w;
}

void QWhisper::clearWatches()
{
}

void QWhisper::poll()
{
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
