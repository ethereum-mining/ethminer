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
/** @file QEthereum.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <boost/filesystem.hpp>
#include <QtCore/QtCore>
#include <QtWebKitWidgets/QWebFrame>
#include <libdevcrypto/FileSystem.h>
#include <libevmface/Instruction.h>
#include <liblll/Compiler.h>
#include <libethereum/Client.h>
#include <libethereum/EthereumHost.h>
#include <libwhisper/Message.h>
#include <libwhisper/WhisperHost.h>
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
		cwarn << "'" << _s.toStdString() << "': Unrecognised format for number/hash. USE eth.fromAscii() if you mean to convert from ASCII.";
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
		u256 b = 0;
		for (auto a: m_accounts)
			if (client()->balanceAt(a.first) > b)
				t.from = a.first, b = client()->balanceAt(a.first);
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
		ret = toQJS(client()->transact(m_accounts[t.from].secret(), t.value, t.data, t.gas, t.gasPrice));
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

QWhisper::QWhisper(QObject* _p, std::shared_ptr<dev::shh::Interface> const& _c, QList<dev::KeyPair> _ids): QObject(_p), m_face(_c)
{
	setIdentities(_ids);
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

static shh::Message toMessage(QString _json)
{
	shh::Message ret;

	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	if (f.contains("from"))
		ret.setFrom(toPublic(f["from"].toString()));
	if (f.contains("to"))
		ret.setTo(toPublic(f["to"].toString()));
	if (f.contains("payload"))
		ret.setPayload(toBytes(f["payload"].toString()));

	return ret;
}

static shh::Envelope toSealed(QString _json, shh::Message const& _m, Secret _from)
{
	unsigned ttl = 50;
	unsigned workToProve = 50;

	shh::BuildTopic bt;

	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	if (f.contains("ttl"))
		ttl = f["ttl"].toInt();
	if (f.contains("workToProve"))
		workToProve = f["workToProve"].toInt();
	if (f.contains("topic"))
	{
		if (f["topic"].isString())
			bt.shift(asBytes(padded(f["topic"].toString(), 32)));
		else if (f["topic"].isArray())
			for (auto i: f["topic"].toArray())
				bt.shift(asBytes(padded(i.toString(), 32)));
	}
	return _m.seal(_from, bt, ttl, workToProve);
}

void QWhisper::doPost(QString _json)
{
	shh::Message m = toMessage(_json);
	Secret from;

	if (m.from() && m_ids.count(m.from()))
	{
		cwarn << "Silently signing message from identity" << m.from().abridged() << ": User validation hook goes here.";
		// TODO: insert validification hook here.
		from = m_ids[m.from()];
	}

	face()->inject(toSealed(_json, m, from));
}

void QWhisper::setIdentities(QList<dev::KeyPair> const& _l)
{
	m_ids.clear();
	for (auto i: _l)
		m_ids[i.pub()] = i.secret();
	emit idsChanged();
}

static pair<shh::TopicMask, Public> toWatch(QString _json)
{
	shh::BuildTopicMask bt(shh::BuildTopicMask::Empty);
	Public to;

	QJsonObject f = QJsonDocument::fromJson(_json.toUtf8()).object();
	if (f.contains("to"))
		to = toPublic(f["to"].toString());

	if (f.contains("topic"))
	{
		if (f["topic"].isString())
			bt.shift(asBytes(padded(f["topic"].toString(), 32)));
		else if (f["topic"].isArray())
			for (auto i: f["topic"].toArray())
				if (i.isString())
					bt.shift(asBytes(padded(i.toString(), 32)));
				else
					bt.shift();
	}
	return make_pair(bt.toTopicMask(), to);
}

// _json contains
// topic: the topic as an array of components, some may be null.
// to: specifies the id to which the message is encrypted. null if broadcast.
unsigned QWhisper::newWatch(QString _json)
{
	auto w = toWatch(_json);
	auto ret = face()->installWatch(w.first);
	m_watches.insert(make_pair(ret, w.second));
	return ret;
}

void QWhisper::killWatch(unsigned _w)
{
	face()->uninstallWatch(_w);
	m_watches.erase(_w);
}

void QWhisper::clearWatches()
{
	for (auto i: m_watches)
		face()->uninstallWatch(i.first);
	m_watches.clear();
}

static QString toJson(h256 const& _h, shh::Envelope const& _e, shh::Message const& _m)
{
	QJsonObject v;
	v["hash"] = toQJS(_h);

	v["expiry"] = (int)_e.expiry();
	v["sent"] = (int)_e.sent();
	v["ttl"] = (int)_e.ttl();
	v["workProved"] = (int)_e.workProved();
	v["topic"] = toQJS(_e.topic());

	v["payload"] = toQJS(_m.payload());
	v["from"] = toQJS(_m.from());
	v["to"] = toQJS(_m.to());

	return QString::fromUtf8(QJsonDocument(v).toJson());
}

QString QWhisper::watchMessages(unsigned _w)
{
	QString ret = "[";
	auto wit = m_watches.find(_w);
	if (wit == m_watches.end())
	{
		cwarn << "watchMessages called with invalid watch id" << _w;
		return "";
	}
	Public p = wit->second;
	if (!p || m_ids.count(p))
		for (h256 const& h: face()->watchMessages(_w))
		{
			auto e = face()->envelope(h);
			shh::Message m;
			if (p)
			{
				cwarn << "Silently decrypting message from identity" << p.abridged() << ": User validation hook goes here.";
				m = e.open(m_ids[p]);
			}
			else
				m = e.open();
			ret.append((ret == "[" ? "" : ",") + toJson(h, e, m));
		}

	return ret + "]";
}

QString QWhisper::newIdentity()
{
	return toQJS(makeIdentity());
}

Public QWhisper::makeIdentity()
{
	KeyPair kp = KeyPair::create();
	emit newIdToAdd(toQJS(kp.sec()));
	return kp.pub();
}

QString QWhisper::newGroup(QString _me, QString _others)
{
	(void)_me;
	(void)_others;
	return "";
}

QString QWhisper::addToGroup(QString _group, QString _who)
{
	(void)_group;
	(void)_who;
	return "";
}

void QWhisper::poll()
{
	for (auto const& w: m_watches)
		if (!w.second || m_ids.count(w.second))
			for (h256 const& h: face()->checkWatch(w.first))
			{
				auto e = face()->envelope(h);
				shh::Message m;
				if (w.second)
				{
					cwarn << "Silently decrypting message from identity" << w.second.abridged() << ": User validation hook goes here.";
					m = e.open(m_ids[w.second]);
					if (!m)
						continue;
				}
				else
					m = e.open();
				emit watchChanged(w.first, toJson(h, e, m));
			}
}

#include <libdevcrypto/FileSystem.h>

QLDB::QLDB(QObject* _p): QObject(_p)
{
	auto path = getDataDir() + "/.web3";
	boost::filesystem::create_directories(path);
	ldb::Options o;
	o.create_if_missing = true;
	ldb::DB::Open(o, path, &m_db);
}

QLDB::~QLDB()
{
}

void QLDB::put(QString _p, QString _k, QString _v)
{
	bytes k = sha3(_p.toStdString()).asBytes() + sha3(_k.toStdString()).asBytes();
	bytes v = toBytes(_v);
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)v.data(), v.size()));
}

QString QLDB::get(QString _p, QString _k)
{
	bytes k = sha3(_p.toStdString()).asBytes() + sha3(_k.toStdString()).asBytes();
	string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return toQJS(dev::asBytes(ret));
}

void QLDB::putString(QString _p, QString _k, QString _v)
{
	bytes k = sha3(_p.toStdString()).asBytes() + sha3(_k.toStdString()).asBytes();
	string v = _v.toStdString();
	m_db->Put(m_writeOptions, ldb::Slice((char const*)k.data(), k.size()), ldb::Slice((char const*)v.data(), v.size()));
}

QString QLDB::getString(QString _p, QString _k)
{
	bytes k = sha3(_p.toStdString()).asBytes() + sha3(_k.toStdString()).asBytes();
	string ret;
	m_db->Get(m_readOptions, ldb::Slice((char const*)k.data(), k.size()), &ret);
	return QString::fromStdString(ret);
}

// extra bits needed to link on VS
#ifdef _MSC_VER

// include moc file, ofuscated to hide from automoc
#include\
"moc_QEthereum.cpp"

#endif
