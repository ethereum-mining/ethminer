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
/** @file QEthereum.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#pragma warning(push)
#pragma warning(disable: 4100 4267)
#include <leveldb/db.h>
#pragma warning(pop)

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QList>
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonEth.h>

namespace ldb = leveldb;

namespace dev {
namespace eth {
class Interface;
}
namespace shh {
class Interface;
}
}

class QJSEngine;
class QWebFrame;

class QEthereum;

inline dev::bytes asBytes(QString const& _s)
{
	dev::bytes ret;
	ret.reserve(_s.size());
	for (QChar c: _s)
		ret.push_back(c.cell());
	return ret;
}

inline QString asQString(dev::bytes const& _s)
{
	QString ret;
	ret.reserve(_s.size());
	for (auto c: _s)
		ret.push_back(QChar(c, 0));
	return ret;
}

dev::bytes toBytes(QString const& _s);

QString padded(QString const& _s, unsigned _l, unsigned _r);
QString padded(QString const& _s, unsigned _l);
QString unpadded(QString _s);

template <unsigned N> dev::FixedHash<N> toFixed(QString const& _s)
{
	if (_s.startsWith("0x"))
		// Hex
		return dev::FixedHash<N>(_s.mid(2).toStdString());
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Decimal
		return (typename dev::FixedHash<N>::Arith)(_s.toStdString());
	else
		// Binary
		return dev::FixedHash<N>(asBytes(padded(_s, N)));
}

template <unsigned N> inline boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> toInt(QString const& _s);

inline dev::Address toAddress(QString const& _s) { return toFixed<sizeof(dev::Address)>(_s); }
inline dev::Public toPublic(QString const& _s) { return toFixed<sizeof(dev::Public)>(_s); }
inline dev::Secret toSecret(QString const& _s) { return toFixed<sizeof(dev::Secret)>(_s); }
inline dev::u256 toU256(QString const& _s) { return toInt<32>(_s); }

template <unsigned S> QString toQJS(dev::FixedHash<S> const& _h) { return QString::fromStdString("0x" + toHex(_h.ref())); }
template <unsigned N> QString toQJS(boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N, N, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> const& _n) { return QString::fromStdString("0x" + dev::toHex(dev::toCompactBigEndian(_n))); }
inline QString toQJS(dev::bytes const& _n) { return "0x" + QString::fromStdString(dev::toHex(_n)); }

inline QString toBinary(QString const& _s)
{
	return unpadded(asQString(toBytes(_s)));
}

inline QString toDecimal(QString const& _s)
{
	return QString::fromStdString(dev::toString(toU256(_s)));
}

inline double fromFixed(QString const& _s)
{
	return (double)toU256(_s) / (double)(dev::u256(1) << 128);
}

inline QString toFixed(double _s)
{
	return toQJS(dev::u256(_s * (double)(dev::u256(1) << 128)));
}

inline QString fromBinary(dev::bytes _s, unsigned _padding = 32)
{
	_s.resize(std::max<unsigned>(_s.size(), _padding));
	return QString::fromStdString("0x" + dev::toHex(_s));
}

inline QString fromBinary(QString const& _s, unsigned _padding = 32)
{
	return fromBinary(asBytes(_s), _padding);
}

class QDev: public QObject
{
	Q_OBJECT

public:
	QDev(QObject* _p): QObject(_p) {}
	virtual ~QDev() {}

	Q_INVOKABLE QString sha3(QString _s) const;
	Q_INVOKABLE QString sha3(QString _s1, QString _s2) const;
	Q_INVOKABLE QString sha3(QString _s1, QString _s2, QString _s3) const;
	Q_INVOKABLE QString offset(QString _s, int _offset) const;

	Q_INVOKABLE QString toAscii(QString _s) const { return ::toBinary(_s); }
	Q_INVOKABLE QString fromAscii(QString _s) const { return ::fromBinary(_s, 32); }
	Q_INVOKABLE QString fromAscii(QString _s, unsigned _padding) const { return ::fromBinary(_s, _padding); }
	Q_INVOKABLE QString toDecimal(QString _s) const { return ::toDecimal(_s); }
	Q_INVOKABLE double fromFixed(QString _s) const { return ::fromFixed(_s); }
	Q_INVOKABLE QString toFixed(double _d) const { return ::toFixed(_d); }
};

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p, dev::eth::Interface* _c, QList<dev::KeyPair> _accounts);
	virtual ~QEthereum();

	dev::eth::Interface* client() const;
	void setClient(dev::eth::Interface* _c) { m_client = _c; }

	/// Call when the client() is going to be deleted to make this object useless but safe.
	void clientDieing();

	void setAccounts(QList<dev::KeyPair> const& _l);

	Q_INVOKABLE QEthereum* self() { return this; }

	Q_INVOKABLE QString lll(QString _s) const;

	// [NEW API] - Use this instead.
	Q_INVOKABLE QString/*dev::u256*/ balanceAt(QString/*dev::Address*/ _a, int _block) const;
	Q_INVOKABLE double countAt(QString/*dev::Address*/ _a, int _block) const;
	Q_INVOKABLE QString/*dev::u256*/ stateAt(QString/*dev::Address*/ _a, QString/*dev::u256*/ _p, int _block) const;
	Q_INVOKABLE QString/*dev::u256*/ codeAt(QString/*dev::Address*/ _a, int _block) const;

	Q_INVOKABLE QString/*dev::u256*/ balanceAt(QString/*dev::Address*/ _a) const;
	Q_INVOKABLE double countAt(QString/*dev::Address*/ _a) const;
	Q_INVOKABLE QString/*dev::u256*/ stateAt(QString/*dev::Address*/ _a, QString/*dev::u256*/ _p) const;
	Q_INVOKABLE QString/*dev::u256*/ codeAt(QString/*dev::Address*/ _a) const;

	Q_INVOKABLE QString/*json*/ getBlock(QString _numberOrHash/*unsigned if < number(), hash otherwise*/) const;
	Q_INVOKABLE QString/*json*/ getTransaction(QString _numberOrHash/*unsigned if < number(), hash otherwise*/, int _index) const;
	Q_INVOKABLE QString/*json*/ getUncle(QString _numberOrHash/*unsigned if < number(), hash otherwise*/, int _index) const;

	Q_INVOKABLE QString/*json*/ getMessages(QString _attribs/*json*/) const;

	Q_INVOKABLE QString doTransact(QString _json);
	Q_INVOKABLE QString doCall(QString _json);

	Q_INVOKABLE unsigned newWatch(QString _json);
	Q_INVOKABLE QString watchMessages(unsigned _w);
	Q_INVOKABLE void killWatch(unsigned _w);
	void clearWatches();

	bool isListening() const;
	bool isMining() const;

	QString/*dev::Address*/ coinbase() const;
	QString/*dev::u256*/ gasPrice() const { return toQJS(10 * dev::eth::szabo); }
	QString/*dev::u256*/ number() const;
	int getDefault() const;

	QStringList/*list of dev::Address*/ accounts() const;

	unsigned peerCount() const;

public slots:
	void setCoinbase(QString/*dev::Address*/);
	void setMining(bool _l);
	void setListening(bool _l);
	void setDefault(int _block);

	/// Check to see if anything has changed, fire off signals if so.
	/// @note Must be called in the QObject's thread.
	void poll();

signals:
	void watchChanged(unsigned _w);
	void coinbaseChanged();
	void keysChanged();
	void netChanged();
	void miningChanged();

private:
	Q_PROPERTY(QString number READ number NOTIFY watchChanged)
	Q_PROPERTY(QString coinbase READ coinbase WRITE setCoinbase NOTIFY coinbaseChanged)
	Q_PROPERTY(QString gasPrice READ gasPrice)
	Q_PROPERTY(QStringList accounts READ accounts NOTIFY keysChanged)
	Q_PROPERTY(bool mining READ isMining WRITE setMining NOTIFY netChanged)
	Q_PROPERTY(bool listening READ isListening WRITE setListening NOTIFY netChanged)
	Q_PROPERTY(unsigned peerCount READ peerCount NOTIFY miningChanged)
	Q_PROPERTY(int defaultBlock READ getDefault WRITE setDefault)

	dev::eth::Interface* m_client;
	std::vector<unsigned> m_watches;
	std::map<dev::Address, dev::KeyPair> m_accounts;
};

class QWhisper: public QObject
{
	Q_OBJECT

public:
	QWhisper(QObject* _p, std::shared_ptr<dev::shh::Interface> const& _c, QList<dev::KeyPair> _ids);
	virtual ~QWhisper();

	std::shared_ptr<dev::shh::Interface> face() const;
	void setFace(std::shared_ptr<dev::shh::Interface> const& _c) { m_face = _c; }

	void setIdentities(QList<dev::KeyPair> const& _l);

	/// Call when the face() is going to be deleted to make this object useless but safe.
	void faceDieing();

	Q_INVOKABLE QWhisper* self() { return this; }

	/// Basic message send.
	Q_INVOKABLE void doPost(QString _json);

	Q_INVOKABLE QString newIdentity();
	Q_INVOKABLE bool haveIdentity(QString _id) { return m_ids.count(toPublic(_id)); }

	Q_INVOKABLE QString newGroup(QString _id, QString _who);
	Q_INVOKABLE QString addToGroup(QString _group, QString _who);

	// Watches interface
	Q_INVOKABLE unsigned newWatch(QString _json);
	Q_INVOKABLE void killWatch(unsigned _w);
	Q_INVOKABLE void clearWatches();
	Q_INVOKABLE QString watchMessages(unsigned _w);

	dev::Public makeIdentity();
	std::map<dev::Public, dev::Secret> const& ids() const { return m_ids; }

public slots:
	/// Check to see if anything has changed, fire off signals if so.
	/// @note Must be called in the QObject's thread.
	void poll();

signals:
	void watchChanged(unsigned _w, QString _envelopeJson);
	void idsChanged();
	void newIdToAdd(QString _id);

private:
	std::weak_ptr<dev::shh::Interface> m_face;
	std::map<unsigned, dev::Public> m_watches;

	std::map<dev::Public, dev::Secret> m_ids;
};

class QLDB: public QObject
{
	Q_OBJECT

public:
	QLDB(QObject* _p);
	~QLDB();

	Q_INVOKABLE void put(QString _name, QString _key, QString _value);
	Q_INVOKABLE QString get(QString _name, QString _key);
	Q_INVOKABLE void putString(QString _name, QString _key, QString _value);
	Q_INVOKABLE QString getString(QString _name, QString _key);

private:
	ldb::ReadOptions m_readOptions;
	ldb::WriteOptions m_writeOptions;

	ldb::DB* m_db;
};

// TODO: add p2p object
#define QETH_INSTALL_JS_NAMESPACE(_frame, _env, _web3, _eth, _shh, _ldb) [_frame, _env, _web3, _eth, _shh, _ldb]() \
{ \
	_frame->disconnect(); \
	_frame->addToJavaScriptWindowObject("env", _env, QWebFrame::QtOwnership); \
	_frame->addToJavaScriptWindowObject("web3", _web3, QWebFrame::ScriptOwnership); \
	if (_ldb) \
	{ \
		_frame->addToJavaScriptWindowObject("_web3_dot_db", _ldb, QWebFrame::QtOwnership); \
		_frame->evaluateJavaScript("web3.db = _web3_dot_db"); \
	} \
	if (_eth) \
	{ \
		_frame->addToJavaScriptWindowObject("_web3_dot_eth", _eth, QWebFrame::ScriptOwnership); \
		_frame->evaluateJavaScript("_web3_dot_eth.makeWatch = function(a) { var ww = _web3_dot_eth.newWatch(a); var ret = { w: ww }; ret.uninstall = function() { _web3_dot_eth.killWatch(this.w); }; ret.changed = function(f) { _web3_dot_eth.watchChanged.connect(function(nw) { if (nw == ww) f() }); }; ret.messages = function() { return JSON.parse(_web3_dot_eth.watchMessages(this.w)) }; return ret; }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.watch = function(a) { return _web3_dot_eth.makeWatch(JSON.stringify(a)) }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.transact = function(a, f) { var r = _web3_dot_eth.doTransact(JSON.stringify(a)); if (f) f(r); }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.call = function(a, f) { var ret = _web3_dot_eth.doCallJson(JSON.stringify(a)); if (f) f(ret); return ret; }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.messages = function(a) { return JSON.parse(_web3_dot_eth.getMessages(JSON.stringify(a))); }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.block = function(a) { return JSON.parse(_web3_dot_eth.getBlock(a)); }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.transaction = function(a) { return JSON.parse(_web3_dot_eth.getTransaction(a)); }"); \
		_frame->evaluateJavaScript("_web3_dot_eth.uncle = function(a) { return JSON.parse(_web3_dot_eth.getUncle(a)); }"); \
		_frame->evaluateJavaScript("web3.eth = _web3_dot_eth"); \
	} \
	if (_shh) \
	{ \
		_frame->addToJavaScriptWindowObject("_web3_dot_shh", _shh, QWebFrame::ScriptOwnership); \
		_frame->evaluateJavaScript("_web3_dot_shh.makeWatch = function(json) { var ww = _web3_dot_shh.newWatch(json); var ret = { w: ww }; ret.uninstall = function() { _web3_dot_shh.killWatch(this.w); }; ret.arrived = function(f) { _web3_dot_shh.watchChanged.connect(function(nw, envelope) { if (nw == ww) f(JSON.parse(envelope)) }); var existing = JSON.parse(_web3_dot_shh.watchMessages(this.w)); for (var e in existing) f(existing[e]) }; return ret; }"); \
		_frame->evaluateJavaScript("_web3_dot_shh.watch = function(filter) { return _web3_dot_shh.makeWatch(JSON.stringify(filter)) }"); \
		_frame->evaluateJavaScript("_web3_dot_shh.post = function(message) { return _web3_dot_shh.doPost(JSON.stringify(message)) }"); \
		_frame->evaluateJavaScript("web3.shh = _web3_dot_shh"); \
	} \
}

template <unsigned N> inline boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> toInt(QString const& _s)
{
	if (_s.startsWith("0x"))
		return dev::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(dev::fromHex(_s.toStdString().substr(2)));
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Hex or Decimal
		return boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>(_s.toStdString());
	else
		// Binary
		return dev::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(asBytes(padded(_s, N)));
}

