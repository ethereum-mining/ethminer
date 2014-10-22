#pragma once

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QList>
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonEth.h>

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

inline dev::Address toAddress(QString const& _s) { return toFixed<20>(_s); }
inline dev::Secret toSecret(QString const& _s) { return toFixed<32>(_s); }
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
	QWhisper(QObject* _p, std::shared_ptr<dev::shh::Interface> const& _c);
	virtual ~QWhisper();

	std::shared_ptr<dev::shh::Interface> face() const;
	void setFace(std::shared_ptr<dev::shh::Interface> const& _c) { m_face = _c; }

	/// Call when the face() is going to be deleted to make this object useless but safe.
	void faceDieing();

	Q_INVOKABLE QWhisper* self() { return this; }

	/// Basic message send.
	Q_INVOKABLE void send(QString /*dev::Address*/ _dest, QString /*ev::KeyPair*/ _from, QString /*dev::h256 const&*/ _topic, QString /*dev::bytes const&*/ _payload);

	// Watches interface

	Q_INVOKABLE unsigned newWatch(QString _json);
	Q_INVOKABLE QString watchMessages(unsigned _w);
	Q_INVOKABLE void killWatch(unsigned _w);
	void clearWatches();

public slots:
	/// Check to see if anything has changed, fire off signals if so.
	/// @note Must be called in the QObject's thread.
	void poll();

signals:
	void watchChanged(unsigned _w);

private:
	std::weak_ptr<dev::shh::Interface> m_face;
	std::vector<unsigned> m_watches;
};

// TODO: add p2p object
#define QETH_INSTALL_JS_NAMESPACE(_frame, _env, _dev, _eth, _shh) [_frame, _env, _dev, _eth, _shh]() \
{ \
	_frame->disconnect(); \
	_frame->addToJavaScriptWindowObject("env", _env, QWebFrame::QtOwnership); \
	_frame->addToJavaScriptWindowObject("dev", _dev, QWebFrame::ScriptOwnership); \
	if (_eth) \
	{ \
		_frame->addToJavaScriptWindowObject("eth", _eth, QWebFrame::ScriptOwnership); \
		_frame->evaluateJavaScript("eth.makeWatch = function(a) { var ww = eth.newWatch(a); var ret = { w: ww }; ret.uninstall = function() { eth.killWatch(w); }; ret.changed = function(f) { eth.watchChanged.connect(function(nw) { if (nw == ww) f() }); }; ret.messages = function() { return JSON.parse(eth.watchMessages(this.w)) }; return ret; }"); \
		_frame->evaluateJavaScript("eth.watch = function(a) { return eth.makeWatch(JSON.stringify(a)) }"); \
		_frame->evaluateJavaScript("eth.transact = function(a, f) { var r = eth.doTransact(JSON.stringify(a)); if (f) f(r); }"); \
		_frame->evaluateJavaScript("eth.call = function(a, f) { var ret = eth.doCallJson(JSON.stringify(a)); if (f) f(ret); return ret; }"); \
		_frame->evaluateJavaScript("eth.messages = function(a) { return JSON.parse(eth.getMessages(JSON.stringify(a))); }"); \
		_frame->evaluateJavaScript("eth.block = function(a) { return JSON.parse(eth.getBlock(a)); }"); \
		_frame->evaluateJavaScript("eth.transaction = function(a) { return JSON.parse(eth.getTransaction(a)); }"); \
		_frame->evaluateJavaScript("eth.uncle = function(a) { return JSON.parse(eth.getUncle(a)); }"); \
	} \
	if (_shh) \
	{ \
		_frame->addToJavaScriptWindowObject("shh", _shh, QWebFrame::ScriptOwnership); \
		_frame->evaluateJavaScript("shh.makeWatch = function(a) { var ww = shh.newWatch(a); var ret = { w: ww }; ret.uninstall = function() { shh.killWatch(w); }; ret.changed = function(f) { shh.watchChanged.connect(function(nw) { if (nw == ww) f() }); }; ret.messages = function() { return JSON.parse(shh.watchMessages(this.w)) }; return ret; }"); \
		_frame->evaluateJavaScript("shh.watch = function(a) { return shh.makeWatch(JSON.stringify(a)) }"); \
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

