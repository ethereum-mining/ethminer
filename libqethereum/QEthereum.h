#pragma once

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QList>
#include <libethential/CommonIO.h>
#include <libethcore/CommonEth.h>

namespace eth {
class Client;
class State;
}

class QJSEngine;
class QWebFrame;

class QEthereum;

inline eth::bytes asBytes(QString const& _s)
{
	eth::bytes ret;
	ret.reserve(_s.size());
	for (QChar c: _s)
		ret.push_back(c.cell());
	return ret;
}

inline QString asQString(eth::bytes const& _s)
{
	QString ret;
	ret.reserve(_s.size());
	for (auto c: _s)
		ret.push_back(QChar(c, 0));
	return ret;
}

eth::bytes toBytes(QString const& _s);

QString padded(QString const& _s, unsigned _l, unsigned _r);
QString padded(QString const& _s, unsigned _l);
QString unpadded(QString _s);

template <unsigned N> eth::FixedHash<N> toFixed(QString const& _s)
{
	if (_s.startsWith("0x"))
		// Hex
		return eth::FixedHash<N>(_s.mid(2).toStdString());
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Decimal
		return (typename eth::FixedHash<N>::Arith)(_s.toStdString());
	else
		// Binary
		return eth::FixedHash<N>(asBytes(padded(_s, N)));
}

template <unsigned N> inline boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> toInt(QString const& _s);

inline eth::Address toAddress(QString const& _s) { return toFixed<20>(_s); }
inline eth::Secret toSecret(QString const& _s) { return toFixed<32>(_s); }
inline eth::u256 toU256(QString const& _s) { return toInt<32>(_s); }

template <unsigned S> QString toQJS(eth::FixedHash<S> const& _h) { return QString::fromStdString("0x" + toHex(_h.ref())); }
template <unsigned N> QString toQJS(boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N, N, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> const& _n) { return QString::fromStdString("0x" + eth::toHex(eth::toCompactBigEndian(_n))); }

inline QString toBinary(QString const& _s)
{
	return asQString(toBytes(_s));
}

inline QString toDecimal(QString const& _s)
{
	return QString::fromStdString(eth::toString(toU256(_s)));
}

inline double toFixed(QString const& _s)
{
	return (double)toU256(_s) / (double)(eth::u256(1) << 128);
}

inline QString fromBinary(eth::bytes const& _s)
{
	return QString::fromStdString("0x" + eth::toHex(_s));
}

inline QString fromBinary(QString const& _s)
{
	return fromBinary(asBytes(_s));
}

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p, eth::Client* _c, QList<eth::KeyPair> _accounts);
	virtual ~QEthereum();

	eth::Client* client() const;
	void setClient(eth::Client* _c) { m_client = _c; }

	/// Call when the client() is going to be deleted to make this object useless but safe.
	void clientDieing();

	void setAccounts(QList<eth::KeyPair> _l) { m_accounts = _l; keysChanged(); }

	Q_INVOKABLE QString ethTest() const { return "Hello world!"; }
	Q_INVOKABLE QEthereum* self() { return this; }

	Q_INVOKABLE QString secretToAddress(QString _s) const;
	Q_INVOKABLE QString lll(QString _s) const;

	Q_INVOKABLE QString sha3(QString _s) const;
	Q_INVOKABLE QString offset(QString _s, int _offset) const;
	Q_INVOKABLE QString pad(QString _s, unsigned _l) const { return padded(_s, _l); }
	Q_INVOKABLE QString pad(QString _s, unsigned _l, unsigned _r) const { return padded(_s, _l, _r); }
	Q_INVOKABLE QString unpad(QString _s) const { return unpadded(_s); }
	Q_INVOKABLE QString toBinary(QString _s) const { return ::toBinary(_s); }
	Q_INVOKABLE QString fromBinary(QString _s) const { return ::fromBinary(_s); }
	Q_INVOKABLE QString toDecimal(QString _s) const { return ::toDecimal(_s); }
	Q_INVOKABLE double toFixed(QString _s) const { return ::toFixed(_s); }

	// [NEW API] - Use this instead.
	Q_INVOKABLE QString/*eth::u256*/ balanceAt(QString/*eth::Address*/ _a, int _block) const;
	Q_INVOKABLE double countAt(QString/*eth::Address*/ _a, int _block) const;
	Q_INVOKABLE QString/*eth::u256*/ stateAt(QString/*eth::Address*/ _a, QString/*eth::u256*/ _p, int _block) const;
	Q_INVOKABLE QString/*eth::u256*/ codeAt(QString/*eth::Address*/ _a, int _block) const;

	Q_INVOKABLE QString/*eth::u256*/ balanceAt(QString/*eth::Address*/ _a) const;
	Q_INVOKABLE double countAt(QString/*eth::Address*/ _a) const;
	Q_INVOKABLE QString/*eth::u256*/ stateAt(QString/*eth::Address*/ _a, QString/*eth::u256*/ _p) const;
	Q_INVOKABLE QString/*eth::u256*/ codeAt(QString/*eth::Address*/ _a) const;

	Q_INVOKABLE QString/*json*/ getMessages(QString _attribs/*json*/) const;

	Q_INVOKABLE QString doCreate(QString _secret, QString _amount, QString _init, QString _gas, QString _gasPrice);
	Q_INVOKABLE void doTransact(QString _secret, QString _amount, QString _dest, QString _data, QString _gas, QString _gasPrice);
	Q_INVOKABLE void doTransact(QString _json);
	Q_INVOKABLE QString doCall(QString _json);

	Q_INVOKABLE unsigned newWatch(QString _json);
	Q_INVOKABLE QString watchMessages(unsigned _w);
	Q_INVOKABLE void killWatch(unsigned _w);
	void clearWatches();

	bool isListening() const;
	bool isMining() const;

	QString/*eth::Address*/ coinbase() const;
	QString/*eth::u256*/ gasPrice() const { return toQJS(10 * eth::szabo); }
	QString/*eth::u256*/ number() const;
	int getDefault() const;

	QString/*eth::KeyPair*/ key() const;
	QStringList/*list of eth::KeyPair*/ keys() const;
	QString/*eth::Address*/ account() const;
	QStringList/*list of eth::Address*/ accounts() const;

	unsigned peerCount() const;

public slots:
	void setCoinbase(QString/*eth::Address*/);
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
	Q_PROPERTY(QString key READ key NOTIFY keysChanged)
	Q_PROPERTY(QStringList keys READ keys NOTIFY keysChanged)
	Q_PROPERTY(bool mining READ isMining WRITE setMining NOTIFY netChanged)
	Q_PROPERTY(bool listening READ isListening WRITE setListening NOTIFY netChanged)
	Q_PROPERTY(unsigned peerCount READ peerCount NOTIFY miningChanged)
	Q_PROPERTY(int defaultBlock READ getDefault WRITE setDefault)

	eth::Client* m_client;
	std::vector<unsigned> m_watches;
	QList<eth::KeyPair> m_accounts;
};

#define QETH_INSTALL_JS_NAMESPACE(frame, eth, env) [frame, eth, env]() \
{ \
	frame->disconnect(); \
	frame->addToJavaScriptWindowObject("env", env, QWebFrame::QtOwnership); \
	frame->addToJavaScriptWindowObject("eth", eth, QWebFrame::ScriptOwnership); \
	frame->evaluateJavaScript("eth.makeWatch = function(a) { var ww = eth.newWatch(a); var ret = { w: ww }; ret.uninstall = function() { eth.killWatch(w); }; ret.changed = function(f) { eth.watchChanged.connect(function(nw) { if (nw == ww) f() }); }; ret.messages = function() { return JSON.parse(eth.watchMessages(this.w)) }; return ret; }"); \
	frame->evaluateJavaScript("eth.watch = function(a) { return eth.makeWatch(JSON.stringify(a)) }"); \
	frame->evaluateJavaScript("eth.watchChain = function() { return eth.makeWatch('chainChanged') }"); \
	frame->evaluateJavaScript("eth.watchPending = function() { return eth.makeWatch('pendingChanged') }"); \
	frame->evaluateJavaScript("eth.create = function(s, v, c, g, p, f) { var v = eth.doCreate(s, v, c, g, p); if (f) f(v) }"); \
	frame->evaluateJavaScript("eth.transact = function(a_s, f_v, t, d, g, p, f) { if (t == null) { eth.doTransact(JSON.stringify(a_s)); if (f_v) f_v(); } else { eth.doTransact(a_s, f_v, t, d, g, p); if (f) f() } }"); \
	frame->evaluateJavaScript("eth.call = function(a, f) { var ret = eth.doCallJson(JSON.stringify(a)); if (f) f(ret); return ret; }"); \
	frame->evaluateJavaScript("eth.messages = function(a) { return JSON.parse(eth.getMessages(JSON.stringify(a))); }"); \
	frame->evaluateJavaScript("eth.transactions = function(a) { env.warn('THIS CALL IS DEPRECATED. USE eth.messages INSTEAD.'); return JSON.parse(eth.getMessages(JSON.stringify(a))); }"); \
	frame->evaluateJavaScript("String.prototype.pad = function(l, r) { return eth.pad(this, l, r) }"); \
	frame->evaluateJavaScript("String.prototype.bin = function() { return eth.toBinary(this) }"); \
	frame->evaluateJavaScript("String.prototype.unbin = function(l) { return eth.fromBinary(this) }"); \
	frame->evaluateJavaScript("String.prototype.unpad = function(l) { return eth.unpad(this) }"); \
	frame->evaluateJavaScript("String.prototype.dec = function() { return eth.toDecimal(this) }"); \
	frame->evaluateJavaScript("String.prototype.fix = function() { return eth.toFixed(this) }"); \
	frame->evaluateJavaScript("String.prototype.sha3 = function() { return eth.sha3(this) }"); \
}

template <unsigned N> inline boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>> toInt(QString const& _s)
{
	if (_s.startsWith("0x"))
		return eth::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(eth::fromHex(_s.toStdString().substr(2)));
	else if (!_s.contains(QRegExp("[^0-9]")))
		// Hex or Decimal
		return boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>(_s.toStdString());
	else
		// Binary
		return eth::fromBigEndian<boost::multiprecision::number<boost::multiprecision::cpp_int_backend<N * 8, N * 8, boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>>(asBytes(padded(_s, N)));
}

