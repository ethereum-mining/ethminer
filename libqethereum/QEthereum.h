#pragma once

#include <QtCore/QObject>
#include <QtCore/QStringList>
#include <QtCore/QList>
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonEth.h>
#include <jsonrpc/rpc.h>

namespace dev
{
namespace eth
{
class Interface;
}
namespace shh
{
class Interface;
}
namespace p2p
{
class Host;
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

	Q_INVOKABLE QString/*json*/ getBlockImpl(QString _json) const;
	Q_INVOKABLE QString/*json*/ getTransactionImpl(QString _json, int _index) const;
	Q_INVOKABLE QString/*json*/ getUncleImpl(QString _json, int _index) const;

	Q_INVOKABLE QString/*json*/ getMessagesImpl(QString _attribs/*json*/) const;
	Q_INVOKABLE QString doTransactImpl(QString _json);
	Q_INVOKABLE QString doCallImpl(QString _json);

	Q_INVOKABLE unsigned newWatch(QString _json);
	Q_INVOKABLE QString watchMessages(unsigned _w);
	Q_INVOKABLE void killWatch(unsigned _w);
	void clearWatches();

	bool isMining() const;

	QString/*dev::Address*/ coinbase() const;
	QString/*dev::u256*/ gasPrice() const { return toQJS(10 * dev::eth::szabo); }
	unsigned/*dev::u256*/ number() const;
	int getDefault() const;

	QStringList/*list of dev::Address*/ accounts() const;

public slots:
	void setCoinbaseImpl(QString/*dev::Address*/);
	void setMiningImpl(bool _l);
	void setDefault(int _block);

	/// Check to see if anything has changed, fire off signals if so.
	/// @note Must be called in the QObject's thread.
	void poll();

signals:
	void netChanged();
	void watchChanged(unsigned _w);
	void coinbaseChanged();
	void keysChanged();

private:
	Q_PROPERTY(QString coinbase READ coinbase WRITE setCoinbaseImpl NOTIFY coinbaseChanged)
	Q_PROPERTY(bool mining READ isMining WRITE setMiningImpl NOTIFY netChanged)
	Q_PROPERTY(QString gasPrice READ gasPrice)
	Q_PROPERTY(QStringList accounts READ accounts NOTIFY keysChanged)
	Q_PROPERTY(int defaultBlock READ getDefault WRITE setDefault)
	Q_PROPERTY(unsigned number READ number NOTIFY watchChanged)

	dev::eth::Interface* m_client;
	std::vector<unsigned> m_watches;
	std::map<dev::Address, dev::KeyPair> m_accounts;
};

class QPeer2Peer : public QObject
{
	Q_OBJECT
	
public:
	QPeer2Peer(QObject *_p, dev::p2p::Host *_p2p);
	virtual ~QPeer2Peer();
	bool isListening() const;
	void setListeningImpl(bool _l);
	unsigned peerCount() const;
	
signals:
	void netChanged();
	void miningChanged();
	
private:
	Q_PROPERTY(bool listening READ isListening WRITE setListeningImpl NOTIFY netChanged)
	Q_PROPERTY(unsigned peerCount READ peerCount NOTIFY miningChanged)
	
	dev::p2p::Host* m_p2p;
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

class QWebThree: public QObject
{
	Q_OBJECT
	
public:
	QWebThree(QObject* _p);
	virtual ~QWebThree();
	
	void installJSNamespace(QWebFrame* _f);

	Q_INVOKABLE void postData(QString _json);
	
signals:
	void processData(QString _json);
	void send(QString _json);
	
private:
	QObject* m_main = nullptr;
	QWebFrame* m_frame = nullptr;
	QDev* m_dev = nullptr;
	QEthereum* m_ethereum = nullptr;
	QWhisper* m_whisper = nullptr;
	QPeer2Peer* m_p2p = nullptr;
};

class QWebThreeConnector: public QObject, public jsonrpc::AbstractServerConnector
{
	Q_OBJECT
	
public:
	QWebThreeConnector(QWebThree* _q);
	virtual ~QWebThreeConnector();
	
	virtual bool StartListening();
	virtual bool StopListening();
	
	bool virtual SendResponse(std::string const& _response,
							  void* _addInfo = NULL);
	
public slots:
	void onMessage(QString const& _json);
	
private:
	QWebThree* m_qweb;
};


// TODO: p2p object condition
#define QETH_INSTALL_JS_NAMESPACE(_frame, _env, _dev, _eth, _shh, _p2p, qweb) [_frame, _env, _dev, _eth, _shh, _p2p, qweb]() \
{ \
	_frame->disconnect(); \
	_frame->addToJavaScriptWindowObject("_web3", qweb, QWebFrame::ScriptOwnership); \
	_frame->evaluateJavaScript("navigator.qt = _web3;"); \
	_frame->evaluateJavaScript("(function () {" \
							"navigator.qt.handlers = [];" \
							"Object.defineProperty(navigator.qt, 'onmessage', {" \
							"	set: function(handler) {" \
							"		navigator.qt.handlers.push(handler);" \
							"	}" \
							"})" \
							"})()"); \
	_frame->evaluateJavaScript("navigator.qt.send.connect(function (res) {" \
							"navigator.qt.handlers.forEach(function (handler) {" \
							"	handler(res);" \
							"})" \	
							"})"); \
}


