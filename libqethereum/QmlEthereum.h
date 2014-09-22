#pragma once

#include <QtCore/QAbstractListModel>
#if ETH_QTQML
#include <QtQml/QtQml>
#endif
#include <libdevcore/CommonIO.h>
#include <libethcore/CommonEth.h>

namespace dev { namespace eth {
class Client;
class State;
}}

class QQmlEngine;
class QmlAccount;
class QmlEthereum;

extern dev::eth::Client* g_qmlClient;
extern QObject* g_qmlMain;

Q_DECLARE_METATYPE(dev::u256)
Q_DECLARE_METATYPE(dev::Address)
Q_DECLARE_METATYPE(dev::Secret)
Q_DECLARE_METATYPE(dev::KeyPair)
//Q_DECLARE_METATYPE(QmlAccount*)
//Q_DECLARE_METATYPE(QmlEthereum*)

class QmlU256Helper: public QObject
{
	Q_OBJECT

public:
	QmlU256Helper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE dev::u256 add(dev::u256 _a, dev::u256 _b) const { return _a + _b; }
	Q_INVOKABLE dev::u256 sub(dev::u256 _a, dev::u256 _b) const { return _a - _b; }
	Q_INVOKABLE dev::u256 mul(dev::u256 _a, int _b) const { return _a * _b; }
	Q_INVOKABLE dev::u256 mul(int _a, dev::u256 _b) const { return _a * _b; }
	Q_INVOKABLE dev::u256 div(dev::u256 _a, int _b) const { return _a / _b; }

	Q_INVOKABLE dev::u256 wei(double _s) const { return (dev::u256)_s; }
	Q_INVOKABLE dev::u256 szabo(double _s) const { return (dev::u256)(_s * (double)dev::eth::szabo); }
	Q_INVOKABLE dev::u256 finney(double _s) const { return (dev::u256)(_s * (double)dev::eth::finney); }
	Q_INVOKABLE dev::u256 ether(double _s) const { return (dev::u256)(_s * (double)dev::eth::ether); }
	Q_INVOKABLE dev::u256 wei(unsigned _s) const { return (dev::u256)_s; }
	Q_INVOKABLE dev::u256 szabo(unsigned _s) const { return (dev::u256)(_s * dev::eth::szabo); }
	Q_INVOKABLE dev::u256 finney(unsigned _s) const { return (dev::u256)(_s * dev::eth::finney); }
	Q_INVOKABLE dev::u256 ether(unsigned _s) const { return (dev::u256)(_s * dev::eth::ether); }
	Q_INVOKABLE double toWei(dev::u256 _t) const { return (double)_t; }
	Q_INVOKABLE double toSzabo(dev::u256 _t) const { return toWei(_t) / (double)dev::eth::szabo; }
	Q_INVOKABLE double toFinney(dev::u256 _t) const { return toWei(_t) / (double)dev::eth::finney; }
	Q_INVOKABLE double toEther(dev::u256 _t) const { return toWei(_t) / (double)dev::eth::ether; }

	Q_INVOKABLE double value(dev::u256 _t) const { return (double)_t; }

	Q_INVOKABLE QString stringOf(dev::u256 _t) const { return QString::fromStdString(dev::eth::formatBalance(_t)); }
};

class QmlKeyHelper: public QObject
{
	Q_OBJECT

public:
	QmlKeyHelper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE dev::KeyPair create() const { return dev::KeyPair::create(); }
	Q_INVOKABLE dev::Address address(dev::KeyPair _p) const { return _p.address(); }
	Q_INVOKABLE dev::Secret secret(dev::KeyPair _p) const { return _p.secret(); }
	Q_INVOKABLE dev::KeyPair keypair(dev::Secret _k) const { return dev::KeyPair(_k); }

	Q_INVOKABLE bool isNull(dev::Address _a) const { return !_a; }

	Q_INVOKABLE dev::Address addressOf(QString _s) const { return dev::Address(_s.toStdString()); }
	Q_INVOKABLE QString stringOf(dev::Address _a) const { return QString::fromStdString(dev::toHex(_a.asArray())); }
	Q_INVOKABLE QString toAbridged(dev::Address _a) const { return QString::fromStdString(_a.abridged()); }
};
#if 0
class QmlAccount: public QObject
{
	Q_OBJECT

public:
	QmlAccount(QObject* _p = nullptr);
	virtual ~QmlAccount();

	Q_INVOKABLE QmlEthereum* ethereum() const { return m_eth; }
	Q_INVOKABLE dev::u256 balance() const;
	Q_INVOKABLE double txCount() const;
	Q_INVOKABLE bool isContract() const;
	Q_INVOKABLE dev::Address address() const { return m_address; }

	// TODO: past transactions models.

public slots:
	void setEthereum(QmlEthereum* _eth);
	void setAddress(dev::Address _a) { m_address = _a; }

signals:
	void changed();
	void ethChanged();

private:
	QmlEthereum* m_eth = nullptr;
	dev::Address m_address;

	Q_PROPERTY(dev::u256 balance READ balance NOTIFY changed STORED false)
	Q_PROPERTY(double txCount READ txCount NOTIFY changed STORED false)
	Q_PROPERTY(bool isContract READ isContract NOTIFY changed STORED false)
	Q_PROPERTY(dev::Address address READ address WRITE setAddress NOTIFY changed)
	Q_PROPERTY(QmlEthereum* ethereum READ ethereum WRITE setEthereum NOTIFY ethChanged)
};

class QmlEthereum: public QObject
{
	Q_OBJECT

public:
	QmlEthereum(QObject* _p = nullptr);
	virtual ~QmlEthereum();

	dev::eth::Client* client() const;

	static QObject* constructU256Helper(QQmlEngine*, QJSEngine*) { return new QmlU256Helper; }
	static QObject* constructKeyHelper(QQmlEngine*, QJSEngine*) { return new QmlKeyHelper; }

	Q_INVOKABLE dev::Address coinbase() const;

	Q_INVOKABLE bool isListening() const;
	Q_INVOKABLE bool isMining() const;

	Q_INVOKABLE dev::u256 balanceAt(dev::Address _a) const;
	Q_INVOKABLE double txCountAt(dev::Address _a) const;
	Q_INVOKABLE bool isContractAt(dev::Address _a) const;

	Q_INVOKABLE unsigned peerCount() const;

	Q_INVOKABLE QmlEthereum* self() { return this; }

public slots:
	void transact(dev::Secret _secret, dev::Address _dest, dev::u256 _amount, dev::u256 _gasPrice, dev::u256 _gas, QByteArray _data);
	void transact(dev::Secret _secret, dev::u256 _amount, dev::u256 _gasPrice, dev::u256 _gas, QByteArray _init);
	void setCoinbase(dev::Address);
	void setMining(bool _l);

	void setListening(bool _l);

signals:
	void coinbaseChanged();
//	void netChanged();
//	void miningChanged();

private:
	Q_PROPERTY(dev::Address coinbase READ coinbase WRITE setCoinbase NOTIFY coinbaseChanged)
	Q_PROPERTY(bool listening READ isListening WRITE setListening)
	Q_PROPERTY(bool mining READ isMining WRITE setMining)
};
#endif
#if 0
template <class T> T to(QVariant const& _s) { if (_s.type() != QVariant::String) return T(); auto s = _s.toString().toLatin1(); assert(s.size() == sizeof(T)); return *(T*)s.data(); }
template <class T> QVariant toQJS(T const& _s) { QLatin1String ret((char*)&_s, sizeof(T)); assert(QVariant(QString(ret)).toString().toLatin1().size() == sizeof(T)); assert(*(T*)(QVariant(QString(ret)).toString().toLatin1().data()) == _s); return QVariant(QString(ret)); }

class U256Helper: public QObject
{
	Q_OBJECT

public:
	U256Helper(QObject* _p = nullptr): QObject(_p) {}

	static dev::u256 in(QVariant const& _s) { return to<dev::u256>(_s); }
	static QVariant out(dev::u256 const& _s) { return toQJS(_s); }

	Q_INVOKABLE QVariant add(QVariant _a, QVariant _b) const { return out(in(_a) + in(_b)); }
	Q_INVOKABLE QVariant sub(QVariant _a, QVariant _b) const { return out(in(_a) - in(_b)); }
	Q_INVOKABLE QVariant mul(QVariant _a, int _b) const { return out(in(_a) * in(_b)); }
	Q_INVOKABLE QVariant mul(int _a, QVariant _b) const { return out(in(_a) * in(_b)); }
	Q_INVOKABLE QVariant div(QVariant _a, int _b) const { return out(in(_a) / in(_b)); }

	Q_INVOKABLE QVariant wei(double _s) const { return out(dev::u256(_s)); }
	Q_INVOKABLE QVariant szabo(double _s) const { return out(dev::u256(_s * (double)dev::eth::szabo)); }
	Q_INVOKABLE QVariant finney(double _s) const { return out(dev::u256(_s * (double)dev::eth::finney)); }
	Q_INVOKABLE QVariant ether(double _s) const { return out(dev::u256(_s * (double)dev::eth::ether)); }
	Q_INVOKABLE QVariant wei(unsigned _s) const { return value(_s); }
	Q_INVOKABLE QVariant szabo(unsigned _s) const { return out(dev::u256(_s) * dev::eth::szabo); }
	Q_INVOKABLE QVariant finney(unsigned _s) const { return out(dev::u256(_s) * dev::eth::finney); }
	Q_INVOKABLE QVariant ether(unsigned _s) const { return out(dev::u256(_s) * dev::eth::ether); }
	Q_INVOKABLE double toWei(QVariant _t) const { return toValue(_t); }
	Q_INVOKABLE double toSzabo(QVariant _t) const { return toWei(_t) / (double)dev::eth::szabo; }
	Q_INVOKABLE double toFinney(QVariant _t) const { return toWei(_t) / (double)dev::eth::finney; }
	Q_INVOKABLE double toEther(QVariant _t) const { return toWei(_t) / (double)dev::eth::ether; }

	Q_INVOKABLE QVariant value(unsigned _s) const { return out(dev::u256(_s)); }
	Q_INVOKABLE double toValue(QVariant _t) const { return (double)in(_t); }

	Q_INVOKABLE QString ethOf(QVariant _t) const { return QString::fromStdString(dev::eth::formatBalance(in(_t))); }
	Q_INVOKABLE QString stringOf(QVariant _t) const { return QString::fromStdString(dev::eth::toString(in(_t))); }

	Q_INVOKABLE QByteArray bytesOf(QVariant _t) const { dev::h256 b = in(_t); return QByteArray((char const*)&b, sizeof(dev::h256)); }
	Q_INVOKABLE QVariant fromHex(QString _s) const { return out((dev::u256)dev::h256(_s.toStdString())); }

	Q_INVOKABLE QVariant fromAddress(QVariant/*dev::Address*/ _a) const { return out((dev::eth::u160)to<dev::Address>(_a)); }
	Q_INVOKABLE QVariant toAddress(QVariant/*dev::Address*/ _a) const { return toQJS<dev::Address>((dev::eth::u160)in(_a)); }

	Q_INVOKABLE bool isNull(QVariant/*dev::Address*/ _a) const { return !in(_a); }
};

class KeyHelper: public QObject
{
	Q_OBJECT

public:
	KeyHelper(QObject* _p = nullptr): QObject(_p) {}

	static dev::Address in(QVariant const& _s) { return to<dev::Address>(_s); }
	static QVariant out(dev::Address const& _s) { return toQJS(_s); }

	Q_INVOKABLE QVariant/*dev::KeyPair*/ create() const { return toQJS(dev::KeyPair::create()); }
	Q_INVOKABLE QVariant/*dev::Address*/ address(QVariant/*dev::KeyPair*/ _p) const { return out(to<dev::KeyPair>(_p).address()); }
	Q_INVOKABLE QVariant/*dev::Secret*/ secret(QVariant/*dev::KeyPair*/ _p) const { return toQJS(to<dev::KeyPair>(_p).secret()); }
	Q_INVOKABLE QVariant/*dev::KeyPair*/ keypair(QVariant/*dev::Secret*/ _k) const { return toQJS(dev::KeyPair(to<dev::Secret>(_k))); }

	Q_INVOKABLE bool isNull(QVariant/*dev::Address*/ _a) const { return !in(_a); }

	Q_INVOKABLE QVariant/*dev::Address*/ addressOf(QString _s) const { return out(dev::Address(_s.toStdString())); }
	Q_INVOKABLE QString stringOf(QVariant/*dev::Address*/ _a) const { return QString::fromStdString(dev::eth::toHex(in(_a).asArray())); }
	Q_INVOKABLE QString toAbridged(QVariant/*dev::Address*/ _a) const { return QString::fromStdString(in(_a).abridged()); }

};

class BytesHelper: public QObject
{
	Q_OBJECT

public:
	BytesHelper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE QByteArray concat(QVariant _v, QVariant _w) const
	{
		QByteArray ba;
		if (_v.type() == QVariant::ByteArray)
			ba = _v.toByteArray();
		else
			ba = _v.toString().toLatin1();
		QByteArray ba2;
		if (_w.type() == QVariant::ByteArray)
			ba2 = _w.toByteArray();
		else
			ba2 = _w.toString().toLatin1();
		ba.append(ba2);
		return QByteArray(ba);
	}
	Q_INVOKABLE QByteArray concat(QByteArray _v, QByteArray _w) const
	{
		_v.append(_w);
		return _v;
	}
	Q_INVOKABLE QByteArray fromString(QString _s) const
	{
		return _s.toLatin1();
	}
	Q_INVOKABLE QByteArray fromString(QString _s, unsigned _padding) const
	{
		QByteArray b = _s.toLatin1();
		for (unsigned i = b.size(); i < _padding; ++i)
			b.append((char)0);
		b.resize(_padding);
		return b;
	}
	Q_INVOKABLE QString toString(QByteArray _b) const
	{
		while (_b.size() && !_b[_b.size() - 1])
			_b.resize(_b.size() - 1);
		return QString::fromLatin1(_b);
	}
	Q_INVOKABLE QVariant u256of(QByteArray _s) const
	{
		while (_s.size() < 32)
			_s.append((char)0);
		dev::h256 ret((uint8_t const*)_s.data(), dev::h256::ConstructFromPointer);
		return toQJS<dev::u256>(ret);
	}
};
#endif
