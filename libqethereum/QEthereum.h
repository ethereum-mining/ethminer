#pragma once

#include <QtCore/QAbstractListModel>
#include <libethereum/CommonEth.h>
#include <libethcore/CommonIO.h>

namespace eth {
class Client;
class State;
}

class QQmlEngine;
class QJSEngine;

class QEthereum;
class QmlAccount;
class QmlEthereum;

extern eth::Client* g_qmlClient;
extern QObject* g_qmlMain;

Q_DECLARE_METATYPE(eth::u256)
Q_DECLARE_METATYPE(eth::Address)
Q_DECLARE_METATYPE(eth::Secret)
Q_DECLARE_METATYPE(eth::KeyPair)
Q_DECLARE_METATYPE(QEthereum*)
Q_DECLARE_METATYPE(QmlAccount*)
Q_DECLARE_METATYPE(QmlEthereum*)

class QmlU256Helper: public QObject
{
	Q_OBJECT

public:
	QmlU256Helper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE eth::u256 add(eth::u256 _a, eth::u256 _b) const { return _a + _b; }
	Q_INVOKABLE eth::u256 sub(eth::u256 _a, eth::u256 _b) const { return _a - _b; }
	Q_INVOKABLE eth::u256 mul(eth::u256 _a, int _b) const { return _a * _b; }
	Q_INVOKABLE eth::u256 mul(int _a, eth::u256 _b) const { return _a * _b; }
	Q_INVOKABLE eth::u256 div(eth::u256 _a, int _b) const { return _a / _b; }

	Q_INVOKABLE eth::u256 wei(double _s) const { return (eth::u256)_s; }
	Q_INVOKABLE eth::u256 szabo(double _s) const { return (eth::u256)(_s * (double)eth::szabo); }
	Q_INVOKABLE eth::u256 finney(double _s) const { return (eth::u256)(_s * (double)eth::finney); }
	Q_INVOKABLE eth::u256 ether(double _s) const { return (eth::u256)(_s * (double)eth::ether); }
	Q_INVOKABLE eth::u256 wei(unsigned _s) const { return (eth::u256)_s; }
	Q_INVOKABLE eth::u256 szabo(unsigned _s) const { return (eth::u256)(_s * eth::szabo); }
	Q_INVOKABLE eth::u256 finney(unsigned _s) const { return (eth::u256)(_s * eth::finney); }
	Q_INVOKABLE eth::u256 ether(unsigned _s) const { return (eth::u256)(_s * eth::ether); }
	Q_INVOKABLE double toWei(eth::u256 _t) const { return (double)_t; }
	Q_INVOKABLE double toSzabo(eth::u256 _t) const { return toWei(_t) / (double)eth::szabo; }
	Q_INVOKABLE double toFinney(eth::u256 _t) const { return toWei(_t) / (double)eth::finney; }
	Q_INVOKABLE double toEther(eth::u256 _t) const { return toWei(_t) / (double)eth::ether; }

	Q_INVOKABLE double value(eth::u256 _t) const { return (double)_t; }

	Q_INVOKABLE QString stringOf(eth::u256 _t) const { return QString::fromStdString(eth::formatBalance(_t)); }
};

class QmlKeyHelper: public QObject
{
	Q_OBJECT

public:
	QmlKeyHelper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE eth::KeyPair create() const { return eth::KeyPair::create(); }
	Q_INVOKABLE eth::Address address(eth::KeyPair _p) const { return _p.address(); }
	Q_INVOKABLE eth::Secret secret(eth::KeyPair _p) const { return _p.secret(); }
	Q_INVOKABLE eth::KeyPair keypair(eth::Secret _k) const { return eth::KeyPair(_k); }

	Q_INVOKABLE bool isNull(eth::Address _a) const { return !_a; }

	Q_INVOKABLE eth::Address addressOf(QString _s) const { return eth::Address(_s.toStdString()); }
	Q_INVOKABLE QString stringOf(eth::Address _a) const { return QString::fromStdString(eth::toHex(_a.asArray())); }
	Q_INVOKABLE QString toAbridged(eth::Address _a) const { return QString::fromStdString(_a.abridged()); }
};

class QmlAccount: public QObject
{
	Q_OBJECT

public:
	QmlAccount(QObject* _p = nullptr);
	virtual ~QmlAccount();

	Q_INVOKABLE QmlEthereum* ethereum() const { return m_eth; }
	Q_INVOKABLE eth::u256 balance() const;
	Q_INVOKABLE double txCount() const;
	Q_INVOKABLE bool isContract() const;
	Q_INVOKABLE eth::Address address() const { return m_address; }

	// TODO: past transactions models.

public slots:
	void setEthereum(QmlEthereum* _eth);
	void setAddress(eth::Address _a) { m_address = _a; }

signals:
	void changed();
	void ethChanged();

private:
	QmlEthereum* m_eth = nullptr;
	eth::Address m_address;

	Q_PROPERTY(eth::u256 balance READ balance NOTIFY changed STORED false)
	Q_PROPERTY(double txCount READ txCount NOTIFY changed STORED false)
	Q_PROPERTY(bool isContract READ isContract NOTIFY changed STORED false)
	Q_PROPERTY(eth::Address address READ address WRITE setAddress NOTIFY changed)
	Q_PROPERTY(QmlEthereum* ethereum READ ethereum WRITE setEthereum NOTIFY ethChanged)
};

class QmlEthereum: public QObject
{
	Q_OBJECT

public:
	QmlEthereum(QObject* _p = nullptr);
	virtual ~QmlEthereum();

	eth::Client* client() const;

	static QObject* constructU256Helper(QQmlEngine*, QJSEngine*) { return new QmlU256Helper; }
	static QObject* constructKeyHelper(QQmlEngine*, QJSEngine*) { return new QmlKeyHelper; }

	Q_INVOKABLE eth::Address coinbase() const;

	Q_INVOKABLE bool isListening() const;
	Q_INVOKABLE bool isMining() const;

	Q_INVOKABLE eth::u256 balanceAt(eth::Address _a) const;
	Q_INVOKABLE double txCountAt(eth::Address _a) const;
	Q_INVOKABLE bool isContractAt(eth::Address _a) const;

	Q_INVOKABLE unsigned peerCount() const;

	Q_INVOKABLE QmlEthereum* self() { return this; }

public slots:
	void transact(eth::Secret _secret, eth::Address _dest, eth::u256 _amount, eth::u256 _gasPrice, eth::u256 _gas, QByteArray _data);
	void transact(eth::Secret _secret, eth::u256 _amount, eth::u256 _gasPrice, eth::u256 _gas, QByteArray _code, QByteArray _init);
	void setCoinbase(eth::Address);
	void setMining(bool _l);

	void setListening(bool _l);

signals:
	void changed();
//	void netChanged();
//	void miningChanged();

private:
	Q_PROPERTY(eth::Address coinbase READ coinbase WRITE setCoinbase NOTIFY changed)
	Q_PROPERTY(bool listening READ isListening WRITE setListening)
	Q_PROPERTY(bool mining READ isMining WRITE setMining)
};

template <class T> T to(QVariant const& _s) { if (_s.type() != QVariant::String) return T(); auto s = _s.toString().toLatin1(); assert(s.size() == sizeof(T)); return *(T*)s.data(); }
template <class T> QVariant toQJS(T const& _s) { QLatin1String ret((char*)&_s, sizeof(T)); assert(QVariant(QString(ret)).toString().toLatin1().size() == sizeof(T)); assert(*(T*)(QVariant(QString(ret)).toString().toLatin1().data()) == _s); return QVariant(QString(ret)); }

class U256Helper: public QObject
{
	Q_OBJECT

public:
	U256Helper(QObject* _p = nullptr): QObject(_p) {}

	static eth::u256 in(QVariant const& _s) { return to<eth::u256>(_s); }
	static QVariant out(eth::u256 const& _s) { return toQJS(_s); }

	Q_INVOKABLE QVariant add(QVariant _a, QVariant _b) const { return out(in(_a) + in(_b)); }
	Q_INVOKABLE QVariant sub(QVariant _a, QVariant _b) const { return out(in(_a) - in(_b)); }
	Q_INVOKABLE QVariant mul(QVariant _a, int _b) const { return out(in(_a) * in(_b)); }
	Q_INVOKABLE QVariant mul(int _a, QVariant _b) const { return out(in(_a) * in(_b)); }
	Q_INVOKABLE QVariant div(QVariant _a, int _b) const { return out(in(_a) / in(_b)); }

	Q_INVOKABLE QVariant wei(double _s) const { return out(eth::u256(_s)); }
	Q_INVOKABLE QVariant szabo(double _s) const { return out(eth::u256(_s * (double)eth::szabo)); }
	Q_INVOKABLE QVariant finney(double _s) const { return out(eth::u256(_s * (double)eth::finney)); }
	Q_INVOKABLE QVariant ether(double _s) const { return out(eth::u256(_s * (double)eth::ether)); }
	Q_INVOKABLE QVariant wei(unsigned _s) const { return value(_s); }
	Q_INVOKABLE QVariant szabo(unsigned _s) const { return out(eth::u256(_s) * eth::szabo); }
	Q_INVOKABLE QVariant finney(unsigned _s) const { return out(eth::u256(_s) * eth::finney); }
	Q_INVOKABLE QVariant ether(unsigned _s) const { return out(eth::u256(_s) * eth::ether); }
	Q_INVOKABLE double toWei(QVariant _t) const { return toValue(_t); }
	Q_INVOKABLE double toSzabo(QVariant _t) const { return toWei(_t) / (double)eth::szabo; }
	Q_INVOKABLE double toFinney(QVariant _t) const { return toWei(_t) / (double)eth::finney; }
	Q_INVOKABLE double toEther(QVariant _t) const { return toWei(_t) / (double)eth::ether; }

	Q_INVOKABLE QVariant value(unsigned _s) const { return out(eth::u256(_s)); }
	Q_INVOKABLE double toValue(QVariant _t) const { return (double)in(_t); }

	Q_INVOKABLE QString ethOf(QVariant _t) const { return QString::fromStdString(eth::formatBalance(in(_t))); }
	Q_INVOKABLE QString stringOf(QVariant _t) const { return QString::fromStdString(eth::toString(in(_t))); }

	Q_INVOKABLE QByteArray bytesOf(QVariant _t) const { eth::h256 b = in(_t); return QByteArray((char const*)&b, sizeof(eth::h256)); }
	Q_INVOKABLE QVariant fromHex(QString _s) const { return out((eth::u256)eth::h256(_s.toStdString())); }

	Q_INVOKABLE QVariant fromAddress(QVariant/*eth::Address*/ _a) const { return out((eth::u160)to<eth::Address>(_a)); }
	Q_INVOKABLE QVariant toAddress(QVariant/*eth::Address*/ _a) const { return toQJS<eth::Address>((eth::u160)in(_a)); }

	Q_INVOKABLE bool isNull(QVariant/*eth::Address*/ _a) const { return !in(_a); }
};

class KeyHelper: public QObject
{
	Q_OBJECT

public:
	KeyHelper(QObject* _p = nullptr): QObject(_p) {}

	static eth::Address in(QVariant const& _s) { return to<eth::Address>(_s); }
	static QVariant out(eth::Address const& _s) { return toQJS(_s); }

	Q_INVOKABLE QVariant/*eth::KeyPair*/ create() const { return toQJS(eth::KeyPair::create()); }
	Q_INVOKABLE QVariant/*eth::Address*/ address(QVariant/*eth::KeyPair*/ _p) const { return out(to<eth::KeyPair>(_p).address()); }
	Q_INVOKABLE QVariant/*eth::Secret*/ secret(QVariant/*eth::KeyPair*/ _p) const { return toQJS(to<eth::KeyPair>(_p).secret()); }
	Q_INVOKABLE QVariant/*eth::KeyPair*/ keypair(QVariant/*eth::Secret*/ _k) const { return toQJS(eth::KeyPair(to<eth::Secret>(_k))); }

	Q_INVOKABLE bool isNull(QVariant/*eth::Address*/ _a) const { return !in(_a); }

	Q_INVOKABLE QVariant/*eth::Address*/ addressOf(QString _s) const { return out(eth::Address(_s.toStdString())); }
	Q_INVOKABLE QString stringOf(QVariant/*eth::Address*/ _a) const { return QString::fromStdString(eth::toHex(in(_a).asArray())); }
	Q_INVOKABLE QString toAbridged(QVariant/*eth::Address*/ _a) const { return QString::fromStdString(in(_a).abridged()); }

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
		eth::h256 ret((uint8_t const*)_s.data(), eth::h256::ConstructFromPointer);
		return toQJS<eth::u256>(ret);
	}
};

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p, eth::Client* _c, QList<eth::KeyPair> _accounts);
	virtual ~QEthereum();

	eth::Client* client() const;

	Q_INVOKABLE QVariant/*eth::Address*/ coinbase() const;

	Q_INVOKABLE bool isListening() const;
	Q_INVOKABLE bool isMining() const;

	Q_INVOKABLE QVariant/*eth::u256*/ balanceAt(QVariant/*eth::Address*/ _a) const;
	Q_INVOKABLE QVariant/*eth::u256*/ storageAt(QVariant/*eth::Address*/ _a, QVariant/*eth::u256*/ _p) const;
	Q_INVOKABLE double txCountAt(QVariant/*eth::Address*/ _a) const;
	Q_INVOKABLE bool isContractAt(QVariant/*eth::Address*/ _a) const;

	Q_INVOKABLE QVariant gasPrice() const { return toQJS(10 * eth::szabo); }

	Q_INVOKABLE QString ethTest() const { return "Hello world!"; }

	Q_INVOKABLE QVariant/*eth::KeyPair*/ key() const;
	Q_INVOKABLE QList<QVariant/*eth::KeyPair*/> keys() const;
	Q_INVOKABLE QVariant/*eth::Address*/ account() const;
	Q_INVOKABLE QList<QVariant/*eth::Address*/> accounts() const;

	Q_INVOKABLE unsigned peerCount() const;

	Q_INVOKABLE QEthereum* self() { return this; }

	Q_INVOKABLE QVariant create(QVariant _secret, QVariant _amount, QByteArray _code, QByteArray _init, QVariant _gas, QVariant _gasPrice);
	Q_INVOKABLE void transact(QVariant _secret, QVariant _amount, QVariant _dest, QByteArray _data, QVariant _gas, QVariant _gasPrice);

	eth::u256 balanceAt(eth::Address _a) const;
	double txCountAt(eth::Address _a) const;
	bool isContractAt(eth::Address _a) const;

public slots:
	void setCoinbase(QVariant/*eth::Address*/);
	void setMining(bool _l);

	void setListening(bool _l);

signals:
	void changed();
//	void netChanged();
//	void miningChanged();

private:
//	Q_PROPERTY(QVariant coinbase READ coinbase WRITE setCoinbase NOTIFY changed)
//	Q_PROPERTY(bool listening READ isListening WRITE setListening)
//	Q_PROPERTY(bool mining READ isMining WRITE setMining)

	eth::Client* m_client;
	QList<eth::KeyPair> m_accounts;
};
