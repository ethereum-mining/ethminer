#ifndef MAIN_H
#define MAIN_H

#include <QtQml/QJSValue>
#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethereum/CommonEth.h>
#include <libethereum/RLP.h>

namespace Ui {
class Main;
}

namespace eth {
class Client;
class State;
}

class QQuickView;
class QQmlEngine;
class QJSEngine;

class QEthereum;
class QAccount;

Q_DECLARE_METATYPE(eth::u256)
Q_DECLARE_METATYPE(eth::Address)
Q_DECLARE_METATYPE(eth::Secret)
Q_DECLARE_METATYPE(eth::KeyPair)
Q_DECLARE_METATYPE(QEthereum*)
Q_DECLARE_METATYPE(QAccount*)

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

	Q_INVOKABLE QVariant fromAddress(QVariant/*eth::Address*/ _a) const { return out((eth::u160)to<eth::Address>(_a)); }
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
};

class QAccount: public QObject
{
	Q_OBJECT

public:
	QAccount(QObject* _p = nullptr);
	virtual ~QAccount();

	Q_INVOKABLE QEthereum* ethereum() const { return m_eth; }
	Q_INVOKABLE QVariant balance() const;
	Q_INVOKABLE double txCount() const;
	Q_INVOKABLE bool isContract() const;

	Q_INVOKABLE QVariant address() { return toQJS(m_address); }

	// TODO: past transactions models.

public slots:
	void setEthereum(QEthereum* _eth);
	void setAddress(QVariant _a) { m_address = to<eth::Address>(_a); changed(); }

signals:
	void changed();
	void ethChanged();

private:
	QEthereum* m_eth = nullptr;
	eth::Address m_address;

	Q_PROPERTY(QVariant balance READ balance NOTIFY changed STORED false)
	Q_PROPERTY(double txCount READ txCount NOTIFY changed STORED false)
	Q_PROPERTY(bool isContract READ isContract NOTIFY changed STORED false)
	Q_PROPERTY(QVariant address READ address WRITE setAddress NOTIFY changed)
	Q_PROPERTY(QEthereum* ethereum READ ethereum WRITE setEthereum NOTIFY ethChanged)
};

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p = nullptr);
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
	Q_PROPERTY(QVariant coinbase READ coinbase WRITE setCoinbase NOTIFY changed)
	Q_PROPERTY(bool listening READ isListening WRITE setListening)
	Q_PROPERTY(bool mining READ isMining WRITE setMining)
};

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

	eth::Client* client() { return m_client.get(); }

	QVector<eth::KeyPair> const& owned() const { return m_myKeys; }
	
private slots:
	void on_connect_triggered();
	void on_mine_triggered();
	void on_send_clicked();
	void on_create_triggered();
	void on_net_triggered();
	void on_verbosity_sliderMoved();
	void on_ourAccounts_doubleClicked();
	void ourAccountsRowsMoved();
	void on_accounts_doubleClicked();
	void on_destination_textChanged();
	void on_data_textChanged();
	void on_idealPeers_valueChanged();
	void on_value_valueChanged() { updateFee(); }
	void on_gas_valueChanged() { updateFee(); }
	void on_valueUnits_currentIndexChanged() { updateFee(); }
	void on_gasPriceUnits_currentIndexChanged() { updateFee(); }
	void on_gasPrice_valueChanged() { updateFee(); }
	void on_log_doubleClicked();
	void on_blocks_currentItemChanged();
	void on_contracts_doubleClicked();
	void on_contracts_currentItemChanged();
	void on_about_triggered();
	void on_nameReg_textChanged();
	void on_preview_triggered() { refresh(true); }
	void on_quit_triggered() { close(); }
	void on_urlEdit_editingFinished();

	void refresh(bool _override = false);
	void refreshNetwork();

signals:
	void changed();	// TODO: manifest

private:
	QString pretty(eth::Address _a) const;

	QString render(eth::Address _a) const;
	eth::Address fromString(QString const& _a) const;

	eth::State const& state() const;

	void updateFee();
	void readSettings();
	void writeSettings();

	bool isCreation() const;
	eth::u256 fee() const;
	eth::u256 total() const;
	eth::u256 value() const;
	eth::u256 gasPrice() const;

	std::unique_ptr<Ui::Main> ui;

	std::unique_ptr<eth::Client> m_client;

	QByteArray m_peers;
	QMutex m_guiLock;
	QTimer* m_refresh;
	QTimer* m_refreshNetwork;
	QStringList m_servers;
	QVector<eth::KeyPair> m_myKeys;
	bool m_keysChanged = false;
	eth::bytes m_data;
	eth::bytes m_init;
	eth::Address m_nameReg;

	unsigned m_backupGas;

	QNetworkAccessManager m_webCtrl;

	QEthereum* m_ethereum;
};

#endif // MAIN_H
