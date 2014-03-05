#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethereum/CommonEth.h>

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

class U256Helper: public QObject
{
	Q_OBJECT

public:
	U256Helper(QObject* _p = nullptr): QObject(_p) {}

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

class KeyHelper: public QObject
{
	Q_OBJECT

public:
	KeyHelper(QObject* _p = nullptr): QObject(_p) {}

	Q_INVOKABLE eth::KeyPair create() const { return eth::KeyPair::create(); }
	Q_INVOKABLE eth::Address address(eth::KeyPair _p) const { return _p.address(); }
	Q_INVOKABLE eth::Secret secret(eth::KeyPair _p) const { return _p.secret(); }
	Q_INVOKABLE eth::KeyPair keypair(eth::Secret _k) const { return eth::KeyPair(_k); }

	Q_INVOKABLE bool isNull(eth::Address _a) const { return !_a; }

	Q_INVOKABLE eth::Address addressOf(QString _s) const { return eth::Address(_s.toStdString()); }
	Q_INVOKABLE QString stringOf(eth::Address _a) const { return QString::fromStdString(eth::toHex(_a.asArray())); }
	Q_INVOKABLE QString toAbridged(eth::Address _a) const { return QString::fromStdString(_a.abridged()); }
};

class QAccount: public QObject
{
	Q_OBJECT

public:
	QAccount(QObject* _p = nullptr);
	virtual ~QAccount();

	Q_INVOKABLE QEthereum* ethereum() const { return m_eth; }
	Q_INVOKABLE eth::u256 balance() const;
	Q_INVOKABLE double txCount() const;
	Q_INVOKABLE bool isContract() const;

	// TODO: past transactions models.

public slots:
	void setEthereum(QEthereum* _eth);

signals:
	void changed();
	void ethChanged();

private:
	QEthereum* m_eth = nullptr;
	eth::Address m_address;

	Q_PROPERTY(eth::u256 balance READ balance NOTIFY changed STORED false)
	Q_PROPERTY(double txCount READ txCount NOTIFY changed STORED false)
	Q_PROPERTY(bool isContract READ isContract NOTIFY changed STORED false)
	Q_PROPERTY(eth::Address address MEMBER m_address NOTIFY changed)
	Q_PROPERTY(QEthereum* ethereum READ ethereum WRITE setEthereum NOTIFY ethChanged)
};

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p = nullptr);
	virtual ~QEthereum();

	eth::Client* client() const;

	static QObject* constructU256Helper(QQmlEngine*, QJSEngine*) { return new U256Helper; }
	static QObject* constructKeyHelper(QQmlEngine*, QJSEngine*) { return new KeyHelper; }

	Q_INVOKABLE eth::Address coinbase() const;

	Q_INVOKABLE bool isListening() const;
	Q_INVOKABLE bool isMining() const;

	Q_INVOKABLE eth::u256 balanceAt(eth::Address _a) const;
	Q_INVOKABLE double txCountAt(eth::Address _a) const;
	Q_INVOKABLE bool isContractAt(eth::Address _a) const;

	Q_INVOKABLE unsigned peerCount() const;

	Q_INVOKABLE QEthereum* self() { return this; }

public slots:
	void transact(eth::Secret _secret, eth::Address _dest, eth::u256 _amount);

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

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

	eth::Client* client() const { return m_client.get(); }
	
private slots:
	void on_connect_triggered();
	void on_mine_triggered();
	void on_create_triggered();
	void on_net_triggered(bool _auto = false);
	void on_about_triggered();
	void on_preview_triggered() { refresh(); }
	void on_quit_triggered() { close(); }

	void refresh();
	void refreshNetwork();

signals:
	void changed();

protected:
	virtual void timerEvent(QTimerEvent *);

private:
/*	QString pretty(eth::Address _a) const;
	QString render(eth::Address _a) const;
	eth::Address fromString(QString const& _a) const;
*/
	eth::State const& state() const;

	void updateFee();
	void readSettings();
	void writeSettings();

	eth::u256 fee() const;
	eth::u256 total() const;
	eth::u256 value() const;

	std::unique_ptr<Ui::Main> ui;

	QByteArray m_peers;
	QMutex m_guiLock;
	QTimer* m_refresh;
	QTimer* m_refreshNetwork;
	QVector<eth::KeyPair> m_myKeys;
	bool m_keysChanged = false;
	int m_port;
	int m_idealPeers;
	QString m_clientName;
	QStringList m_servers;

	QQuickView* m_view;

	QNetworkAccessManager m_webCtrl;

	std::unique_ptr<eth::Client> m_client;
};

#endif // MAIN_H
