#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethereum/Common.h>
using eth::u256;	// workaround for Q_PROPERTY which can't handle scoped types.

namespace Ui {
class Main;
}

namespace eth {
class Client;
class State;
}

class QQuickView;
class QEthereum;
class QQmlEngine;
class QJSEngine;

Q_DECLARE_METATYPE(eth::u256)

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

	Q_INVOKABLE QString toString(eth::u256 _t) const { return QString::fromStdString(eth::formatBalance(_t)); }

	Q_INVOKABLE QString test() const { return "Hello"; }
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

	Q_INVOKABLE eth::Address fromString(QString _s) const { return eth::Address(_s.toStdString()); }
	Q_INVOKABLE QString toString(eth::Address _a) const { return QString::fromStdString(eth::asHex(_a.asArray())); }
	Q_INVOKABLE QString toAbridged(eth::Address _a) const { return QString::fromStdString(_a.abridged()); }
};

class QEthereum: public QObject
{
	Q_OBJECT

public:
	QEthereum(QObject* _p = nullptr);
	virtual ~QEthereum();

	eth::Client* client() const { return m_client.get(); }

	static QObject* constructU256Helper(QQmlEngine*, QJSEngine*) { return new U256Helper; }
	static QObject* constructKeyHelper(QQmlEngine*, QJSEngine*) { return new KeyHelper; }

	eth::u256 balance() const;

	Q_INVOKABLE eth::Address address() const;
	Q_INVOKABLE eth::u256 balanceAt(eth::Address _a) const;

	Q_INVOKABLE unsigned peerCount() const;

public slots:
	void transact(eth::Secret _secret, eth::Address _dest, eth::u256 _amount);

signals:
	void changed();

protected:
	virtual void timerEvent(QTimerEvent *);

private:
	Q_PROPERTY(eth::u256 balance READ balance NOTIFY changed)

	std::unique_ptr<eth::Client> m_client;
};

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();
	
private slots:
	void on_connect_triggered();
	void on_mine_triggered();
	void on_create_triggered();
	void on_net_triggered(bool _auto = false);
	void on_about_triggered();
	void on_preview_triggered() { refresh(true); }
	void on_quit_triggered() { close(); }

	void refresh(bool _override = false);
	void refreshNetwork();

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

	QEthereum* m_eth;
};

#endif // MAIN_H
