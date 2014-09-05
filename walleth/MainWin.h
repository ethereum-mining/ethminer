#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethcore/CommonEth.h>
#include <libqethereum/QmlEthereum.h>

namespace Ui {
class Main;
}

namespace dev { namespace eth {
class Client;
class State;
}}

class QQuickView;
class QQmlEngine;
class QJSEngine;

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

	dev::eth::Client* client() const { return m_client.get(); }
	
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

protected:
	virtual void timerEvent(QTimerEvent *);

private:
/*	QString pretty(dev::Address _a) const;
	QString render(dev::Address _a) const;
	dev::Address fromString(QString const& _a) const;
*/
	dev::eth::State const& state() const;

	void updateFee();
	void readSettings();
	void writeSettings();

	dev::u256 fee() const;
	dev::u256 total() const;
	dev::u256 value() const;

	std::unique_ptr<Ui::Main> ui;

	QByteArray m_peers;
	QMutex m_guiLock;
	QTimer* m_refresh;
	QTimer* m_refreshNetwork;
	QVector<dev::KeyPair> m_myKeys;
	bool m_keysChanged = false;
	int m_port;
	int m_idealPeers;
	QString m_clientName;
	QStringList m_servers;

	QQuickView* m_view;

	QNetworkAccessManager m_webCtrl;

	std::unique_ptr<dev::eth::Client> m_client;
};

#endif // MAIN_H
