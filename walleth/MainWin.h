#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethereum/Common.h>

namespace Ui {
class Main;
}

namespace eth {
class Client;
class State;
}

class QQuickView;

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

	std::unique_ptr<eth::Client> m_client;

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
};

#endif // MAIN_H
