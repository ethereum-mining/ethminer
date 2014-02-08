#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QAbstractListModel>
#include <QMainWindow>
#include <QMutex>
#include <libethereum/Common.h>

namespace Ui {
class Main;
}

namespace eth {
class Client;
}

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();
	
private slots:
	void on_connect_triggered();
	void on_mine_triggered();
	void on_send_clicked();
	void on_create_triggered();
	void on_net_triggered();
	void on_verbosity_sliderMoved();
	void on_ourAccounts_doubleClicked();
	void on_accounts_doubleClicked();
	void on_log_doubleClicked();
	void on_quit_triggered() { close(); }

	void refresh();

private:
	void readSettings();
	void writeSettings();

	Ui::Main *ui;

	eth::Client* m_client;

	QMutex m_guiLock;
	QTimer* m_refresh;
	QStringList m_servers;
	QVector<eth::KeyPair> m_myKeys;

	QNetworkAccessManager m_webCtrl;
};

#endif // MAIN_H
