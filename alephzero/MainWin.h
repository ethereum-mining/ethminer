#ifndef MAIN_H
#define MAIN_H

#include <QtNetwork/QNetworkAccessManager>
#include <QAbstractListModel>
#include <QDialog>
#include <QMutex>
#include <libethereum/Client.h>

namespace Ui {
class Main;
}

class Main : public QDialog
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();
	
private slots:
	void on_connect_clicked();
	void on_mine_toggled();
	void on_send_clicked();
	void on_create_clicked();
	void on_net_toggled();

	void refresh();

private:
	void readSettings();
	void writeSettings();

	Ui::Main *ui;

	eth::Client m_client;

	eth::KeyPair m_myKey;
	std::vector<bi::tcp::endpoint> m_peers;

	QMutex m_guiLock;
	QTimer* m_refresh;
	QStringList m_servers;

	QNetworkAccessManager m_webCtrl;
};

#endif // MAIN_H
