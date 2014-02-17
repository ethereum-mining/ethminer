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
	void on_destination_textChanged();
	void on_data_textChanged();
	void on_idealPeers_valueChanged();
	void on_value_valueChanged() { updateFee(); }
	void on_valueUnits_currentIndexChanged() { updateFee(); }
	void on_log_doubleClicked();
	void on_about_triggered();
	void on_quit_triggered() { close(); }

	void refresh();

private:
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
	std::unique_ptr<QTimer> m_refresh;
	QStringList m_servers;
	QVector<eth::KeyPair> m_myKeys;
	QStringList m_data;

	QNetworkAccessManager m_webCtrl;
};

#endif // MAIN_H
