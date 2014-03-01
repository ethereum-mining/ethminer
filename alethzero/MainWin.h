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
	void ourAccountsRowsMoved();
	void on_accounts_doubleClicked();
	void on_destination_textChanged();
	void on_data_textChanged();
	void on_idealPeers_valueChanged();
	void on_value_valueChanged() { updateFee(); }
	void on_valueUnits_currentIndexChanged() { updateFee(); }
	void on_log_doubleClicked();
	void on_blocks_currentItemChanged();
	void on_contracts_doubleClicked();
	void on_contracts_currentItemChanged();
	void on_about_triggered();
	void on_nameReg_textChanged();
	void on_preview_triggered() { refresh(true); }
	void on_quit_triggered() { close(); }

	void refresh(bool _override = false);
	void refreshNetwork();

private:
	QString pretty(eth::Address _a) const;

	QString render(eth::Address _a) const;
	eth::Address fromString(QString const& _a) const;

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
	QStringList m_servers;
	QVector<eth::KeyPair> m_myKeys;
	bool m_keysChanged = false;
	eth::u256s m_data;
	eth::Address m_nameReg;

	QNetworkAccessManager m_webCtrl;
};

#endif // MAIN_H
