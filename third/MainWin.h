/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file MainWin.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once


#include <map>
#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libethential/RLP.h>
#include <libethcore/CommonEth.h>
#include <libethereum/State.h>
#include <libqethereum/QEthereum.h>

namespace Ui {
class Main;
}

namespace eth {
class Client;
class State;
class MessageFilter;
}

class QQuickView;

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

	eth::Client* client() { return m_client.get(); }

	QList<eth::KeyPair> const& owned() const { return m_myKeys; }
	
public slots:
	void note(QString _entry);
	void debug(QString _entry);
	void warn(QString _entry);
	void eval(QString const& _js);

	void onKeysChanged();

private slots:
	void on_mine_triggered();
	void on_ourAccounts_doubleClicked();
	void ourAccountsRowsMoved();
	void on_about_triggered();
	void on_connect_triggered();
	void on_quit_triggered() { close(); }
	void on_urlEdit_returnPressed();
	void on_importKey_triggered();
	void on_exportKey_triggered();

signals:
	void poll();

private:
	QString pretty(eth::Address _a) const;
	QString render(eth::Address _a) const;
	eth::Address fromString(QString const& _a) const;
	QString lookup(QString const& _n) const;

	void readSettings(bool _skipGeometry = false);
	void writeSettings();

	unsigned installWatch(eth::MessageFilter const& _tf, std::function<void()> const& _f);
	unsigned installWatch(eth::h256 _tf, std::function<void()> const& _f);

	void onNewBlock();
	void onNameRegChange();
	void onCurrenciesChange();
	void onBalancesChange();

	void installWatches();
	void installCurrenciesWatch();
	void installNameRegWatch();
	void installBalancesWatch();

	virtual void timerEvent(QTimerEvent*);
	void ensureNetwork();

	void refreshAll();
	void refreshBlockCount();
	void refreshBalances();
	void refreshNetwork();
	void refreshMining();

	std::unique_ptr<Ui::Main> ui;

	std::unique_ptr<eth::Client> m_client;

	QList<eth::KeyPair> m_myKeys;

	std::map<unsigned, std::function<void()>> m_handlers;
	unsigned m_nameRegFilter = (unsigned)-1;
	unsigned m_currenciesFilter = (unsigned)-1;
	unsigned m_balancesFilter = (unsigned)-1;

	QByteArray m_peers;
	QStringList m_servers;

	QNetworkAccessManager m_webCtrl;

	QEthereum* m_ethereum = nullptr;
};
