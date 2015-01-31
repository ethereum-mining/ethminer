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
#include <libdevcore/RLP.h>
#include <libethcore/CommonEth.h>
#include <libethereum/State.h>
#include <libqwebthree/QWebThree.h>

namespace Ui {
class Main;
}

namespace dev { class WebThreeDirect;
namespace eth {
class Client;
class State;
class LogFilter;
}
namespace shh {
class WhisperHost;
}
}

class QQuickView;
class WebThreeStubServer;

using WatchHandler = std::function<void(dev::eth::LocalisedLogEntries const&)>;

class Main : public QMainWindow
{
	Q_OBJECT
	
public:
	explicit Main(QWidget *parent = 0);
	~Main();

	dev::WebThreeDirect* web3() const { return m_web3.get(); }
	dev::eth::Client* ethereum() const;
	std::shared_ptr<dev::shh::WhisperHost> whisper() const;

	QList<dev::KeyPair> owned() const { return m_myKeys + m_myIdentities;}
	
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
	QString pretty(dev::Address _a) const;
	QString render(dev::Address _a) const;
	dev::Address fromString(QString const& _a) const;
	QString lookup(QString const& _n) const;

	void readSettings(bool _skipGeometry = false);
	void writeSettings();

	unsigned installWatch(dev::eth::LogFilter const& _tf, WatchHandler const& _f);
	unsigned installWatch(dev::h256 _tf, WatchHandler const& _f);

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

	std::unique_ptr<dev::WebThreeDirect> m_web3;

	QList<dev::KeyPair> m_myKeys;
	QList<dev::KeyPair> m_myIdentities;

	std::map<unsigned, WatchHandler> m_handlers;
	unsigned m_nameRegFilter = (unsigned)-1;
	unsigned m_currenciesFilter = (unsigned)-1;
	unsigned m_balancesFilter = (unsigned)-1;

	QByteArray m_networkConfig;
	QStringList m_servers;

	QNetworkAccessManager m_webCtrl;

	std::unique_ptr<WebThreeStubServer> m_server;
	QWebThreeConnector m_qwebConnector;
	QWebThree* m_qweb = nullptr;
};
