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

#ifdef Q_MOC_RUN
#define BOOST_MPL_IF_HPP_INCLUDED
#endif

#include <map>
#include <QtNetwork/QNetworkAccessManager>
#include <QtCore/QAbstractListModel>
#include <QtCore/QMutex>
#include <QtWidgets/QMainWindow>
#include <libdevcore/RLP.h>
#include <libethcore/CommonEth.h>
#include <libethereum/State.h>
#include <libethereum/Executive.h>
#include <libqwebthree/QWebThree.h>
#include <libwebthree/WebThree.h>
#include <libsolidity/CompilerStack.h>

#include "NatspecHandler.h"

namespace Ui {
class Main;
}

namespace dev { namespace eth {
class Client;
class State;
}}

class QQuickView;
class OurWebThreeStubServer;

struct WorldState
{
	uint64_t steps;
	dev::Address cur;
	dev::u256 curPC;
	dev::eth::Instruction inst;
	dev::bigint newMemSize;
	dev::u256 gas;
	dev::h256 code;
	dev::h256 callData;
	dev::u256s stack;
	dev::bytes memory;
	dev::bigint gasCost;
	std::map<dev::u256, dev::u256> storage;
	std::vector<WorldState const*> levels;
};

using WatchHandler = std::function<void(dev::eth::LocalisedLogEntries const&)>;

QString contentsOfQResource(std::string const& res);

class Main : public QMainWindow
{
	Q_OBJECT

public:
	explicit Main(QWidget *parent = 0);
	~Main();

	dev::WebThreeDirect* web3() const { return m_webThree.get(); }
	dev::eth::Client* ethereum() const { return m_webThree->ethereum(); }
	std::shared_ptr<dev::shh::WhisperHost> whisper() const { return m_webThree->whisper(); }

	std::string lookupNatSpec(dev::h256 const& _contractHash) const;
	std::string lookupNatSpecUserNotice(dev::h256 const& _contractHash, dev::bytes const& _transactionData);

	QList<dev::KeyPair> owned() const { return m_myIdentities + m_myKeys; }
	
	QVariant evalRaw(QString const& _js);

public slots:
	void load(QString _file);
	void note(QString _entry);
	void debug(QString _entry);
	void warn(QString _entry);
	QString contents(QString _file);

	void onKeysChanged();

private slots:
	void eval(QString const& _js);

	void on_connect_triggered();
	void on_mine_triggered();
	void on_send_clicked();
	void on_create_triggered();
	void on_killAccount_triggered();
	void on_net_triggered();
	void on_verbosity_valueChanged();
	void on_ourAccounts_doubleClicked();
	void ourAccountsRowsMoved();
	void on_accounts_doubleClicked();
	void on_destination_currentTextChanged();
	void on_data_textChanged();
	void on_idealPeers_valueChanged();
	void on_value_valueChanged() { updateFee(); }
	void on_gas_valueChanged() { updateFee(); }
	void on_valueUnits_currentIndexChanged() { updateFee(); }
	void on_gasPriceUnits_currentIndexChanged() { updateFee(); }
	void on_gasPrice_valueChanged() { updateFee(); }
	void on_log_doubleClicked();
	void on_blocks_currentItemChanged();
	void on_contracts_doubleClicked();
	void on_contracts_currentItemChanged();
	void on_transactionQueue_currentItemChanged();
	void on_about_triggered();
	void on_paranoia_triggered();
	void on_nameReg_textChanged();
	void on_preview_triggered();
	void on_quit_triggered() { close(); }
	void on_urlEdit_returnPressed();
	void on_debugStep_triggered();
	void on_debugStepBack_triggered();
	void on_debug_clicked();
	void on_debugTimeline_valueChanged();
	void on_jsInput_returnPressed();
	void on_killBlockchain_triggered();
	void on_clearPending_triggered();
	void on_importKey_triggered();
	void on_exportKey_triggered();
	void on_inject_triggered();
	void on_showAll_triggered() { refreshBlockChain(); }
	void on_showAllAccounts_triggered() { refreshAccounts(); }
	void on_loadJS_triggered();
	void on_blockChainFilter_textChanged();
	void on_forceMining_triggered();
	void on_dumpTrace_triggered();
	void on_dumpTraceStorage_triggered();
	void on_dumpTracePretty_triggered();
	void on_debugStepInto_triggered();
	void on_debugStepOut_triggered();
	void on_debugStepBackOut_triggered();
	void on_debugStepBackInto_triggered();
	void on_callStack_currentItemChanged();
	void on_debugCurrent_triggered();
	void on_debugDumpState_triggered(int _add = 1);
	void on_debugDumpStatePre_triggered();
	void on_refresh_triggered();
	void on_usePrivate_triggered();
	void on_enableOptimizer_triggered();
	void on_turboMining_triggered();
	void on_go_triggered();
	void on_importKeyFile_triggered();
	void on_post_clicked();
	void on_newIdentity_triggered();
	void on_jitvm_triggered();

	void refreshWhisper();
	void refreshBlockChain();
	void addNewId(QString _ids);

signals:
	void poll();

private:
	dev::p2p::NetworkPreferences netPrefs() const;

	QString pretty(dev::Address _a) const;
	QString prettyU256(dev::u256 _n) const;

	QString lookup(QString const& _n) const;
	dev::Address getNameReg() const;
	dev::Address getCurrencies() const;

	void populateDebugger(dev::bytesConstRef r);
	void initDebugger();
	void updateDebugger();
	void debugFinished();
	QString render(dev::Address _a) const;
	dev::Address fromString(QString const& _a) const;
	std::string renderDiff(dev::eth::StateDiff const& _d) const;

	void alterDebugStateGroup(bool _enable) const;

	void updateFee();
	void readSettings(bool _skipGeometry = false);
	void writeSettings();

	bool isCreation() const;
	dev::u256 fee() const;
	dev::u256 total() const;
	dev::u256 value() const;
	dev::u256 gasPrice() const;

	unsigned installWatch(dev::eth::LogFilter const& _tf, WatchHandler const& _f);
	unsigned installWatch(dev::h256 _tf, WatchHandler const& _f);
	void uninstallWatch(unsigned _w);

	void keysChanged();

	void onNewPending();
	void onNewBlock();
	void onNameRegChange();
	void onCurrenciesChange();
	void onBalancesChange();

	void installWatches();
	void installCurrenciesWatch();
	void installNameRegWatch();
	void installBalancesWatch();

	virtual void timerEvent(QTimerEvent*);

	void refreshNetwork();
	void refreshMining();
	void refreshWhispers();

	void refreshAll();
	void refreshPending();
	void refreshAccounts();
	void refreshDestination();
	void refreshBlockCount();
	void refreshBalances();

	/// Attempts to infer that @c _source contains Solidity code
	bool sourceIsSolidity(std::string const& _source);
	/// @eturns all method hashes of a Solidity contract in a string
	std::string const getFunctionHashes(dev::solidity::CompilerStack const &_compiler, std::string const& _contractName = "");

	std::unique_ptr<Ui::Main> ui;

	std::unique_ptr<dev::WebThreeDirect> m_webThree;

	std::map<unsigned, WatchHandler> m_handlers;
	unsigned m_nameRegFilter = (unsigned)-1;
	unsigned m_currenciesFilter = (unsigned)-1;
	unsigned m_balancesFilter = (unsigned)-1;

	QByteArray m_networkConfig;
	QStringList m_servers;
	QList<dev::KeyPair> m_myKeys;
	QList<dev::KeyPair> m_myIdentities;
	QString m_privateChain;
	dev::bytes m_data;
	dev::Address m_nameReg;

	unsigned m_backupGas;

	dev::eth::State m_executiveState;
	std::unique_ptr<dev::eth::Executive> m_currentExecution;
	dev::h256 m_lastCode;
	dev::h256 m_lastData;
	std::vector<WorldState const*> m_lastLevels;

	QMap<unsigned, unsigned> m_pcWarp;
	QList<WorldState> m_history;
	std::map<dev::u256, dev::bytes> m_codes;	// and pcWarps
	bool m_enableOptimizer = true;

	QNetworkAccessManager m_webCtrl;

	QList<QPair<QString, QString>> m_consoleHistory;
	QMutex m_logLock;
	QString m_logHistory;
	bool m_logChanged = true;

	std::unique_ptr<QWebThreeConnector> m_qwebConnector;
	std::unique_ptr<OurWebThreeStubServer> m_server;
	QWebThree* m_qweb = nullptr;

	static QString fromRaw(dev::h256 _n, unsigned* _inc = nullptr);
	NatspecHandler m_natspecDB;
};
