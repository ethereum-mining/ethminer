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
/** @file OurWebThreeStubServer.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <queue>
#include <QtCore/QObject>
#include <libdevcore/Guards.h>
#include <libethcore/CommonJS.h>
#include <libdevcrypto/Common.h>
#include <libweb3jsonrpc/WebThreeStubServer.h>
#include <libweb3jsonrpc/AccountHolder.h>

class Main;

class OurAccountHolder: public QObject, public dev::eth::AccountHolder
{
	Q_OBJECT

public:
	OurAccountHolder(Main* _main);

public slots:
	void doValidations();

protected:
	// easiest to return keyManager.addresses();
	virtual dev::AddressHash realAccounts() const override;
	// use web3 to submit a signed transaction to accept
	virtual dev::h256 authenticate(dev::eth::TransactionSkeleton const& _t) override;

private:
	bool showAuthenticationPopup(std::string const& _title, std::string const& _text);
	bool showCreationNotice(dev::eth::TransactionSkeleton const& _t, bool _toProxy);
	bool showSendNotice(dev::eth::TransactionSkeleton const& _t, bool _toProxy);
	bool showUnknownCallNotice(dev::eth::TransactionSkeleton const& _t, bool _toProxy);

	bool validateTransaction(dev::eth::TransactionSkeleton const& _t, bool _toProxy);

	std::queue<dev::eth::TransactionSkeleton> m_queued;
	dev::Mutex x_queued;

	Main* m_main;
};

class OurWebThreeStubServer: public QObject, public dev::WebThreeStubServer
{
	Q_OBJECT

public:
	OurWebThreeStubServer(
		jsonrpc::AbstractServerConnector& _conn,
		Main* main
	);

	virtual std::string shh_newIdentity() override;

signals:
	void onNewId(QString _s);

private:
	Main* m_main;
};
