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
/** @file OurWebThreeStubServer.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include "OurWebThreeStubServer.h"

#include <QMessageBox>
#include <QAbstractButton>
#include <libwebthree/WebThree.h>
#include <libnatspec/NatspecExpressionEvaluator.h>

#include "MainWin.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

OurWebThreeStubServer::OurWebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, WebThreeDirect& _web3,
											 vector<KeyPair> const& _accounts, Main* main):
	WebThreeStubServer(_conn, _web3, _accounts), m_web3(&_web3), m_main(main)
{}

string OurWebThreeStubServer::shh_newIdentity()
{
	KeyPair kp = dev::KeyPair::create();
	emit onNewId(QString::fromStdString(toJS(kp.sec())));
	return toJS(kp.pub());
}

bool OurWebThreeStubServer::showAuthenticationPopup(string const& _title, string const& _text) const
{
	QMessageBox userInput;
	userInput.setText(QString::fromStdString(_title));
	userInput.setInformativeText(QString::fromStdString(_text + "\n Do you wish to allow this?"));
	userInput.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	userInput.button(QMessageBox::Ok)->setText("Allow");
	userInput.button(QMessageBox::Cancel)->setText("Reject");
	userInput.setDefaultButton(QMessageBox::Cancel);
	return userInput.exec() == QMessageBox::Ok;
}

void OurWebThreeStubServer::showBasicValueTransferNotice(u256 _value) const
{
	QMessageBox notice;
	notice.setText("Basic Value Transfer Transaction");
	notice.setInformativeText(QString::fromStdString("Value is " + toString(_value)));
	notice.setStandardButtons(QMessageBox::Ok);
	notice.exec();
}

bool OurWebThreeStubServer::authenticate(TransactionSkeleton const& _t)
{
	h256 contractCodeHash = m_web3->ethereum()->postState().codeHash(_t.to);
	if (contractCodeHash == EmptySHA3)
		// contract creation
		return true;

	if (false) //TODO: When is is just a value transfer?
	{
		// recipient has no code - nothing special about this transaction, show basic value transfer info
		showBasicValueTransferNotice(_t.value);
		return true;
	}

	string userNotice = m_main->lookupNatSpecUserNotice(contractCodeHash, _t.data);

	if (userNotice.empty())
		return showAuthenticationPopup("Unverified Pending Transaction",
									   "An undocumented transaction is about to be executed.");

	NatspecExpressionEvaluator evaluator;
	userNotice = evaluator.evalExpression(QString::fromStdString(userNotice)).toStdString();

	// otherwise it's a transaction to a contract for which we have the natspec
	return showAuthenticationPopup("Pending Transaction", userNotice);
}
