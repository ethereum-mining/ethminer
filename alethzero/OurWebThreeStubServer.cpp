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
	int button;
	QMetaObject::invokeMethod(m_main, "authenticate", Qt::BlockingQueuedConnection, Q_RETURN_ARG(int, button), Q_ARG(QString, QString::fromStdString(_title)), Q_ARG(QString, QString::fromStdString(_text)));
	return button == QMessageBox::Ok;
}

bool OurWebThreeStubServer::showCreationNotice(TransactionSkeleton const& _t) const
{
	return showAuthenticationPopup("Contract Creation Transaction", "ÐApp is attemping to create a contract; to be endowed with " + formatBalance(_t.value) + ", with additional network fees of up to " + formatBalance(_t.gas * _t.gasPrice) + ".\n\nMaximum total cost is <b>" + formatBalance(_t.value + _t.gas * _t.gasPrice) + "</b>.");
}

bool OurWebThreeStubServer::showSendNotice(TransactionSkeleton const& _t) const
{
	return showAuthenticationPopup("Fund Transfer Transaction", "ÐApp is attempting to send " + formatBalance(_t.value) + " to a recipient " + m_main->pretty(_t.to).toStdString() + ", with additional network fees of up to " + formatBalance(_t.gas * _t.gasPrice) + ".\n\nMaximum total cost is <b>" + formatBalance(_t.value + _t.gas * _t.gasPrice) + "</b>.");
}

bool OurWebThreeStubServer::showUnknownCallNotice(TransactionSkeleton const& _t) const
{
	return showAuthenticationPopup("DANGEROUS! Unknown Contract Transaction!",
		"ÐApp is attempting to call into an unknown contract at address " +
		m_main->pretty(_t.to).toStdString() +
		".\n\nCall involves sending " +
		formatBalance(_t.value) + " to the recipient, with additional network fees of up to " +
		formatBalance(_t.gas * _t.gasPrice) +
		"However, this also does other stuff which we don't understand, and does so in your name.\n\n" +
		"WARNING: This is probably going to cost you at least " +
		formatBalance(_t.value + _t.gas * _t.gasPrice) +
		", however this doesn't include any side-effects, which could be of far greater importance.\n\n" +
		"REJECT UNLESS YOU REALLY KNOW WHAT YOU ARE DOING!");
}

bool OurWebThreeStubServer::authenticate(TransactionSkeleton const& _t)
{
	if (_t.creation)
	{
		// recipient has no code - nothing special about this transaction, show basic value transfer info
		return showCreationNotice(_t);
	}

	h256 contractCodeHash = m_web3->ethereum()->postState().codeHash(_t.to);
	if (contractCodeHash == EmptySHA3)
	{
		// recipient has no code - nothing special about this transaction, show basic value transfer info
		return showSendNotice(_t);
	}

	// TODO: include total cost in Ether
	string userNotice = m_main->natSpec()->getUserNotice(contractCodeHash, _t.data);

	if (userNotice.empty())
		return showUnknownCallNotice(_t);

	NatspecExpressionEvaluator evaluator;
	userNotice = evaluator.evalExpression(QString::fromStdString(userNotice)).toStdString();

	// otherwise it's a transaction to a contract for which we have the natspec
	return showAuthenticationPopup("Contract Transaction",
		"ÐApp attempting to conduct contract interaction with " +
		m_main->pretty(_t.to).toStdString() +
		": <b>" + userNotice + "</b>.\n\n" +
		(_t.value > 0 ?
			"In addition, ÐApp is attempting to send " +
			formatBalance(_t.value) + " to said recipient, with additional network fees of up to " +
			formatBalance(_t.gas * _t.gasPrice) + " = <b>" +
			formatBalance(_t.value + _t.gas * _t.gasPrice) + "</b>."
		:
			"Additional network fees are at most" +
			formatBalance(_t.gas * _t.gasPrice) + ".")
		);
}
