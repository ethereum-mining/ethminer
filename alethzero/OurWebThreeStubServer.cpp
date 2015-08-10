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
using namespace az;
using namespace eth;

OurWebThreeStubServer::OurWebThreeStubServer(
	jsonrpc::AbstractServerConnector& _conn,
	Main* _main
):
	WebThreeStubServer(_conn, *_main->web3(), make_shared<OurAccountHolder>(_main), _main->owned().toVector().toStdVector(), _main->keyManager(), *static_cast<TrivialGasPricer*>(_main->ethereum()->gasPricer().get())),
	m_main(_main)
{
}

string OurWebThreeStubServer::shh_newIdentity()
{
	KeyPair kp = dev::KeyPair::create();
	emit onNewId(QString::fromStdString(toJS(kp.sec())));
	return toJS(kp.pub());
}

OurAccountHolder::OurAccountHolder(Main* _main):
	AccountHolder([=](){ return _main->ethereum(); }),
	m_main(_main)
{
	connect(_main, SIGNAL(poll()), this, SLOT(doValidations()));
}

bool OurAccountHolder::showAuthenticationPopup(string const& _title, string const& _text)
{
	if (!m_main->confirm())
	{
		cnote << "Skipping confirmation step for: " << _title << "\n" << _text;
		return true;
	}

	QMessageBox userInput;
	userInput.setText(QString::fromStdString(_title));
	userInput.setInformativeText(QString::fromStdString(_text));
	userInput.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	userInput.button(QMessageBox::Ok)->setText("Allow");
	userInput.button(QMessageBox::Cancel)->setText("Reject");
	userInput.setDefaultButton(QMessageBox::Cancel);
	return userInput.exec() == QMessageBox::Ok;
	//QMetaObject::invokeMethod(m_main, "authenticate", Qt::BlockingQueuedConnection, Q_RETURN_ARG(int, button), Q_ARG(QString, QString::fromStdString(_title)), Q_ARG(QString, QString::fromStdString(_text)));
	//return button == QMessageBox::Ok;
}

bool OurAccountHolder::showCreationNotice(TransactionSkeleton const& _t, bool _toProxy)
{
	return showAuthenticationPopup("Contract Creation Transaction", string("ÐApp is attemping to create a contract; ") + (_toProxy ? "(this transaction is not executed directly, but forwarded to another ÐApp) " : "") + "to be endowed with " + formatBalance(_t.value) + ", with additional network fees of up to " + formatBalance(_t.gas * _t.gasPrice) + ".\n\nMaximum total cost is " + formatBalance(_t.value + _t.gas * _t.gasPrice) + ".");
}

bool OurAccountHolder::showSendNotice(TransactionSkeleton const& _t, bool _toProxy)
{
	return showAuthenticationPopup("Fund Transfer Transaction", "ÐApp is attempting to send " + formatBalance(_t.value) + " to a recipient " + m_main->pretty(_t.to) + (_toProxy ? " (this transaction is not executed directly, but forwarded to another ÐApp)" : "") +
", with additional network fees of up to " + formatBalance(_t.gas * _t.gasPrice) + ".\n\nMaximum total cost is " + formatBalance(_t.value + _t.gas * _t.gasPrice) + ".");
}

bool OurAccountHolder::showUnknownCallNotice(TransactionSkeleton const& _t, bool _toProxy)
{
	return showAuthenticationPopup("DANGEROUS! Unknown Contract Transaction!",
		"ÐApp is attempting to call into an unknown contract at address " +
		m_main->pretty(_t.to) + ".\n\n" +
		(_toProxy ? "This transaction is not executed directly, but forwarded to another ÐApp.\n\n" : "")  +
		"Call involves sending " +
		formatBalance(_t.value) + " to the recipient, with additional network fees of up to " +
		formatBalance(_t.gas * _t.gasPrice) +
		"However, this also does other stuff which we don't understand, and does so in your name.\n\n" +
		"WARNING: This is probably going to cost you at least " +
		formatBalance(_t.value + _t.gas * _t.gasPrice) +
		", however this doesn't include any side-effects, which could be of far greater importance.\n\n" +
		"REJECT UNLESS YOU REALLY KNOW WHAT YOU ARE DOING!");
}

h256 OurAccountHolder::authenticate(TransactionSkeleton const& _t)
{
	Guard l(x_queued);
	m_queued.push(_t);
	return h256();
}

void OurAccountHolder::doValidations()
{
	Guard l(x_queued);
	while (!m_queued.empty())
	{
		auto t = m_queued.front();
		m_queued.pop();

		bool proxy = isProxyAccount(t.from);
		if (!proxy && !isRealAccount(t.from))
		{
			cwarn << "Trying to send from non-existant account" << t.from;
			return;
		}

		// TODO: determine gas price.

		if (!validateTransaction(t, proxy))
			return;

		if (proxy)
			queueTransaction(t);
		else
			// sign and submit.
			if (Secret s = m_main->retrieveSecret(t.from))
				m_main->ethereum()->submitTransaction(t, s);
	}
}

AddressHash OurAccountHolder::realAccounts() const
{
	return m_main->keyManager().accountsHash();
}

bool OurAccountHolder::validateTransaction(TransactionSkeleton const& _t, bool _toProxy)
{
	if (!m_main->doConfirm())
		return true;

	if (_t.creation)
	{
		// show notice concerning the creation code. TODO: this needs entering into natspec.
		return showCreationNotice(_t, _toProxy);
	}

	h256 contractCodeHash = m_main->ethereum()->postState().codeHash(_t.to);
	if (contractCodeHash == EmptySHA3)
	{
		// recipient has no code - nothing special about this transaction, show basic value transfer info
		return showSendNotice(_t, _toProxy);
	}

	string userNotice = m_main->natSpec()->getUserNotice(contractCodeHash, _t.data);

	if (userNotice.empty())
		return showUnknownCallNotice(_t, _toProxy);

	NatspecExpressionEvaluator evaluator;
	userNotice = evaluator.evalExpression(QString::fromStdString(userNotice)).toStdString();

	// otherwise it's a transaction to a contract for which we have the natspec
	return showAuthenticationPopup("Contract Transaction",
		"ÐApp attempting to conduct contract interaction with " +
		m_main->pretty(_t.to) +
		": <b>" + userNotice + "</b>.\n\n" +
		(_toProxy ? "This transaction is not executed directly, but forwarded to another ÐApp.\n\n" : "") +
		(_t.value > 0 ?
			"In addition, ÐApp is attempting to send " +
			formatBalance(_t.value) + " to said recipient, with additional network fees of up to " +
			formatBalance(_t.gas * _t.gasPrice) + " = " +
			formatBalance(_t.value + _t.gas * _t.gasPrice) + "."
		:
			"Additional network fees are at most" +
			formatBalance(_t.gas * _t.gasPrice) + ".")
		);
}
