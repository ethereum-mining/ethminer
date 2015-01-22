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

#include "MainWin.h"

using namespace std;
using namespace dev;
using namespace dev::eth;

OurWebThreeStubServer::OurWebThreeStubServer(jsonrpc::AbstractServerConnector& _conn, dev::WebThreeDirect& _web3,
											 std::vector<dev::KeyPair> const& _accounts, Main* main):
	WebThreeStubServer(_conn, _web3, _accounts), m_web3(&_web3), m_main(main)
{}

std::string OurWebThreeStubServer::shh_newIdentity()
{
	dev::KeyPair kp = dev::KeyPair::create();
	emit onNewId(QString::fromStdString(toJS(kp.sec())));
	return toJS(kp.pub());
}

bool OurWebThreeStubServer::showAuthenticationPopup(std::string const& _title, std::string const& _text) const
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

bool OurWebThreeStubServer::authenticate(dev::TransactionSkeleton const& _t)
{
	h256 contractCodeHash = m_web3->ethereum()->postState().codeHash(_t.to);
	if (contractCodeHash == EmptySHA3)
		// recipient has no code - nothing special about this transaction.
		// TODO: show basic message for value transfer.
		return true;

	std::string userNotice = m_main->lookupNatSpecUserNotice(contractCodeHash, _t.data);
	
	if (userNotice.empty())
		return showAuthenticationPopup("Unverified Pending Transaction",
									   "An undocumented transaction is about to be executed.");

	QNatspecExpressionEvaluator evaluator(this, m_main);
	userNotice = evaluator.evalExpression(QString::fromStdString(userNotice)).toStdString();

	// otherwise it's a transaction to a contract for which we have the natspec
	return showAuthenticationPopup("Pending Transaction", userNotice);
}

QNatspecExpressionEvaluator::QNatspecExpressionEvaluator(OurWebThreeStubServer* _server, Main* _main)
: m_server(_server), m_main(_main)
{}

QNatspecExpressionEvaluator::~QNatspecExpressionEvaluator()
{}

QString QNatspecExpressionEvaluator::evalExpression(QString const& _expression)
{
	
	// evaluate the natspec
	m_main->evalRaw(contentsOfQResource(":/js/natspec.js"));
	
	// _expression should be in the format like this
	// auto toEval = QString::fromStdString("the result of calling multiply(4) is `multiply(4)`");
	auto toEval = _expression;
	auto result = m_main->evalRaw("evaluateExpression('" + toEval + "')");
	
	return result.toString();
}






