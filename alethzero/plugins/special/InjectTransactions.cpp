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
/** @file InjectTransactions.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "InjectTransactions.h"
#include <QMessageBox>
#include <QInputDialog>
#include <libdevcore/Log.h>
#include <libethereum/Client.h>
#include "ui_InjectTransactions.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(InjectTransactions);

InjectTransactions::InjectTransactions(MainFace* _m):
	Plugin(_m, "InjectTransactions")
{
	connect(addMenuItem("Inject Transaction...", "menuSpecial", true), SIGNAL(triggered()), SLOT(injectOne()));
	connect(addMenuItem("Bulk Inject Transactions...", "menuSpecial", false), SIGNAL(triggered()), SLOT(injectBulk()));
}

InjectTransactions::~InjectTransactions()
{
}

void InjectTransactions::injectOne()
{
	bool ok;
	QString s = QInputDialog::getText(main(), "Inject Transaction", "Enter transaction dump in hex", QLineEdit::Normal, QString(), &ok);
	if (ok)
		doInject(s);
}

void InjectTransactions::injectBulk()
{
	QDialog d;
	Ui_InjectTransactions u;
	u.setupUi(&d);
	d.setWindowTitle("Bulk Inject Transactions");
	if (d.exec() == QDialog::Accepted)
		for (QString const& s: u.transactions->toPlainText().split("\n"))
			doInject(s);
}

void InjectTransactions::doInject(QString _txHex)
{
	try
	{
		bytes b = fromHex(_txHex.toStdString(), WhenError::Throw);
		main()->ethereum()->injectTransaction(b);
	}
	catch (BadHexCharacter& _e)
	{
		if (QMessageBox::warning(main(), "Invalid Transaction Hex", "Invalid hex character in:\n" + _txHex + "\nTransaction rejected.", QMessageBox::Ignore, QMessageBox::Abort) == QMessageBox::Abort)
			return;
	}
	catch (Exception& _e)
	{
		if (QMessageBox::warning(main(), "Transaction Rejected", "Invalid transaction; due to" + QString::fromStdString(_e.what()) + "\n" + _txHex + "\nTransaction rejected.", QMessageBox::Ignore, QMessageBox::Abort) == QMessageBox::Abort)
			return;
	}
	catch (...)
	{
		// Should not happen under normal circumstances.
		return;
	}
}
