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
/** @file NewAccount.h
 * @author Marek Kotewicz <marek@ethdev.com>
 * @date 2015
 */

#include "NewAccount.h"
#include <QMenu>
#include <QDialog>
#include <libdevcore/Log.h>
#include <libethcore/KeyManager.h>
#include <libethereum/Client.h>
#include "ui_NewAccount.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

bool beginsWith(Address _a, bytes const& _b)
{
	for (unsigned i = 0; i < min<unsigned>(20, _b.size()); ++i)
		if (_a[i] != _b[i])
			return false;
	return true;
}

DEV_AZ_NOTE_PLUGIN(NewAccount);

NewAccount::NewAccount(MainFace* _m):
	Plugin(_m, "NewAccount")
{
	connect(addMenuItem("New Account...", "menuTools", true), SIGNAL(triggered()), SLOT(create()));
}

NewAccount::~NewAccount()
{
}

void NewAccount::create()
{
	QDialog d;
	Ui::NewAccount u;
	u.setupUi(&d);
	d.setWindowTitle("New Account Wallet");
	u.hexText->setEnabled(false);
	u.passwordText->setEnabled(false);
	u.passwordAgainText->setEnabled(false);

	QStringList items =
	{
		"No vanity (instant)",
		"Direct ICAP address",
		"Two pairs first (a few seconds)",
		"Two pairs first and second (a few minutes)",
		"Three pairs first (a few minutes)",
		"Four pairs first (several hours)",
		"Specific hex string"
	};
	u.typeComboBox->addItems(items);

	void (QComboBox:: *indexChangedSignal)(int) = &QComboBox::currentIndexChanged;
	connect(u.typeComboBox, indexChangedSignal, [&](int index)
	{
		u.hexText->setEnabled(index == StringMatch);
	});

	connect(u.additionalCheckBox, &QCheckBox::clicked, [&]()
	{
		bool checked = u.additionalCheckBox->checkState() == Qt::CheckState::Checked;
		u.passwordText->setEnabled(checked);
		u.passwordAgainText->setEnabled(checked);
	});

	connect(u.create, &QPushButton::clicked, [&]()
	{
		if (u.additionalCheckBox->checkState() == Qt::CheckState::Checked && !validatePassword(u))
		{
			u.passwordAgainLabel->setStyleSheet("QLabel { color : red }");
			u.passwordAgainLabel->setText("Invalid! Please re-enter password correctly:");
			return;
		}

		d.accept();
	});

	if (d.exec() == QDialog::Accepted)
		onDialogAccepted(u);

}

bool NewAccount::validatePassword(Ui::NewAccount const& _u)
{
	return QString::compare(_u.passwordText->toPlainText(), _u.passwordAgainText->toPlainText()) == 0;
}

void NewAccount::onDialogAccepted(Ui::NewAccount const& _u)
{
	Type v = (Type)_u.typeComboBox->currentIndex();
	bytes bs = fromHex(_u.hexText->toPlainText().toStdString());
	KeyPair p = newKeyPair(v, bs);
	QString s = _u.nameText->toPlainText();
	if (_u.additionalCheckBox->checkState() == Qt::CheckState::Checked)
	{
		std::string hint = _u.hintText->toPlainText().toStdString();
		std::string password = _u.passwordText->toPlainText().toStdString();
		main()->keyManager().import(p.secret(), s.toStdString(), password, hint);
	}
	else
		main()->keyManager().import(p.secret(), s.toStdString());

	main()->noteKeysChanged();
}

KeyPair NewAccount::newKeyPair(Type _type, bytes const& _prefix)
{
	KeyPair p;
	bool keepGoing = true;
	unsigned done = 0;
	function<void()> f = [&]() {
		KeyPair lp;
		while (keepGoing)
		{
			done++;
			if (done % 1000 == 0)
				cnote << "Tried" << done << "keys";
			lp = KeyPair::create();
			auto a = lp.address();
			if (_type == NoVanity ||
					(_type == DirectICAP && !a[0]) ||
					(_type == FirstTwo && a[0] == a[1]) ||
					(_type == FirstTwoNextTwo && a[0] == a[1] && a[2] == a[3]) ||
					(_type == FirstThree && a[0] == a[1] && a[1] == a[2]) ||
					(_type == FirstFour && a[0] == a[1] && a[1] == a[2] && a[2] == a[3]) ||
					(_type == StringMatch && beginsWith(lp.address(), _prefix))
					)
				break;
		}
		if (keepGoing)
			p = lp;
		keepGoing = false;
	};

	vector<std::thread*> ts;
	for (unsigned t = 0; t < std::thread::hardware_concurrency() - 1; ++t)
		ts.push_back(new std::thread(f));
	f();

	for (std::thread* t: ts)
	{
		t->join();
		delete t;
	}
	return p;
}
