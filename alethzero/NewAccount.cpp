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
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "NewAccount.h"
#include <QMenu>
#include <QDialog>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <libdevcore/Log.h>
#include <libdevcrypto/WordList.h>
#include <libethcore/KeyManager.h>
#include <libethereum/Client.h>
#include "ui_NewAccount.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

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
	Ui_NewAccount u;
	u.setupUi(&d);
	d.setWindowTitle("New Account Wallet");
	u.enterHexText->setEnabled(false);
	u.enterPasswordText->setEnabled(false);
	u.enterPasswordAgainText->setEnabled(false);
	enum { NoVanity = 0, DirectICAP, FirstTwo, FirstTwoNextTwo, FirstThree, FirstFour, StringMatch };

	QStringList items = {"No vanity (instant)", "Direct ICAP address", "Two pairs first (a few seconds)", "Two pairs first and second (a few minutes)", "Three pairs first (a few minutes)", "Four pairs first (several hours)", "Specific hex string"};
	u.selectTypeComboBox->addItems(items);

	void (QComboBox:: *indexChangedSignal)(int) = &QComboBox::currentIndexChanged;
	connect(u.selectTypeComboBox, indexChangedSignal, [&](int index) {
		u.enterHexText->setEnabled(index == StringMatch);
	});

	connect(u.additionalCheckBox, &QCheckBox::clicked, [&]() {
		bool checked = u.additionalCheckBox->checkState() == Qt::CheckState::Checked;
		u.enterPasswordText->setEnabled(checked);
		u.enterPasswordAgainText->setEnabled(checked);
	});

	if (d.exec() == QDialog::Accepted)
	{
		//main()->noteKeysChanged();
	}
}
