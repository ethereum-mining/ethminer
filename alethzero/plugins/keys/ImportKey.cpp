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
/** @file ImportKey.cpp
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "ImportKey.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <libdevcore/Log.h>
#include <libethcore/KeyManager.h>
#include <libethcore/ICAP.h>
#include <libethereum/Client.h>
#include "ui_ImportKey.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(ImportKey);

ImportKey::ImportKey(MainFace* _m):
	Plugin(_m, "ImportKey")
{
	connect(addMenuItem("Import Key...", "menuTools", true), SIGNAL(triggered()), SLOT(import()));
}

ImportKey::~ImportKey()
{
}

void ImportKey::import()
{
	QDialog d;
	Ui_ImportKey u;
	u.setupUi(&d);
	d.setWindowTitle("Import Key");

	string lastKey;
	Secret lastSecret;
	string lastPassword;
	Address lastAddress;

	auto updateAction = [&](){
		if (!u.import_2->isEnabled())
			u.action->clear();
		else if (lastKey.empty() && !lastSecret)
			u.action->setText("Import brainwallet with given address and hint");
		else if (!lastKey.empty() && !lastSecret)
		{
			h256 ph;
			DEV_IGNORE_EXCEPTIONS(ph = h256(u.passwordHash->text().toStdString()));
			if (ph)
				u.action->setText("Import untouched key with given address and hint");
			else
				u.action->setText("Import untouched key with given address, password hash and hint");
		}
		else
		{
			bool mp = u.noPassword->isChecked();
			if (mp)
				u.action->setText("Import recast key using master password and given hint");
			else
				u.action->setText("Import recast key with given password and hint");
		}
	};

	auto updateImport = [&](){
		u.import_2->setDisabled(u.addressOut->text().isEmpty() || u.name->text().isEmpty() || !(u.oldPassword->isChecked() || u.newPassword->isChecked() || u.noPassword->isChecked()));
		updateAction();
	};

	auto updateAddress = [&](){
		lastAddress.clear();
		string as = u.address->text().toStdString();
		try
		{
			lastAddress = eth::toAddress(as);
			u.addressOut->setText(QString::fromStdString(main()->render(lastAddress)));
		}
		catch (...)
		{
			u.addressOut->setText("");
		}
		updateImport();
	};

	auto updatePassword = [&](){
		u.passwordHash->setText(QString::fromStdString(sha3(u.password->text().toStdString()).hex()));
		updateAction();
	};

	function<void()> updateKey = [&](){
		// update according to key.
		if (lastKey == u.key->text().toStdString())
			return;
		lastKey = u.key->text().toStdString();
		lastSecret.clear();
		u.address->clear();
		u.oldPassword->setEnabled(false);
		u.oldPassword->setChecked(false);
		bytes b;
		DEV_IGNORE_EXCEPTIONS(b = fromHex(lastKey, WhenError::Throw));
		if (b.size() == 32)
		{
			lastSecret = Secret(b);
			bytesRef(&b).cleanse();
		}
		while (!lastKey.empty() && !lastSecret)
		{
			bool ok;
			lastPassword = QInputDialog::getText(&d, "Open Key File", "Enter the password protecting this key file. Cancel if you do not want to provide te password.", QLineEdit::Password, QString(), &ok).toStdString();
			if (!ok)
			{
				lastSecret.clear();
				break;
			}
			// Try to open as a file.
			lastSecret = KeyManager::presaleSecret(contentsString(lastKey), [&](bool first){ return first ? lastPassword : string(); }).secret();
			if (!lastSecret)
				lastSecret = Secret(SecretStore::secret(contentsString(lastKey), lastPassword));
			if (!lastSecret && QMessageBox::warning(&d, "Invalid Password or Key File", "The given password could not be used to decrypt the key file given. Are you sure it is a valid key file and that the password is correct?", QMessageBox::Abort, QMessageBox::Retry) == QMessageBox::Abort)
			{
				u.key->clear();
				updateKey();
				return;
			}
		}
		u.oldPassword->setEnabled(!!lastSecret);
		u.newPassword->setEnabled(!!lastSecret);
		u.noPassword->setEnabled(!!lastSecret);
		u.masterLabel->setEnabled(!!lastSecret);
		u.oldLabel->setEnabled(!!lastSecret);
		u.showPassword->setEnabled(!!lastSecret);
		u.password->setEnabled(!!lastSecret);
		u.passwordHash->setReadOnly(!!lastSecret);
		u.address->setReadOnly(!!lastSecret);
		if (lastSecret)
		{
			u.oldPassword->setEnabled(!lastPassword.empty());
			if (lastPassword.empty())
				u.oldPassword->setChecked(false);
			u.address->setText(QString::fromStdString(ICAP(toAddress(lastSecret)).encoded()));
			updateAddress();
		}
		else
			u.address->clear();
		updateImport();
	};

	connect(u.noPassword, &QRadioButton::clicked, [&](){
		u.passwordHash->clear();
		u.hint->setText("No additional password (same as master password).");
		updateAction();
	});
	connect(u.oldPassword, &QRadioButton::clicked, [&](){
		u.passwordHash->setText(QString::fromStdString(sha3(lastPassword).hex()));
		u.hint->setText("Same as original password for file " + QString::fromStdString(lastKey));
		updateAction();
	});
	connect(u.newPassword, &QRadioButton::clicked, [&](){
		u.hint->setText("");
		updatePassword();
	});
	connect(u.password, &QLineEdit::textChanged, [&](){ updatePassword(); });
	connect(u.address, &QLineEdit::textChanged, [&](){ updateAddress(); });
	connect(u.key, &QLineEdit::textEdited, [&](){ updateKey(); });
	connect(u.name, &QLineEdit::textEdited, [&](){ updateImport(); });
	connect(u.showPassword, &QCheckBox::toggled, [&](bool show){ u.password->setEchoMode(show ? QLineEdit::Normal : QLineEdit::Password); });
	connect(u.openKey, &QToolButton::clicked, [&](){
		QString fn = QFileDialog::getOpenFileName(main(), "Open Key File", QDir::homePath(), "JSON Files (*.json);;All Files (*)");
		if (!fn.isEmpty())
		{
			u.key->setText(fn);
			updateKey();
		}
	});

	if (d.exec() == QDialog::Accepted)
	{
		Address a = ICAP::decoded(lastAddress).direct();
		string n = u.name->text().toStdString();
		string h = u.hint->text().toStdString();

		// check for a brain wallet import
		if (lastKey.empty() && !lastSecret)
			main()->keyManager().importExistingBrain(a, n, h);
		else if (!lastKey.empty() && !lastSecret)
		{
			h256 ph;
			DEV_IGNORE_EXCEPTIONS(ph = h256(u.passwordHash->text().toStdString()));
			main()->keyManager().importExisting(main()->keyManager().store().importKey(lastKey), n, a, ph, h);
		}
		else
		{
			bool mp = u.noPassword->isChecked();
			string p = mp ? string() : u.oldPassword ? lastPassword : u.password->text().toStdString();
			if (mp)
				main()->keyManager().import(lastSecret, n);
			else
				main()->keyManager().import(lastSecret, n, p, h);
		}

		main()->noteKeysChanged();
	}
}
