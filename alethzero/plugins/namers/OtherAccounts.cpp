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
/** @file OtherAccounts.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "OtherAccounts.h"
#include <QSettings>
#include <QMessageBox>
#include <libdevcore/Log.h>
#include <libethereum/Client.h>
#include <ui_OtherAccounts.h>
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(OtherAccounts);

OtherAccounts::OtherAccounts(MainFace* _m):
	AccountNamerPlugin(_m, "OtherAccounts")
{
	connect(addMenuItem("Register Third-party Address Names...", "menuTools", true), SIGNAL(triggered()), SLOT(import()));
}

void OtherAccounts::import()
{
	QDialog d;
	Ui_OtherAccounts u;
	u.setupUi(&d);
	d.setWindowTitle("Add Named Accounts");
	if (d.exec() == QDialog::Accepted)
	{
		QStringList sl = u.accounts->toPlainText().split("\n");
		unsigned line = 1;
		for (QString const& s: sl)
		{
			try
			{
				Address addr = dev::eth::toAddress(s.section(QRegExp("[ \\0\\t]+"), 0, 0).trimmed().toStdString());
				string name = s.section(QRegExp("[ \\0\\t]+"), 1).trimmed().toStdString();
				m_toName[addr] = name;
				m_toAddress[name] = addr;
			}
			catch (...)
			{
				if (QMessageBox::warning(&d, "Invalid Line Format", "Line format or address given on line " + QString::number(line) + " is invalid:\n" + s, QMessageBox::Abort, QMessageBox::Ignore) == QMessageBox::Abort)
					break;
			}
			line++;
		}
		main()->noteSettingsChanged();
		noteKnownChanged();
	}
}

void OtherAccounts::readSettings(QSettings const& _s)
{
	m_toName.clear();
	m_toAddress.clear();
	for (QVariant const& i: _s.value("OtherAccounts", QVariantList()).toList())
	{
		QStringList l = i.toStringList();
		if (l.size() == 2)
		{
			m_toName[Address(l[0].toStdString())] = l[1].toStdString();
			m_toAddress[l[1].toStdString()] = Address(l[0].toStdString());
		}
	}
	noteKnownChanged();
}

void OtherAccounts::writeSettings(QSettings& _s)
{
	QVariantList r;
	for (auto const& i: m_toName)
	{
		QStringList l;
		l += QString::fromStdString(i.first.hex());
		l += QString::fromStdString(i.second);
		r += QVariant(l);
	}
	_s.setValue("OtherAccounts", r);
}
