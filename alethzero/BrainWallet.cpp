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
/** @file BrainWallet.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "BrainWallet.h"
#include <QMenu>
#include <QDialog>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <libdevcore/Log.h>
#include <libdevcrypto/WordList.h>
#include <libethcore/KeyManager.h>
#include <libethereum/Client.h>
#include "ui_BrainWallet.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

DEV_AZ_NOTE_PLUGIN(BrainWallet);

BrainWallet::BrainWallet(MainFace* _m):
	Plugin(_m, "BrainWallet")
{
	connect(addMenuItem("New Brain Wallet...", "menuTools", true), SIGNAL(triggered()), SLOT(create()));
}

BrainWallet::~BrainWallet()
{
}

void BrainWallet::create()
{
	QDialog d;
	Ui_BrainWallet u;
	u.setupUi(&d);
	d.setWindowTitle("New Brain Wallet");
	connect(u.generate, &QPushButton::clicked, [&](){
		boost::random_device d;
		boost::random::uniform_int_distribution<unsigned> pickWord(0, WordList.size() - 1);
		QString t;
		for (int i = 0; i < 13; ++i)
			t += (t.size() ? " " : "") + QString::fromStdString(WordList[pickWord(d)]);
		u.seed->setPlainText(t);
	});

	if (d.exec() == QDialog::Accepted)
	{
		main()->keyManager().importBrain(u.seed->toPlainText().trimmed().toStdString(), u.name->text().toStdString(), u.hint->toPlainText().toStdString());
		main()->noteKeysChanged();
	}
}
