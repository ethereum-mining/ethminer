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
/** @file AllAccounts.h
 * @author Gav Wood <i@gavwood.com>
 * @date 2015
 */

#include "AllAccounts.h"
#include <sstream>
#include <libdevcore/Log.h>
#include <libdevcore/SHA3.h>
#include <libevmcore/Instruction.h>
#include <libethereum/Client.h>
#include "ui_AllAccounts.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

AllAccounts::AllAccounts(MainFace* _m):
	Plugin(_m, "AllAccounts"),
	m_ui(new Ui::AllAccounts)
{
	m_ui->setupUi(dock());
	installWatches();
	refresh();
}

AllAccounts::~AllAccounts()
{
}

void AllAccounts::installWatches()
{
	installWatch(ChainChangedFilter, [=](LocalisedLogEntries const&){ onAllChange(); });
	installWatch(PendingChangedFilter, [=](LocalisedLogEntries const&){ onAllChange(); });
}

void AllAccounts::refresh()
{
	DEV_TIMED_FUNCTION;
#if ETH_FATDB || !ETH_TRUE
	cwatch << "refreshAccounts()";
	m_ui->accounts->clear();
	bool showContract = m_ui->showContracts->isChecked();
	bool showBasic = m_ui->showBasic->isChecked();
	bool onlyNamed = m_ui->onlyNamed->isChecked();
	for (auto const& i: ethereum()->addresses())
	{
		bool isContract = (ethereum()->codeHashAt(i) != EmptySHA3);
		if (!((showContract && isContract) || (showBasic && !isContract)))
			continue;
		string r = render(i);
		if (onlyNamed && !(r.find('"') != string::npos || r.substr(0, 2) == "XE"))
			continue;
		(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(ethereum()->balanceAt(i)).c_str()).arg(QString::fromStdString(r)).arg((unsigned)ethereum()->countAt(i)), ui->accounts))
			->setData(Qt::UserRole, QByteArray((char const*)i.data(), Address::size));
	}
#endif
	m_refreshAccounts->setEnabled(false);
}

void AllAccounts::onAllChange()
{
	ui->refreshAccounts->setEnabled(true);
}

void AllAccounts::on_accounts_currentItemChanged()
{
	m_ui->accountInfo->clear();
	if (auto item = m_ui->accounts->currentItem())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		assert(hba.size() == 20);
		auto address = h160((byte const*)hba.data(), h160::ConstructFromPointer);

		stringstream s;
		try
		{
			auto storage = ethereum()->storageAt(address);
			for (auto const& i: storage)
				s << "@" << showbase << hex << main()->prettyU256(i.first) << "&nbsp;&nbsp;&nbsp;&nbsp;" << showbase << hex << main()->prettyU256(i.second) << "<br/>";
			s << "<h4>Body Code (" << sha3(ethereum()->codeAt(address)).abridged() << ")</h4>" << disassemble(ethereum()->codeAt(address));
			s << ETH_HTML_DIV(ETH_HTML_MONO) << toHex(ethereum()->codeAt(address)) << "</div>";
			m_ui->accountInfo->appendHtml(QString::fromStdString(s.str()));
		}
		catch (dev::InvalidTrie)
		{
			m_ui->accountInfo->appendHtml("Corrupted trie.");
		}
		ui->accountInfo->moveCursor(QTextCursor::Start);
	}
}

void AllAccounts::on_accounts_doubleClicked()
{
	if (ui->accounts->count())
	{
		auto hba = ui->accounts->currentItem()->data(Qt::UserRole).toByteArray();
		auto h = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		qApp->clipboard()->setText(QString::fromStdString(toHex(h.asArray())));
	}
}

