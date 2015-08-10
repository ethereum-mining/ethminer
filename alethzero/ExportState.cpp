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
/** @file ExportState.cpp
 * @author Arkadiy Paronyan <arkadiy@ethdev.com>
 * @date 2015
 */

#include "ExportState.h"
#include <QFileDialog>
#include <QTextStream>
#include <libethereum/Client.h>
#include "MainWin.h"
#include "ui_ExportState.h"
using namespace std;
using namespace dev;
using namespace az;
using namespace eth;

ExportStateDialog::ExportStateDialog(Main* _parent):
	QDialog(_parent),
	ui(new Ui::ExportState),
	m_main(_parent)
{
	ui->setupUi(this);
	connect(ui->close, &QPushButton::clicked, this, &ExportStateDialog::close);
	connect(ui->accounts, &QListWidget::itemSelectionChanged, this, &ExportStateDialog::generateJSON);
	connect(ui->contracts, &QListWidget::itemSelectionChanged, this, &ExportStateDialog::generateJSON);
	fillBlocks();
}

ExportStateDialog::~ExportStateDialog()
{
}

Client* ExportStateDialog::ethereum() const
{
	return m_main->ethereum();
}

void ExportStateDialog::on_block_editTextChanged()
{
	QString text = ui->block->currentText();
	int i = ui->block->count();
	while (i-- >= 0)
		if (ui->block->itemText(i) == text)
			return;
	fillBlocks();
}

void ExportStateDialog::on_block_currentIndexChanged(int _index)
{
	m_block = ui->block->itemData(_index).toUInt();
	fillContracts();
}

void ExportStateDialog::fillBlocks()
{
	BlockChain const& bc = ethereum()->blockChain();
	QStringList filters = ui->block->currentText().toLower().split(QRegExp("\\s+"), QString::SkipEmptyParts);
	const unsigned numLastBlocks = 10;
	if (ui->block->count() == 0)
	{
		unsigned i = numLastBlocks;
		for (auto h = bc.currentHash(); bc.details(h) && i; h = bc.details(h).parent, --i)
		{
			auto d = bc.details(h);
			ui->block->addItem(QString("#%1 %2").arg(d.number).arg(h.abridged().c_str()), d.number);
			if (h == bc.genesisHash())
				break;
		}
		if (ui->block->currentIndex() < 0)
			ui->block->setCurrentIndex(0);
		m_recentBlocks = numLastBlocks - i;
	}

	int i = ui->block->count();
	while (i > 0 && i >= m_recentBlocks)
		ui->block->removeItem(i--);

	h256Hash blocks;
	for (QString f: filters)
	{
		if (f.startsWith("#"))
			f = f.remove(0, 1);
		if (f.size() == 64)
		{
			h256 h(f.toStdString());
			if (bc.isKnown(h))
				blocks.insert(h);
			for (auto const& b: bc.withBlockBloom(LogBloom().shiftBloom<3>(sha3(h)), 0, -1))
				blocks.insert(bc.numberHash(b));
		}
		else if (f.toLongLong() <= bc.number())
			blocks.insert(bc.numberHash((unsigned)f.toLongLong()));
		else if (f.size() == 40)
		{
			Address h(f.toStdString());
			for (auto const& b: bc.withBlockBloom(LogBloom().shiftBloom<3>(sha3(h)), 0, -1))
				blocks.insert(bc.numberHash(b));
		}
	}

	for (auto const& h: blocks)
	{
		auto d = bc.details(h);
		ui->block->addItem(QString("#%1 %2").arg(d.number).arg(h.abridged().c_str()), d.number);
	}
}

void ExportStateDialog::fillContracts()
{
	ui->accounts->clear();
	ui->contracts->clear();
	ui->accounts->setEnabled(true);
	ui->contracts->setEnabled(true);
	try
	{
		for (auto i: ethereum()->addresses(m_block))
		{
			string r = m_main->render(i);
			(new QListWidgetItem(QString("%2: %1 [%3]").arg(formatBalance(ethereum()->balanceAt(i)).c_str()).arg(QString::fromStdString(r)).arg((unsigned)ethereum()->countAt(i)), ethereum()->codeAt(i).empty() ? ui->accounts : ui->contracts))
				->setData(Qt::UserRole, QByteArray((char const*)i.data(), Address::size));
		}
	}
	catch (InterfaceNotSupported const&)
	{
		ui->accounts->setEnabled(false);
		ui->contracts->setEnabled(false);
		ui->json->setEnabled(false);
		ui->json->setText(QString("This feature requires compilation with FATDB support."));
	}
}

void ExportStateDialog::generateJSON()
{
	std::stringstream json;
	json << "{\n";
	std::string prefix;
	for(QListWidgetItem* item: ui->accounts->selectedItems())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		auto address = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		json << prefix << "\t\"" << toHex(address.ref()) << "\": {  \"wei\": \"" << ethereum()->balanceAt(address, m_block) << "\" }";
		prefix = ",\n";
	}
	for(QListWidgetItem* item: ui->contracts->selectedItems())
	{
		auto hba = item->data(Qt::UserRole).toByteArray();
		auto address = Address((byte const*)hba.data(), Address::ConstructFromPointer);
		json << prefix << "\t\"" << toHex(address.ref()) << "\":\n\t{\n\t\t\"wei\": \"" << ethereum()->balanceAt(address, m_block) << "\",\n";
		json << "\t\t\"code\": \"" << toHex(ethereum()->codeAt(address, m_block)) << "\",\n";
		std::unordered_map<u256, u256> storage = ethereum()->storageAt(address, m_block);
		if (!storage.empty())
		{
			json << "\t\t\"storage\":\n\t\t{\n";
			std::string storagePrefix;
			for (auto s: storage)
			{
				json << storagePrefix << "\t\t\t\"" << toHex(s.first) << "\": \"" << toHex(s.second) << "\"";
				storagePrefix = ",\n";
			}
			json << "\n\t\t}\n";
		}
		json << "\t}";
		prefix = ",\n";
	}
	json << "\n}";
	json.flush();

	ui->json->setEnabled(true);
	ui->json->setText(QString::fromStdString(json.str()));
	ui->saveButton->setEnabled(true);
}

void ExportStateDialog::on_saveButton_clicked()
{
	QString fn = QFileDialog::getSaveFileName(this, "Save state", QString(), "JSON Files (*.json)");
	if (!fn.endsWith(".json"))
		fn = fn.append(".json");
	ofstream file(fn.toStdString());
	if (file.is_open())
		file << ui->json->toPlainText().toStdString();
}
