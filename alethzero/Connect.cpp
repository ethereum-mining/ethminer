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
/** @file Connect.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#include "Connect.h"
#include <libp2p/Host.h>
#include "ui_Connect.h"
using namespace dev;
using namespace az;

Connect::Connect(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::Connect)
{
	ui->setupUi(this);
}

Connect::~Connect()
{
	delete ui;
}

void Connect::setEnvironment(QStringList const& _nodes)
{
	if (ui->host->count() == 0)
		ui->host->addItems(_nodes);
}

void Connect::reset()
{
	ui->nodeId->clear();
	ui->required->setChecked(true);
}

QString Connect::host()
{
	return ui->host->currentText();
}

QString Connect::nodeId()
{
	return ui->nodeId->text();
}

bool Connect::required()
{
	return ui->required->isChecked();
}
