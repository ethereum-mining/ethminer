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
/** @file Connect.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2015
 */

#pragma once

#include <QDialog>
#include <QList>

namespace Ui { class Connect; }
namespace dev { namespace p2p { class Host; } }

class Connect : public QDialog
{
	Q_OBJECT

public:
	explicit Connect(QWidget* _parent = 0);
	~Connect();

	void setEnvironment(QStringList const& _nodes);

	/// clear dialogue inputs
	void reset();

	// Form field values:
	
	QString host();
	QString nodeId();
	bool required();

private:
    Ui::Connect* ui;
};
