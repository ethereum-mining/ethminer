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

namespace dev
{

namespace p2p { class Host; }

namespace az
{

class Connect: public QDialog
{
	Q_OBJECT

public:
	explicit Connect(QWidget* _parent = 0);
	~Connect();

	/// Populate host chooser with default host entries.
	void setEnvironment(QStringList const& _nodes);

	/// Clear dialogue inputs.
	void reset();

	/// @returns the host string, as chosen or entered by the user. Assumed to be "hostOrIP:port" (:port is optional).
	QString host();
	
	/// @returns the identity of the node, as entered by the user. Assumed to be a 64-character hex string.
	QString nodeId();
	
	/// @returns true if Required is checked by the user, indicating that the host is a required Peer.
	bool required();

private:
	Ui::Connect* ui;
};

}
}
