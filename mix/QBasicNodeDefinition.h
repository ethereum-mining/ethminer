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
/** @file QBasicNodeDefinition.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#pragma once

#include <string>
#include <QObject>
#include <libevmasm/SourceLocation.h>

namespace dev
{

namespace solidity
{
class Declaration;
}

namespace mix
{

class QBasicNodeDefinition: public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString name READ name CONSTANT)

public:
	QBasicNodeDefinition(QObject* _parent = nullptr): QObject(_parent) {}
	~QBasicNodeDefinition() {}
	QBasicNodeDefinition(QObject* _parent, solidity::Declaration const* _d);
	QBasicNodeDefinition(QObject* _parent, std::string const& _name);
	/// Get the name of the node.
	QString name() const { return m_name; }
	dev::SourceLocation const& location() { return m_location; }

private:
	QString m_name;
	dev::SourceLocation m_location;
};

}
}
