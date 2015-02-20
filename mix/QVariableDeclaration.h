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
/** @file QVariableDeclaration.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <QDebug>
#include <QStringList>
#include <libsolidity/AST.h>
#include "QBasicNodeDefinition.h"

#pragma once

namespace dev
{
namespace mix
{

class QVariableDeclaration: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QString type READ type WRITE setType)

public:
	QVariableDeclaration() {}
	QVariableDeclaration(solidity::VariableDeclaration const* _v): QBasicNodeDefinition(_v), m_type(QString::fromStdString(_v->getType()->toString())) {}
	QVariableDeclaration(std::string const& _name, std::string const& _type): QBasicNodeDefinition(_name), m_type(QString::fromStdString(_type)) {}
	QString type() const { return m_type; }
	void setType(QString _type) { m_type = _type; }

private:
	QString m_type;
};

}
}

Q_DECLARE_METATYPE(dev::mix::QVariableDeclaration*)
