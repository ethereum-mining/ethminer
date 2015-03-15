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
#include <QVariantList>
#include "QBasicNodeDefinition.h"
#include "SolidityType.h"

#pragma once

namespace solidity
{
class Type;
class VariableDeclaration;
}

namespace dev
{
namespace mix
{

class QSolidityType: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int type READ type CONSTANT) //qml does not support enum properties
	Q_PROPERTY(int size READ size CONSTANT)
	Q_PROPERTY(QString name READ name CONSTANT)
	Q_PROPERTY(QVariantList members READ members CONSTANT)

public:
	QSolidityType() {}
	QSolidityType(QObject* _parent, SolidityType const& _type);
	using Type = SolidityType::Type;
	enum QmlType //TODO: arrays and structs
	{
		SignedInteger,
		UnsignedInteger,
		Hash,
		Bool,
		Address,
		String,
		Enum,
		Struct
	};

	Q_ENUMS(QmlType)
	Type type() const { return m_type; }
	int size() const { return m_size; }
	QString name() const { return m_name; }
	QVariantList members() const { return m_members; }

private:
	Type m_type;
	int m_size;
	QString m_name;
	QVariantList m_members;
};

class QVariableDeclaration: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QSolidityType* type READ type CONSTANT)

public:
	QVariableDeclaration() {}
	QVariableDeclaration(QObject* _parent, solidity::VariableDeclaration const* _v);
	QVariableDeclaration(QObject* _parent, std::string const& _name,  SolidityType const& _type);
	QVariableDeclaration(QObject* _parent, std::string const& _name,  solidity::Type const* _type);
	QSolidityType* type() const { return m_type; }
	void setType(QSolidityType* _type) { m_type = _type; }

private:
	QSolidityType* m_type;
};


}
}
