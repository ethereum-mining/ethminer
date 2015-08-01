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
#include <libsolidity/AST.h>
#include "QBasicNodeDefinition.h"
#include "SolidityType.h"

#pragma once

namespace dev
{

namespace solidity
{
class Type;
class VariableDeclaration;
}

namespace mix
{

/// UI wrapper around solidity type
class QSolidityType: public QObject
{
	Q_OBJECT
	Q_PROPERTY(int category READ category CONSTANT) //qml does not support enum properties
	Q_PROPERTY(int size READ size CONSTANT)
	Q_PROPERTY(QString name READ name CONSTANT)
	Q_PROPERTY(QVariantList members READ members CONSTANT)
	Q_PROPERTY(bool array READ array CONSTANT)

public:
	QSolidityType() {}
	QSolidityType(QObject* _parent, SolidityType const& _type);
	using Type = SolidityType::Type;
	enum QmlType //TODO: Q_ENUMS does not support enum forwarding. Keep in sync with SolidityType::Type
	{
		SignedInteger,
		UnsignedInteger,
		Hash,
		Bool,
		Address,
		Bytes,
		String,
		Enum,
		Struct
	};

	Q_ENUMS(QmlType)
	SolidityType const& type() const { return m_type; }
	Type category() const { return m_type.type; }
	int size() const { return m_type.size; }
	QString name() const { return m_type.name; }
	QVariantList members() const;
	bool array() const { return m_type.array; }

private:
	SolidityType m_type;
};

/// UI wrapper around declaration (name + type)
class QVariableDeclaration: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QSolidityType* type READ type CONSTANT)

public:
	QVariableDeclaration() {}
	QVariableDeclaration(QObject* _parent, solidity::ASTPointer<solidity::VariableDeclaration> const _v);
	QVariableDeclaration(QObject* _parent, std::string const& _name,  SolidityType const& _type, bool _isIndexed = false);
	QVariableDeclaration(QObject* _parent, std::string const& _name,  solidity::Type const* _type, bool _isIndexed = false);
	QSolidityType* type() const { return m_type; }
	void setType(QSolidityType* _type) { m_type = _type; }
	bool isIndexed() { return m_isIndexed; }

private:
	QSolidityType* m_type;
	bool m_isIndexed;
};


}
}
