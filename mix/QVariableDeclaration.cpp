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
/** @file QVariableDeclaration.app
 * @author Yann yann@ethdev.com
 * @author Arkadiy Paronyan arkadiy@ethdev.com
 * @date 2015
 */

#include "QVariableDeclaration.h"
#include <libsolidity/AST.h>
#include "CodeModel.h"

using namespace solidity;

namespace dev
{
namespace mix
{

QVariableDeclaration::QVariableDeclaration(QObject* _parent, ASTPointer<VariableDeclaration> const _v):
	QBasicNodeDefinition(_parent, _v.get()),
	m_type(new QSolidityType(this, CodeModel::nodeType(_v->getType().get())))
{
	m_isIndexed = _v->isIndexed();
}

QVariableDeclaration::QVariableDeclaration(QObject* _parent, std::string const& _name,  SolidityType const& _type, bool _isIndexed):
	QBasicNodeDefinition(_parent, _name),
	m_type(new QSolidityType(_parent, _type))
{
	m_isIndexed = _isIndexed;
}

QVariableDeclaration::QVariableDeclaration(QObject* _parent, std::string const& _name,  solidity::Type const* _type, bool _isIndexed):
	QBasicNodeDefinition(_parent, _name),
	m_type(new QSolidityType(this, CodeModel::nodeType(_type)))
{
	m_isIndexed = _isIndexed;
}

QSolidityType::QSolidityType(QObject* _parent, SolidityType const& _type):
	QObject(_parent),
	m_type(_type)
{
}

QVariantList QSolidityType::members() const
{
	QVariantList members;
	if (m_type.type == Type::Struct)
		for (auto const& structMember: m_type.members)
			members.push_back(QVariant::fromValue(new QVariableDeclaration(parent(), structMember.name.toStdString(), structMember.type)));
	if (m_type.type == Type::Enum)
		for (auto const& enumName: m_type.enumNames)
			members.push_back(QVariant::fromValue(enumName));
	return members;
}

}
}

