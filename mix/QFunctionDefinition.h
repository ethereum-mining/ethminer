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
/** @file QFunctionDefinition.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#pragma once

#include <QObject>
#include <QQmlListProperty>
#include <libsolidity/AST.h>
#include "QVariableDeclaration.h"
#include "QBasicNodeDefinition.h"

namespace dev
{
namespace mix
{

class QFunctionDefinition: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QQmlListProperty<dev::mix::QVariableDeclaration> parameters READ parameters)

public:
	QFunctionDefinition(){}
	QFunctionDefinition(QObject* _parent): QBasicNodeDefinition(_parent) {}
	QFunctionDefinition(QObject* _parent, solidity::FunctionTypePointer const& _f);
	QFunctionDefinition(QObject* _parent, solidity::ASTPointer<solidity::EventDefinition> const& _f);
	QFunctionDefinition(QObject* _parent, solidity::ASTPointer<solidity::FunctionDefinition> const& _f);
	/// Init members
	void init(dev::solidity::FunctionTypePointer _f);
	/// Get all input parameters of this function.
	QList<QVariableDeclaration*> const& parametersList() const { return m_parameters; }
	/// Get all input parameters of this function as QML property.
	QQmlListProperty<QVariableDeclaration> parameters() const { return QQmlListProperty<QVariableDeclaration>(const_cast<QFunctionDefinition*>(this), const_cast<QFunctionDefinition*>(this)->m_parameters); }
	/// Get all return parameters of this function.
	QList<QVariableDeclaration*> returnParameters() const { return m_returnParameters; }
	/// Get the hash of this function declaration on the contract ABI.
	FixedHash<4> hash() const { return m_hash; }
	/// Get the full hash of this function declaration on the contract ABI.
	FixedHash<32> fullHash() const { return m_fullHash; }
	/// Get the hash of this function declaration on the contract ABI. returns QString
	Q_INVOKABLE QString qhash() const { return QString::fromStdString(m_hash.hex()); }

private:
	int m_index;
	FixedHash<4> m_hash;
	FixedHash<32> m_fullHash;
	QList<QVariableDeclaration*> m_parameters;
	QList<QVariableDeclaration*> m_returnParameters;
	void initQParameters();
};

}
}
