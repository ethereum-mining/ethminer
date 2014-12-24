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
	Q_PROPERTY(int index READ index)

public:
	QFunctionDefinition() {}
	QFunctionDefinition(solidity::FunctionDefinition const* _f, int _index);
	/// Get all input parameters of this function.
	QList<QVariableDeclaration*> const& parametersList() const { return m_parameters; }
	/// Get all input parameters of this function as QML property.
	QQmlListProperty<QVariableDeclaration> parameters() const { return QQmlListProperty<QVariableDeclaration>(const_cast<QFunctionDefinition*>(this), const_cast<QFunctionDefinition*>(this)->m_parameters); }
	/// Get all return parameters of this function.
	QList<QVariableDeclaration*> returnParameters() const { return m_returnParameters; }
	/// Get the index of this function on the contract ABI.
	int index() const { return m_index; }

private:
	int m_index;
	QList<QVariableDeclaration*> m_parameters;
	QList<QVariableDeclaration*> m_returnParameters;
	void initQParameters();
};

}
}
