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
#include <libsolidity/AST.h>
#include <QVariableDeclaration.h>
#include "QBasicNodeDefinition.h"

namespace dev
{
namespace mix
{

class QFunctionDefinition: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QList<QVariableDeclaration*> parameters READ parameters)
	Q_PROPERTY(int index READ index)

public:
	QFunctionDefinition(solidity::FunctionDefinition const* _f, int _index): QBasicNodeDefinition(_f), m_index(_index), m_functions(_f) { initQParameters(); }
	/// Get all input parameters of this function.
	QList<QVariableDeclaration*> parameters() const { return m_parameters; }
	/// Get all return parameters of this function.
	QList<QVariableDeclaration*> returnParameters() const { return m_returnParameters; }
	/// Get the index of this function on the contract ABI.
	int index() const { return m_index; }

private:
	int m_index;
	solidity::FunctionDefinition const* m_functions;
	QList<QVariableDeclaration*> m_parameters;
	QList<QVariableDeclaration*> m_returnParameters;
	void initQParameters();
};

}
}
