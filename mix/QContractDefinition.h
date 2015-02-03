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
/** @file QContractDefinition.h
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#pragma once

#include <QObject>
#include <QQmlListProperty>
#include <libsolidity/AST.h>
#include "QFunctionDefinition.h"
#include "QBasicNodeDefinition.h"

namespace dev
{
namespace mix
{

class QContractDefinition: public QBasicNodeDefinition
{
	Q_OBJECT
	Q_PROPERTY(QQmlListProperty<dev::mix::QFunctionDefinition> functions READ functions CONSTANT)
	Q_PROPERTY(dev::mix::QFunctionDefinition* constructor READ constructor CONSTANT)

public:
	QContractDefinition() {}
	QContractDefinition(solidity::ContractDefinition const* _contract);
	/// Get all the functions of the contract.
	QQmlListProperty<QFunctionDefinition> functions() const { return QQmlListProperty<QFunctionDefinition>(const_cast<QContractDefinition*>(this), const_cast<QContractDefinition*>(this)->m_functions); }
	/// Get the constructor of the contract.
	QFunctionDefinition* constructor() const { return m_constructor; }
	QList<QFunctionDefinition*> const& functionsList() const { return m_functions; }
	/// Find function by hash, returns nullptr if not found
	QFunctionDefinition* getFunction(dev::FixedHash<4> _hash);
private:
	QList<QFunctionDefinition*> m_functions;
	QFunctionDefinition* m_constructor;
};

}
}

