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
/** @file QContractDefinition.cpp
 * @author Yann yann@ethdev.com
 * @date 2014
 */

#include <QObject>
#include "QContractDefinition.h"
#include "libsolidity/Scanner.h"
#include "libsolidity/Parser.h"
#include "libsolidity/Scanner.h"
#include "libsolidity/NameAndTypeResolver.h"
#include "QContractDefinition.h"
using namespace dev::solidity;
using namespace dev::mix;

std::shared_ptr<QContractDefinition> QContractDefinition::Contract(QString _source)
{
	Parser parser;
	std::shared_ptr<ContractDefinition> contract = parser.parse(std::make_shared<Scanner>(CharStream(_source.toStdString())));
	NameAndTypeResolver resolver({});
	resolver.resolveNamesAndTypes(*contract);
	return std::make_shared<QContractDefinition>(contract);
}

QContractDefinition::QContractDefinition(std::shared_ptr<ContractDefinition> _contract): QBasicNodeDefinition(_contract)
{
	initQFunctions();
}

void QContractDefinition::initQFunctions()
{
	std::vector<FunctionDefinition const*> functions = ((ContractDefinition*)m_dec.get())->getInterfaceFunctions();
	for (unsigned i = 0; i < functions.size(); i++)
	{
		FunctionDefinition* func = (FunctionDefinition*)functions.at(i);
		std::shared_ptr<FunctionDefinition> sharedFunc(func);
		m_functions.append(new QFunctionDefinition(sharedFunc, i));
	}
}
